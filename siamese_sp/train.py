from loader import Loader
from imgaug import augmenters as iaa
import logging
from my_augmenters import rescale_augmenter, Normalize
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.optim as optim
from siamese import Siamese
import params
import torch
import datetime
import os
from os.path import join as pjoin
import yaml
from tensorboardX import SummaryWriter
import utils as utls
import tqdm
from skimage.future.graph import show_rag


def main(cfg):

    device = torch.device('cuda' if cfg.cuda else 'cpu')

    model = Siamese(in_channels=3,
                    depth=cfg.depth).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()

    transf = iaa.Sequential([
        iaa.Invert(0.5) if 'Dataset1' in 'Dataset'+cfg.train_dir else iaa.Noop(),
        iaa.SomeOf(3,
                    [iaa.Affine(
                        scale={
                            "x": (1 - cfg.aug_scale,
                                    1 + cfg.aug_scale),
                            "y": (1 - cfg.aug_scale,
                                    1 + cfg.aug_scale)
                        },
                        rotate=(-cfg.aug_rotate,
                                cfg.aug_rotate),
                        shear=(-cfg.aug_shear,
                                cfg.aug_shear)),
                    iaa.AdditiveGaussianNoise(
                        scale=cfg.aug_noise*255),
                    iaa.Fliplr(p=0.5),
                    iaa.Flipud(p=0.5)]),
        iaa.Resize(cfg.in_shape),
        Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        rescale_augmenter])

    dl_train = Loader(pjoin(cfg.in_root, 'Dataset'+cfg.train_dir),
                      augmentation=transf)

    dl_prev = torch.utils.data.ConcatDataset([Loader(pjoin(cfg.in_root,
                                                           'Dataset'+d),
                                                     augmentation=transf)
                                              for d in cfg.prev_dirs])

    dataloader_train = DataLoader(dl_train,
                                  batch_size=cfg.batch_size,
                                  sampler=SubsetRandomSampler(
                                      cfg.n_frames_epoch * cfg.train_frames),
                                  collate_fn=dl_train.collate_fn,
                                  num_workers=cfg.n_workers)

    dataloader_prev = DataLoader(dl_prev,
                                 batch_size=cfg.batch_size,
                                 collate_fn=dl_train.collate_fn,
                                 sampler=torch.utils.data.RandomSampler(
                                     dl_prev,
                                     replacement=True,
                                     num_samples=cfg.batch_size),
                                 num_workers=cfg.n_workers)

    dataloaders = {'train': dataloader_train,
                   'prev': dataloader_prev}

    d = datetime.datetime.now()

    ds_dir = os.path.split('Dataset'+cfg.train_dir)[-1]

    run_dir = pjoin(cfg.out_dir, '{}_{:%Y-%m-%d_%H-%M}_amt'.format(ds_dir, d))

    prev_im_dir = pjoin(run_dir, 'prevs')

    if(not os.path.exists(run_dir)):
        os.makedirs(run_dir)

    if(not os.path.exists(prev_im_dir)):
        os.makedirs(prev_im_dir)

    # Save cfg
    with open(pjoin(run_dir, 'cfg.yml'), 'w') as outfile:
        yaml.dump(cfg.__dict__, stream=outfile, default_flow_style=False)

    writer = SummaryWriter(run_dir)

    # convert batch to device
    batch_to_device = lambda batch: {
        k: v.to(device) if (isinstance(v, torch.Tensor)) else v
        for k, v in batch.items()
    }
    optimizer = optim.Adam(model.parameters(),
                           lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    optimizer = optim.SGD(
        model.parameters(),
        momentum=cfg.momentum,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay)

    utls.setup_logging(run_dir)
    logger = logging.getLogger('unet_region')

    logger.info('run_dir: {}'.format(run_dir))

    best_loss = float('inf')
    for epoch in range(cfg.epochs):

        logger.info('Epoch {}/{}'.format(epoch + 1, cfg.epochs))

        # Each epoch has a training and validation phase
        for phase in dataloaders.keys():
            if phase == 'train':
                model.train()
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            if (phase in ['train', 'prev']):
                # Iterate over data.
                pbar = tqdm.tqdm(total=len(dataloaders[phase]))
                for i, data in enumerate(dataloaders[phase]):
                    data = batch_to_device(data)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    edges_to_pool = None

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        if(phase == 'prev'):
                            edges_to_pool = [[e for e in g.edges] for g in data['graph']]

                        res, y = model(data['image'], data['graph'], data['labels'],
                                       edges_to_pool)

                        if(phase == 'prev'):
                            res_ = torch.cat(res)
                            y_ = torch.cat(res)
                        else:
                            res_ = torch.stack(res)
                            y_ = torch.stack(y)

                        loss = criterion(res_,
                                         y_)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                        else:

                            fig = utls.make_grid_rag(data,
                                                     [torch.sigmoid(r) for r in res])
                            writer.add_figure('test/img', fig, epoch)
                            fig.savefig(pjoin(prev_im_dir, 'prev_{}.png'.format(epoch)),
                                        dpi=200)


                    running_loss += loss.cpu().detach().numpy()
                    loss_ = running_loss / ((i + 1) * cfg.batch_size)
                    pbar.set_description('[{}] loss: {:.4f}'.format(
                        phase, loss_))
                    pbar.update(1)

                pbar.close()
                writer.add_scalar('{}/loss'.format(phase),
                                  loss_,
                                  epoch)

            # save checkpoint
            if phase == 'prev':
                is_best = False
                if (loss_ < best_loss):
                    is_best = True
                    best_loss = loss_
                path = pjoin(run_dir, 'checkpoints')
                utls.save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'model': model,
                        'best_loss': best_loss,
                        'optimizer': optimizer.state_dict()
                    },
                    is_best,
                    path=path)


if __name__ == "__main__":

    p = params.get_params()

    p.add('--out-dir', required=True)
    p.add('--in-root', required=True)
    p.add('--train-dir', required=True)
    p.add('--train-frames', nargs='+', type=int, required=True)
    p.add('--prev-dirs', nargs='+', type=str, required=True)

    p.add('--checkpoint-path', default=None)

    cfg = p.parse_args()

    main(cfg)
