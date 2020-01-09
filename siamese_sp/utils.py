import logging
import os
from os.path import join as pjoin
import yaml
import matplotlib.pyplot as plt
from skimage.future.graph import show_rag
from skimage import segmentation
import numpy as np
import torch
import shutil


def setup_logging(log_path,
                  conf_path='logging.yaml',
                  default_level=logging.INFO,
                  env_key='LOG_CFG'):
    """Setup logging configuration

    """
    path = conf_path

    # Get absolute path to logging.yaml
    path = pjoin(os.path.dirname(__file__), path)
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
            config['handlers']['info_file_handler']['filename'] = pjoin(
                log_path, 'info.log')
            config['handlers']['error_file_handler']['filename'] = pjoin(
                log_path, 'error.log')
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


def make_grid_rag(data, res):

    fig, ax = plt.subplots(len(res), 2)
    ax = ax.reshape(len(res), 2)
    # make preview images
    im = data['image_unnormal'].cpu().numpy()
    im = [np.rollaxis(im[i, ...], 0, 3) for i in range(im.shape[0])]
    truth = data['label/segmentation'].cpu().numpy()
    truth = [truth[i, 0, ...] for i in range(truth.shape[0])]
    labels = data['labels'].cpu().numpy()
    labels = [labels[i, ...][0] for i in range(labels.shape[0])]
    for i, (im_, truth_, labels_,
            g) in enumerate(zip(im, truth, labels, data['rag'])):
        truth_contour = segmentation.find_boundaries(truth_, mode='thick')
        g.add_edges_from(
            (e[0], e[1], dict(weight=res[i][j].detach().cpu().numpy()))
            for j, e in enumerate(g.edges))
        lc = show_rag(labels_.astype(int),
                      g,
                      im_,
                      ax=ax[i, 1],
                      edge_width=0.5,
                      edge_cmap='viridis')
        fig.colorbar(lc, ax=ax[i, 1], fraction=0.03)
        im_[truth_contour, ...] = (1, 0, 0)
        ax[i, 0].imshow(im_)

    return fig

def save_checkpoint(dict_,
                    is_best,
                    path,
                    fname_cp='checkpoint.pth.tar',
                    fname_bm='best_model.pth.tar'):

    cp_path = os.path.join(path, fname_cp)
    bm_path = os.path.join(path, fname_bm)

    if (not os.path.exists(path)):
        os.makedirs(path)

    try:
        state_dict = dict_['model'].module.state_dict()
    except AttributeError:
        state_dict = dict_['model'].state_dict()

    torch.save(state_dict, cp_path)

    if (is_best):
        shutil.copyfile(cp_path, bm_path)

