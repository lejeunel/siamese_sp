import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import operator
from sklearn.utils.class_weight import compute_class_weight
from siamese_sp.my_augmenters import rescale_augmenter
from siamese_sp import utils as utls
from siamese_sp.loader import Loader
from torch.utils.data import DataLoader, SubsetRandomSampler, RandomSampler
from imgaug import augmenters as iaa
from itertools import chain


def compute_weights(edges):
    edges_ = np.array(edges)
    labels = np.unique(edges_[:, -1])
    weights = compute_class_weight('balanced', labels, edges_[:, -1])

    weights_ = np.zeros(edges_.shape[0])
    if (labels.size < 2):
        weights_[edges_[:, -1] == labels[0]] = weights[0]
    else:
        weights_[edges_[:, -1] == 0] = weights[0]
        weights_[edges_[:, -1] == 1] = weights[1]

    return weights_ / weights_.sum()


class SuperpixelPooling(nn.Module):
    """
    Given a RAG containing superpixel labels, this layer
    randomly samples couples of connected feature vector
    """
    def __init__(self, n_samples, balanced=False, use_max=False):

        super(SuperpixelPooling, self).__init__()

        self.n_samples = n_samples
        self.balanced = balanced
        self.use_max = use_max
        if (use_max):
            self.pooling = lambda x: torch.max(x, dim=1)
        else:
            self.pooling = lambda x: torch.mean(x, dim=1)

    def pool(self, x, dim):
        if(self.use_max):
            return x.max(dim=dim)[0]

        return x.mean(dim=dim)

    def forward(self, x, graphs, label_maps, edges_to_pool=None):

        X = [[] for _ in range(len(x))]
        Y = [[] for _ in range(len(x))]

        for i, (x_, g) in enumerate(zip(x, graphs)):
            if (edges_to_pool is None):
                edges = [(e[0], e[1], g.edges[e]['weight']) for e in g.edges]
                if (self.balanced):
                    weights = compute_weights(edges)
                else:
                    weights = None
                edges = [
                    edges[i] for i in np.random.choice(
                        len(edges), self.n_samples, p=weights)
                ]
            else:
                edges = edges_to_pool[i]

            for e in edges:
                X[i].append([
                    self.pool(x[i, ..., label_maps[i, 0, ...] == e[0]], dim=1),
                    self.pool(x[i, ..., label_maps[i, 0, ...] == e[1]], dim=1)
                ])
                Y[i].append([torch.tensor(e[-1]).float().to(x)])

        X = [(torch.stack([x_[0] for x_ in x],
                          dim=0), torch.stack([x_[1] for x_ in x], dim=0))
             for x in X]
        Y = [torch.stack([y_[0] for y_ in y], dim=0) for y in Y]
        return X, Y

        

class Siamese(nn.Module):
    """

    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """
    def __init__(self,
                 autoenc,
                 in_channels=3,
                 start_filts=64,
                 n_edges=100,
                 sp_pool_use_max=False,
                 with_batchnorm=False,
                 balanced=True):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
        """
        super(Siamese, self).__init__()

        self.balanced = balanced
        self.n_edges = n_edges

        self.in_channels = in_channels

        self.autoenc = autoenc
        self.sigmoid = nn.Sigmoid()

        # freeze weights of feature extractor (encoder + ASPP)
        # for param in chain(self.autoenc.encoder.parameters(),
        #                    self.autoenc.aspp.parameters()):
        #     param.requires_grad = False

        self.sp_pool = SuperpixelPooling(self.n_edges,
                                         self.balanced,
                                         use_max=sp_pool_use_max)

        feats_dim = self.autoenc.feats_dim

        self.linear1 = nn.Linear(
            feats_dim, feats_dim // 2)
        self.linear2 = nn.Linear(feats_dim // 2, 1)

    def calc_probas(self, x):
        res = []

        # iterate on batch
        for b in range(len(x)):
            res_ = []
            for i in range(len(x[b])):
                x_ = self.linear1(x[b][i])
                res_.append(F.relu(x_))

            res_ = torch.abs(res_[1] - res_[0])
            res_ = self.linear2(res_)
            res_ = self.sigmoid(res_)

            res.append(res_.squeeze())

        return res

    def forward(self, x, graphs, label_maps, edges_to_pool=None):

        input_shape = x.shape[-2:]

        x_tilde, feats = self.autoenc(x)

        pooled_feats, y = self.sp_pool(feats, graphs, label_maps, edges_to_pool)

        edge_probas = self.calc_probas(pooled_feats)

        return {'similarities': edge_probas,
                'similarities_labels': y,
                'feats': feats,
                'recons': x_tilde}


if __name__ == "__main__":
    path = '/home/ubelix/lejeune/data/medical-labeling/Dataset30/'
    transf = iaa.Sequential([
        rescale_augmenter])

    dl = Loader(path, augmentation=transf)

    dataloader_prev = DataLoader(dl,
                                batch_size=2,
                                shuffle=True,
                                collate_fn=dl.collate_fn,
                                num_workers=0)

    model = Siamese()

    for data in dataloader_prev:

        edges_to_pool = [[e for e in g.edges] for g in data['rag']]
        res = model(data['image'], data['rag'], data['labels'],
                    edges_to_pool)
        fig = utls.make_grid_rag(data, [F.sigmoid(r) for r in res['similarities_labels']])

        # fig.show()
        fig.savefig('test.png', dpi=200)
        break
