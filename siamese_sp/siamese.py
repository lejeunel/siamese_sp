import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from siamese_sp.drn_contours import DRNContours


class SuperpixelPooling(nn.Module):
    """
    Given a RAG containing superpixel labels, this layer
    randomly samples couples of connected feature vector
    """
    def __init__(self, n_samples, balanced=False):

        super(SuperpixelPooling, self).__init__()

        self.n_samples = n_samples
        self.balanced = balanced

    def forward(self, x, graphs, label_maps, edges_to_pool=None):

        X = [[] for _ in range(len(x))]
        Y = [[] for _ in range(len(x))]

        for i, (x_, g) in enumerate(zip(x, graphs)):
            if (edges_to_pool is None):
                edges = [(e[0], e[1], g.edges[e]['weight']) for e in g.edges]
                if (self.balanced):
                    e_pos = [e for e in edges if (e[-1] == True)]
                    e_neg = [e for e in edges if (e[-1] == False)]
                    edges = [
                        e_pos[i]
                        for i in np.random.choice(self.n_samples // 2,
                                                  size=self.n_samples // 2,
                                                  replace=False)
                    ]
                    edges += [
                        e_neg[i]
                        for i in np.random.choice(self.n_samples // 2,
                                                  size=self.n_samples // 2,
                                                  replace=False)
                    ]
                else:
                    edges = np.random.choice(g.edges(), self.n_samples)
            else:
                edges = edges_to_pool[i]

            for e in edges:
                X[i].append([
                    x[i, ..., label_maps[i, 0, ...] == e[0]].mean(dim=1),
                    x[i, ..., label_maps[i, 0, ...] == e[1]].mean(dim=1)
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
                 in_channels=3,
                 depth=5,
                 start_filts=64,
                 up_mode='transpose',
                 with_batchnorm=False,
                 balanced=True,
                 merge_mode='concat'):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the 
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(Siamese, self).__init__()

        self.balanced = balanced

        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        self.feat_extr = DRNContours()

        self.sp_pool = SuperpixelPooling(10, self.balanced)

        self.linear1 = nn.Linear(self.feat_extr.upconv3.upsample[0].out_channels,
                                 512)
        self.linear2 = nn.Linear(512, 1)


    def forward(self, x, graphs, label_maps, edges_to_pool=None):

        res = []

        in_shape = x.shape[2:]

        x = self.feat_extr(x)

        x, y = self.sp_pool(x, graphs, label_maps, edges_to_pool)

        # iterate on batch
        for b in range(len(x)):
            res_ = []
            for i in range(len(x[b])):
                x_ = self.linear1(x[b][i])
                res_.append(F.relu(x_))

            res_ = torch.abs(res_[1] - res_[0])
            res_ = self.linear2(res_)

            res.append(res_.squeeze())

        return res, y
