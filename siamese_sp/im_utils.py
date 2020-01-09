import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.ndimage
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import collections
import os
from skimage import io
from imgaug.augmenters import Augmenter
from multiprocessing import Pool


def make_1d_gauss(length, std, x0):

    x = np.arange(length)
    y = np.exp(-0.5 * ((x - x0) / std)**2)

    return y / np.sum(y)


def make_2d_gauss(shape, std, center):
    """
    Make object prior (gaussians) on center
    """

    g = np.zeros(shape)
    g_x = make_1d_gauss(shape[1], std, center[1])
    g_x = np.tile(g_x, (shape[0], 1))
    g_y = make_1d_gauss(shape[0], std, center[0])
    g_y = np.tile(g_y.reshape(-1, 1), (1, shape[1]))

    g = g_x * g_y

    return g / np.sum(g)


def imread(path, scale=True):
    im = io.imread(path)
    if (scale):
        im = im / 255

    if(im.dtype == 'uint16'):
        im = (im / 255).astype(np.uint8)

    if(len(im.shape) < 3):
        im = np.repeat(im[..., None], 3, -1)

    if (im.shape[-1] > 3):
        im = im[..., 0:3]

    return im


def img_tensor_to_img(tnsr, rescale=False):
    im = tnsr.detach().cpu().numpy().transpose((1, 2, 0))
    if (rescale):
        range_ = im.max() - im.min()
        im = (im - im.min()) / range_
    return im


def one_channel_tensor_to_img(tnsr, rescale=False):
    im = tnsr.detach().cpu().numpy().transpose((1, 2, 0))
    if (rescale):
        range_ = im.max() - im.min()
        im = (im - im.min()) / range_
    im = np.repeat(im, 3, axis=-1)
    return im


def save_tensors(tnsr_list, kind, path, rescale=True):
    """
    tnsr_list: list of tensors
    modes iterable of strings ['image', 'map']
    """

    if (not isinstance(rescale, list)):
        rescale = [True] * len(tnsr_list)

    path_ = os.path.split(path)[0]
    if (not os.path.exists(path_)):
        os.makedirs(path_)

    arr_list = list()
    for i, t in enumerate(tnsr_list):
        if (kind[i] == 'image'):
            arr_list.append(img_tensor_to_img(t, rescale[i]))
        elif (kind[i] == 'map'):
            arr_list.append(one_channel_tensor_to_img(t, rescale[i]))
        else:
            raise Exception("kind must be 'image' or 'map'")

    all = np.concatenate(arr_list, axis=1)
    all = (all * 255).astype(np.uint8)
    io.imsave(path, all)

    return all


def center_crop(x, center_crop_size):
    assert x.ndim == 3
    centerw, centerh = x.shape[1] // 2, x.shape[2] // 2
    halfw, halfh = center_crop_size[0] // 2, center_crop_size[1] // 2
    return x[:, centerw - halfw:centerw + halfw, centerh - halfh:centerh +
             halfh]


def to_np_array(x):
    return x.numpy().transpose((1, 2, 0))


def to_tensor(x):
    import torch

    # copy to avoid "negative stride numpy" problem
    x = x.transpose((2, 0, 1)).copy()
    return torch.from_numpy(x).float()

