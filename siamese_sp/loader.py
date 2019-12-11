import os
from os.path import join as pjoin
from skimage import io, segmentation, measure, future
import glob
import numpy as np
import matplotlib.pyplot as plt
import imgaug as ia
import torch
import networkx as nx
from scipy import sparse

def imread(path, scale=True):
    im = io.imread(path)

    if(im.dtype == 'uint16'):
        im = (im / 255).astype(np.uint8)

    if (scale):
        im = im / 255

    if(len(im.shape) < 3):
        im = np.repeat(im[..., None], 3, -1)

    if (im.shape[-1] > 3):
        im = im[..., 0:3]

    return im



class Loader:
    def __init__(self,
                 root_path,
                 truth_type='hand',
                 ksp_pm_thr=0.8,
                 fix_frames=None,
                 late_fn=None,
                 augmentation=None,
                 n_segments=300,
                 compactness=20.,
                 ignore_empty_truths=True):
        """

        """

        self.root_path = root_path
        self.truth_type = truth_type
        self.ksp_pm_thr = ksp_pm_thr

        self.n_segments = n_segments
        self.compactness = compactness

        self.augmentation = augmentation
        self.late_fn = late_fn

        self.ignore_collate = ['frame_idx', 'frame_name', 'graph']

        exts = ['*.png', '*.jpg', '*.jpeg']
        img_paths = []
        for e in exts:
            img_paths.extend(sorted(glob.glob(pjoin(root_path,
                                           'input-frames',
                                                     e))))
        if(truth_type == 'hand'):
            truth_paths = []
            for e in exts:
                truth_paths.extend(sorted(glob.glob(pjoin(root_path,
                                            'ground_truth-frames',
                                                        e))))
            if(fix_frames is not None):
                self.truth_paths = [truth_paths[i] for i in range(len(truth_paths))
                                    if(i in fix_frames)]
                self.img_paths = [img_paths[i] for i in range(len(img_paths))
                                  if(i in fix_frames)]
            else:
                self.truth_paths = truth_paths
                self.img_paths = img_paths

            self.truths = [
                io.imread(f).astype('bool') for f in self.truth_paths
            ]
            self.truths = [t if(len(t.shape) < 3) else t[..., 0]
                           for t in self.truths]
            self.imgs = [
                imread(f, scale=False) for f in self.img_paths]

        if(ignore_empty_truths):
            is_not_empty = [np.sum(truth) > 0 for truth in self.truths]
            self.truths = [t for i, t in enumerate(self.truths) if is_not_empty[i]]
            self.imgs = [t for i, t in enumerate(self.imgs) if is_not_empty[i]]


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):

        truth = self.truths[idx]
        im = self.imgs[idx]

        if(self.augmentation is not None):
            truth = ia.SegmentationMapsOnImage(
                truth, shape=truth.shape)
            seq_det = self.augmentation.to_deterministic()
            im = seq_det.augment_image(im)
            truth = seq_det.augment_segmentation_maps([truth])[0].get_arr()

        labels = segmentation.slic(im,
                                   n_segments=self.n_segments,
                                   compactness=self.compactness)

        graph = future.graph.RAG(labels)
        y = [((labels == l) * truth).sum()/(labels == l).sum() > 0.5
             for l in np.unique(labels)]
        node_truth = [[l, dict(fg=y_, labels=[l])] for l, y_ in zip(np.unique(labels), y)]
        graph.add_nodes_from(node_truth)
        edges = [(n0, n1, dict(weight=graph.nodes[n0]['fg'] == graph.nodes[n1]['fg']))
                 for n0, n1 in graph.edges()]
        graph.add_edges_from(edges)

        labels = labels[..., None]
        truth = truth[..., None]

        out = {'image': im,
               'frame_idx': idx,
               'frame_name': os.path.split(self.img_paths[idx])[-1],
               'graph': graph,
               'labels': labels,
               'label/segmentation': truth}
        return out


    def collate_fn(self, samples):
        out = {k: [dic[k] for dic in samples] for k in samples[0]}

        for k in out.keys():
            if(k not in self.ignore_collate):
                out[k] = np.array(out[k])
                out[k] = np.rollaxis(out[k], -1, 1)
                out[k] = torch.from_numpy(out[k]).float().squeeze(-1)

        return out

