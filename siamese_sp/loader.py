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
import itertools


def imread(path, scale=True):
    im = io.imread(path)

    if (im.dtype == 'uint16'):
        im = (im / 255).astype(np.uint8)

    if (scale):
        im = im / 255

    if (len(im.shape) < 3):
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
                 normalization=None,
                 n_segments=300,
                 delta_segments=0,
                 compactness=20.,
                 nn_radius=0.1):
        """

        """

        self.root_path = root_path
        self.truth_type = truth_type
        self.ksp_pm_thr = ksp_pm_thr

        self.n_segments = n_segments
        self.delta_segments = delta_segments
        self.compactness = compactness

        self.nn_radius = nn_radius

        self.augmentation = augmentation
        self.normalization = normalization

        self.late_fn = late_fn

        self.do_siam_data = True

        self.ignore_collate = [
            'frame_idx', 'frame_name', 'rag', 'nn_graph', 'centroids'
        ]

        exts = ['*.png', '*.jpg', '*.jpeg']
        img_paths = []
        for e in exts:
            img_paths.extend(
                sorted(glob.glob(pjoin(root_path, 'input-frames', e))))
        if (truth_type == 'hand'):
            truth_paths = []
            for e in exts:
                truth_paths.extend(
                    sorted(
                        glob.glob(pjoin(root_path, 'ground_truth-frames', e))))
            if (fix_frames is not None):
                self.truth_paths = [
                    truth_paths[i] for i in range(len(truth_paths))
                    if (i in fix_frames)
                ]
                self.img_paths = [
                    img_paths[i] for i in range(len(img_paths))
                    if (i in fix_frames)
                ]
            else:
                self.truth_paths = truth_paths
                self.img_paths = img_paths

            self.truths = [
                io.imread(f).astype('bool') for f in self.truth_paths
            ]
            self.truths = [
                t if (len(t.shape) < 3) else t[..., 0] for t in self.truths
            ]
            self.imgs = [imread(f, scale=False) for f in self.img_paths]

    def get_fnames(self):
        return [os.path.split(p)[-1] for p in self.img_paths]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):

        truth = self.truths[idx]
        im = self.imgs[idx]

        if (self.augmentation is not None):
            truth = ia.SegmentationMapsOnImage(truth, shape=truth.shape)
            seq_det = self.augmentation.to_deterministic()
            im = seq_det.augment_image(im)
            truth = seq_det.augment_segmentation_maps([truth])[0].get_arr()

        if (self.delta_segments > 0):
            n_segments = np.random.randint(
                low=self.n_segments - self.delta_segments,
                high=self.n_segments + self.delta_segments)
        else:
            n_segments = self.n_segments

        labels = segmentation.slic(im,
                                   n_segments=n_segments,
                                   compactness=self.compactness)

        im_unnormal = im.copy()

        if (self.normalization is not None):
            im = self.normalization.augment_image(im)

        if (self.do_siam_data):

            regions = measure.regionprops(labels + 1, intensity_image=truth)
            node_list = [[
                p['label'] - 1,
                dict(fg=p['mean_intensity'] > 0.5,
                     labels=[p['label'] - 1],
                     centroid=(p['centroid'][1] / labels.shape[1],
                               p['centroid'][0] / labels.shape[0]))
            ] for p in regions]

            centroids = [(p['centroid'][1] / labels.shape[1],
                          p['centroid'][0] / labels.shape[0]) for p in regions]

            # region adjancency graph
            rag = future.graph.RAG(labels)
            rag.add_nodes_from(node_list)
            edges = [(n0, n1,
                      dict(weight=rag.nodes[n0]['fg'] == rag.nodes[n1]['fg']))
                     for n0, n1 in rag.edges()]
            rag.add_edges_from(edges)

            # nearest neighbor graph
            nn_graph = nx.Graph()
            nn_graph.add_nodes_from(node_list)
            node_label_list = [n[0] for n in node_list]
            nodes_ = np.array(np.meshgrid(node_label_list,
                                          node_label_list)).T.reshape(-1, 2)
            centroids_x = [n[1]['centroid'][0] for n in node_list]
            centroids_x = np.array(np.meshgrid(centroids_x,
                                               centroids_x)).T.reshape(-1, 2)
            centroids_y = [n[1]['centroid'][1] for n in node_list]
            centroids_y = np.array(np.meshgrid(centroids_y,
                                               centroids_y)).T.reshape(-1, 2)
            centroids_ = np.concatenate((centroids_x, centroids_y), axis=1)

            dists = np.sqrt((centroids_[:, 0] - centroids_[:, 1])**2 +
                            (centroids_[:, 2] - centroids_[:, 3])**2)
            inds = np.argwhere(dists < self.nn_radius).ravel()

            edges = [(nodes_[i, 0], nodes_[i, 1],
                      dict(weight=nn_graph.nodes[nodes_[i, 0]]['fg'] ==
                           nn_graph.nodes[nodes_[i, 1]]['fg'])) for i in inds]
            nn_graph.add_edges_from(edges)

            labels = labels[..., None]
        else:
            rag = None
            centroids = None
            nn_graph = None
            labels = np.zeros_like(truth)
        truth = truth[..., None]

        out = {
            'image': im,
            'image_unnormal': im_unnormal,
            'frame_idx': idx,
            'frame_name': os.path.split(self.img_paths[idx])[-1],
            'rag': rag,
            'centroids': centroids,
            'nn_graph': nn_graph,
            'labels': labels,
            'label/segmentation': truth
        }
        return out

    def collate_fn(self, samples):
        out = {k: [dic[k] for dic in samples] for k in samples[0]}

        for k in out.keys():
            if (k not in self.ignore_collate):
                out[k] = np.array(out[k])
                out[k] = np.rollaxis(out[k], -1, 1)
                out[k] = torch.from_numpy(out[k]).float().squeeze(-1)

        return out
