import random
import itertools

import numpy as np
from sklearn.preprocessing import LabelEncoder
import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def get_disentanglement_dataset(
        dataset: str,
        train: bool,
        batch_size: int,
        **kwargs
    ):
    datasets = {"DSPRITES": DSprites, "MPI3D": MPI3D, "SHAPES3D": Shapes3D}
    dataset = datasets[dataset.upper()](train=train, **kwargs)
    if train:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model_kwargs = {
        'n_concepts': dataset.n_concepts,
        'dim_y': dataset.n_classes_task,
        'dim_c': dataset.dim_c,
        'continuous_c': False,
        'continuous_y': False,
        'ch_in': dataset.n_channels,
        'image_size': 64,
        'imbalance_ratio': dataset.imbalance_ratio,
    }
    attr_groups = [[i] for i in range(dataset.n_concepts)]
    random.shuffle(attr_groups)
    return dataloader, model_kwargs, attr_groups


class DisentanglementDataset(Dataset):

    def __init__(
            self,
            filepath,
            task,
            concepts,
            nuisances_task,
            n_samples=None,
            train=True,
            seed=42,
            flatten=False,
            return_nuisances=False,
            binarize_concepts=True,
            **kwargs
        ):
        
        self.all_factors = list(self.factors_nvalues.keys())

        self.task = task
        self.concepts = concepts
        self.nuisances_task = nuisances_task
        self.n_samples = n_samples
        self.train = train
        self.seed = seed
        self.flatten = flatten
        self.return_nuisances = return_nuisances
        self.binarize_concepts = binarize_concepts

        if binarize_concepts:
            self.n_concepts = sum([self.factors_nvalues[k] for k in concepts])
            self.dim_c = 1
        else:
            self.n_concepts = len(concepts)
            self.dim_c = [self.factors_nvalues[k] for k in concepts]
        self.n_classes_task = self.factors_nvalues[task]
        self.nuisances_nontask = list(set(self.all_factors) - \
            set([task] + concepts + nuisances_task))
        self._read_dataset(filepath)
        self._filter_samples_nuisances()
        self._select_samples()
        self._find_imbalance()
        self._precompute_labels()


    def _precompute_labels(self):
        """Precompute task/concept/nuisance tensors for the whole (filtered)
        set once, so __getitem__ is a fast slice instead of running F.one_hot
        per sample (the dominant data-loading cost for these datasets)."""
        labels = torch.as_tensor(self.labels, dtype=torch.long)

        def cols(names, one_hot):
            out = []
            for name in names:
                idx = self.all_factors.index(name)
                if one_hot:
                    out.append(F.one_hot(labels[:, idx],
                                         self.factors_nvalues[name]).float())
                else:
                    out.append(labels[:, idx:idx+1].float())
            return torch.cat(out, dim=1) if out else None

        self._task_t = labels[:, self.all_factors.index(self.task)]
        self._concepts_t = cols(self.concepts, self.binarize_concepts)
        self._nuis_task_t = cols(self.nuisances_task, one_hot=False)
        self._nuis_nontask_t = cols(self.nuisances_nontask, one_hot=False)


    def _read_dataset(self, npz_path):
        dataset = np.load(npz_path, allow_pickle=True)
        self.images = dataset['imgs'][:,None]
        self.labels = dataset['latents_classes'][:,1:]


    def _filter_samples_nuisances(self):
        random.seed(self.seed)
        def split_into_groups(data, n):
            k, r = divmod(len(data), n)
            groups = [data[i * k + min(i, r):(i + 1) * k + min(i + 1, r)] for i in range(n)]
            return groups
        
        factors_task = self.concepts + self.nuisances_task
        sets = [range(self.factors_nvalues[k]) for k in factors_task]
        combinations = list(itertools.product(*sets))
        random.shuffle(combinations)
        groups = split_into_groups(combinations, self.factors_nvalues[self.task])
        groups_task = [[i] + list(group) for i in range(self.factors_nvalues[self.task])\
            for group in groups[i]]

        idxs = [self.all_factors.index(i) for i in [self.task]+factors_task]
        #matches = (self.labels[:,idxs][:, None] == np.array(groups_task)).all(axis=2)
        #keep = list(matches.any(axis=1))

        def find_row_indices(A, B):
            A_view = np.core.records.fromarrays(A.T)
            B_view = np.core.records.fromarrays(B.T)
            return np.nonzero(np.in1d(A_view, B_view))[0]
        keep = find_row_indices(self.labels[:,idxs], np.array(groups_task))

        self.images = self.images[keep]
        self.labels = self.labels[keep]
    

    def _select_samples(self):
        if self.n_samples is None or self.n_samples>self.images.shape[0]:
            self.n_samples = self.images.shape[0]

        idxs = random.sample(range(self.images.shape[0]), self.n_samples)
        train_samples = self.n_samples*75//100
        idxs = idxs[:train_samples] if self.train else idxs[train_samples:]

        self.images = self.images[idxs]
        self.labels = self.labels[idxs]


    def _find_imbalance(self):
        # Vectorized equivalent of stacking per-sample concept vectors — the
        # original per-sample Python loop cost minutes on the large filtered
        # sets. One-hot each concept factor column at once, then concatenate.
        labels = torch.as_tensor(self.labels, dtype=torch.long)
        cols = []
        for name in self.concepts:
            idx = self.all_factors.index(name)
            if self.binarize_concepts:
                cols.append(F.one_hot(labels[:, idx],
                                      self.factors_nvalues[name]).float())
            else:
                cols.append(labels[:, idx:idx+1].float())
        concepts = torch.cat(cols, dim=1)
        self.imbalance_ratio = concepts.shape[0] / torch.sum(concepts, axis=0) - 1


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        image = torch.as_tensor(self.images[idx], dtype=torch.float32) / 255.
        if self.flatten:
            image = image.flatten()
        task = self._task_t[idx]
        concepts = self._concepts_t[idx]
        if self.return_nuisances:
            nuisances_task = self._nuis_task_t[idx] if self._nuis_task_t is not None \
                else torch.tensor([])
            nuisances_nontask = self._nuis_nontask_t[idx] if self._nuis_nontask_t is not None \
                else torch.tensor([])
            return image, task, concepts, nuisances_task, nuisances_nontask
        else:
            return image, task, concepts
    

    def _get_label(self, factors, set_labels, one_hot=True):
        if one_hot:
            return torch.cat([
                F.one_hot(
                    factors[self.all_factors.index(i)], 
                    self.factors_nvalues[i]
                ) for i in set_labels
            ]).float()
        else:
            return torch.stack([
                factors[self.all_factors.index(i)] for i in set_labels
            ]).float()
    


class DSprites(DisentanglementDataset):

    factors_nvalues = {
        'shape': 3, 
        'scale': 6, 
        'orientation': 40, 
        'posX': 32, 
        'posY': 32,
    }
    n_channels = 1

    def _read_dataset(self, npz_path):
        dataset = np.load(npz_path, allow_pickle=True)
        self.images = dataset['imgs'][:,None]
        self.labels = dataset['latents_classes'][:,1:]



class MPI3D(DisentanglementDataset):

    factors_nvalues = {
        'object_color': 6, 
        'object_shape': 6, 
        'object_size': 2, 
        'camera_height': 3, 
        'background_color': 3, 
        'horizontal_axis': 40, 
        'vertical_axis': 40
    }
    n_channels = 3

    def _read_dataset(self, npz_path):
        dataset = np.load(npz_path, allow_pickle=True)
        images = dataset['images']
        # MPI3D real images are stored HWC (N,64,64,3); the encoder expects
        # CHW. Keep the transpose as a view (no full-array copy) — the later
        # per-nuisance filtering (self.images[keep]) materializes a contiguous
        # subset, so we avoid duplicating the full 12.7GB array in RAM.
        if images.ndim == 4 and images.shape[-1] == 3:
            images = images.transpose(0, 3, 1, 2)
        self.images = images
        # F.one_hot / indexing require integer index tensors.
        self.labels = dataset['labels'].astype(np.int64)



class Shapes3D(DisentanglementDataset):

    factors_nvalues = {
        'floor_hue': 10, 
        'wall_hue': 10, 
        'object_hue': 10, 
        'scale': 8, 
        'shape': 4, 
        'orientation': 15
    }    
    n_channels = 3

    def _read_dataset(self, h5_path):
        dataset = h5py.File(h5_path, 'r')
        self.images = np.array(dataset['images']).transpose(0, 3, 1, 2)
        labels = np.array(dataset['labels'])
        encoded_labels = np.zeros_like(labels, dtype=int)
        for i in range(labels.shape[1]):
            le = LabelEncoder()
            encoded_labels[:, i] = le.fit_transform(labels[:, i])
        self.labels = encoded_labels