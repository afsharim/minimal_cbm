import os
import random
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_cifar10(
        train,
        batch_size: int,
        resampling: bool=True,
        **kwargs
    ):

    if train:
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
            transforms.Resize(size=(224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    dataset = CIFAR10(train=train, transform=transform, **kwargs)
    # Worker/prefetch counts are configurable via env so several CIFAR runs can
    # share the host without exhausting RAM (each worker holds prefetched 224^2
    # batches). These do not affect the seeded data order, so results are
    # unchanged. Defaults are reduced from the original 8 workers / prefetch 4.
    n_workers = int(os.environ.get("MCBM_CIFAR_WORKERS", "3"))
    prefetch = int(os.environ.get("MCBM_CIFAR_PREFETCH", "2"))
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True if train and not resampling else False,
        'drop_last': True if train and resampling else False,
        'pin_memory': True,
        'num_workers': min(n_workers, os.cpu_count()),
        'persistent_workers': True if n_workers > 0 else False,
        'prefetch_factor': prefetch if os.cpu_count() > 1 and n_workers > 0 else None,
    }
    dataloader = DataLoader(dataset, **loader_kwargs)
    model_kwargs = {
        'n_concepts': dataset.n_concepts,
        'dim_y': 10,
        'dim_c': 1,
        'continuous_y': False,
        'continuous_c': False,
        'ch_in': 3,
        'image_size': 32,
        'imbalance_ratio': None,
        #'imbalance_ratio': None,
    }
    attr_groups = dataset.concepts_groups
    return dataloader, model_kwargs, attr_groups


class CIFAR10(datasets.CIFAR10):

    def __init__(
            self,
            train: bool,
            root: str,
            n_concepts: int,
            seed: int=42,
            return_nuisances: bool=False,
            intervention_stage: bool=False,
            **kwargs
        ):
        super().__init__(train=train, root=root, **kwargs)

        self.n_concepts = n_concepts
        self.return_nuisances = return_nuisances
        self.seed = seed
        self.concepts_groups = [[i] for i in range(n_concepts)]
        if train:
            self.attributes = torch.load(os.path.join(root, "cifar10_train_concept_labels.pt")) * 1
        else:
            self.attributes = torch.load(os.path.join(root, "cifar10_test_concept_labels.pt")) * 1
        self.n_attributes = self.attributes.shape[1]

        self._set_concepts_nuisances()
        

    def __getitem__(self, idx):
        image, task = super().__getitem__(idx)
        concepts = self.concepts[idx].float()
        if self.return_nuisances:
            nuisances_task = self.nuisances_task[idx].float()
            nuisances_nontask = torch.tensor([]).float()
            return image, task, concepts, nuisances_task, nuisances_nontask
        else:
            return image, task, concepts


    def _set_concepts_nuisances(self):
        random.seed(self.seed)
        concepts_idxs = random.sample(range(self.n_attributes), self.n_concepts)
        nuisances_idxs = list(set(range(self.n_attributes)) - set(concepts_idxs))
        self.concepts = self.attributes[:, concepts_idxs]
        self.nuisances_task = self.attributes[:, nuisances_idxs]
        self.concepts_groups = [[c] for c in concepts_idxs]