import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


def get_awa2(
        train: bool,
        batch_size: int,
        root: str,
        img_size: int=224,
        num_workers: int=4,
        **kwargs
    ):
    if train:
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    dataset = AwA2(train=train, root=root, transform=transform, **kwargs)
    if train:
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=num_workers, pin_memory=True)
    else:
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=num_workers, pin_memory=True)
    model_kwargs = {
        'n_concepts': dataset.n_concepts,
        'dim_y': 50,
        'dim_c': 1,
        'continuous_y': False,
        'continuous_c': False,
        'ch_in': 3,
        'image_size': img_size,
        'imbalance_ratio': dataset.imbalance_ratio,
    }
    attr_groups = dataset.concepts_groups
    return dataloader, model_kwargs, attr_groups


class AwA2(Dataset):
    """AwA2 with the MCBM-paper task/concept/nuisance partition.

    y: animal class (50). c: first 20 attributes of predicates.txt
    (appearance + morphology). n_y: remaining 65 attributes. n_ybar: empty.
    Attributes are class-level (Xian et al. 2017 original_att binarized at
    0.5). Fixed train/test split prepared by scripts/prepare_awa2.py.
    """

    def __init__(
            self,
            train: bool,
            root: str,
            n_concepts: int=20,
            seed: int=42,
            return_nuisances: bool=False,
            intervention_stage: bool=False,
            transform=None,
        ) -> None:

        self.train = train
        self.root = root
        self.n_concepts = n_concepts
        self.return_nuisances = return_nuisances
        self.transform = transform

        list_file = os.path.join(
            root, "mcbm_awa2", "train_list.txt" if train else "test_list.txt")
        self.samples = []
        with open(list_file) as f:
            for line in f.read().strip().split("\n"):
                rel, lab = line.split("\t")
                self.samples.append((rel, int(lab)))
        self.imgs_root = os.path.join(
            root, "AwA2-data/Animals_with_Attributes2/JPEGImages")

        class_attr = np.load(os.path.join(root, "mcbm_awa2", "class_attr_bin.npy"))
        self.class_concepts = torch.tensor(
            class_attr[:, :n_concepts], dtype=torch.float32)
        self.class_nuisances = torch.tensor(
            class_attr[:, n_concepts:], dtype=torch.float32)
        self.concepts_groups = [[i] for i in range(n_concepts)]

        labels = np.array([lab for _, lab in self.samples])
        concepts = class_attr[labels, :n_concepts]
        n = concepts.shape[0]
        n_pos = concepts.sum(axis=0)
        self.imbalance_ratio = [float(n / p - 1) if p > 0 else 1.0 for p in n_pos]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel, lab = self.samples[idx]
        image = Image.open(os.path.join(self.imgs_root, rel)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        concepts = self.class_concepts[lab]
        if self.return_nuisances:
            nuisances_task = self.class_nuisances[lab]
            nuisances_nontask = torch.tensor([]).float()
            return image, lab, concepts, nuisances_task, nuisances_nontask
        else:
            return image, lab, concepts
