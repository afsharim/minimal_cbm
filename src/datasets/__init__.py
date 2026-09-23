def get_loader(dataset, **kwargs):
    if dataset.upper()=="CUB200":
        from .cub200 import get_cub200
        return get_cub200(**kwargs)
    elif dataset.upper()=="CIFAR10":
        from .cifar10 import get_cifar10
        return get_cifar10(**kwargs)
    elif dataset.upper() in ["DSPRITES", "MPI3D", "SHAPES3D"]:
        from .disentanglement import get_disentanglement_dataset
        return get_disentanglement_dataset(dataset, **kwargs)
    elif dataset.upper()=="AWA2":
        from .awa2 import get_awa2
        return get_awa2(**kwargs)
    elif dataset.upper()=="CELEBA":
        from .celeba import get_celeba
        return get_celeba(**kwargs)
    elif dataset.upper()=="SPIRALS":
        from .spirals import get_spirals
        return get_spirals(**kwargs)
    else:
        raise ValueError