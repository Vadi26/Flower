
import torch
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST

def get_mnist(data_path: str = './data'):

    # Normalize ((Variance,), (stddev,))
    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    trainset = MNIST(data_path, train=True, download=True, transform=tr)
    testset = MNIST(data_path, train=False, download=True, transform=tr)

    return trainset, testset

def prepare_dataset(num_partitions: int, 
                    batch_size: int,
                    val_ratio: float=0.1):
    
    # Downloads the dataset
    trainset, testset = get_mnist()

    # Split trainset into 'num_partitions' trainsets so that we can assign a portion of the dataset to every client
    num_images = len(trainset) // num_partitions
    partition_len = [num_images] * num_partitions
    trainsets = random_split(trainset, partition_len, torch.Generator().manual_seed(2023))

    # Create dataloaders with train + val support
    # We split the created trainsets into trainset and valset
    trainloaders = []
    valloaders = []
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(trainset_, [num_train, num_val], torch.Generator().manual_seed(2023))

        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=True, num_workers=2))
    
    testloader = DataLoader(testset, batch_size=128)

    return trainloaders, valloaders, testloader
