from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

def dataloader(path, batch_size, do_shuffle = True):
    transform = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )
    dataset = datasets.DatasetFolder(path, loader=custom_loader, extensions='.npy', transform=transform)
    loader = DataLoader(dataset, shuffle=do_shuffle, batch_size=batch_size, num_workers=0)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)
        except StopIteration:
            dataset = datasets.DatasetFolder(path, loader=custom_loader, extensions='.npy', transform=transform)
            loader = DataLoader(dataset, shuffle=do_shuffle, batch_size=batch_size, num_workers=0)
            loader = iter(loader)
            yield next(loader)

def custom_loader(path):
    image = np.load(path)
    image = 1000*((image - 0.0194) / 0.0194)
    image[image<-1000] = -1000
    image = image / 4000
    return image

