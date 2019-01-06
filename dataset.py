from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class PCamDataset(Dataset):
    """ Wrapper for Patch Camelyon Dataset """
    def __init__(self, root_dir, mode='train',
                 label_filename=None, transform=None):
        """
        Args:
            root_dir (string): directory with all the images.
            mode (string): dataset mode e.g. train/test
            label_filename (string): path to the csv file with annotations.
            transform (callable, optional): optional transform to be applied
                                            on a sample.
        """
        self.data_dir = Path(root_dir, mode)
        filepaths = [f for f in self.data_dir.iterdir() if 'tif' in f.name]
        if label_filename:
            label_df = pd.read_csv(label_filename) \
                         .set_index('id', verify_integrity=True)
            self.labels = [int(label_df.loc[f.stem, 'label'])
                           for f in filepaths]
            self.labels = tuple(self.labels)
        else:
            self.labels = None
        self.filepaths = tuple(filepaths)
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filename = self.filepaths[idx]
        if self.labels:
            label = self.labels[idx]
        else:
            label = -1
        sample = Image.open(filename)
        if self.transform:
            sample = self.transform(sample)
            label = torch.FloatTensor([label])
        return sample, label


def calculate_statistics(dset, batch_size=1, num_workers=1):
    dloader = DataLoader(dset, batch_size=batch_size,
                         shuffle=False, num_workers=num_workers)
    x, _ = dloader.dataset[0]
    num_channels = x.size(0)
    mean, std = torch.zeros(num_channels), torch.zeros(num_channels)
    for i, (x, _) in enumerate(dloader, start=0):
        x = x.permute(1, 0, 2, 3).contiguous().view(x.size(1), -1)
        mean.add_(x.mean(dim=1))
        std.add_(x.std(dim=1))
    mean.div_(i+1)
    std.div_(i+1)
    return mean.tolist(), std.tolist()
