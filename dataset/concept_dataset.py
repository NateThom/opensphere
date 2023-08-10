import os.path as osp

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from .utils import image_pipeline


class ConceptDataset(Dataset):
    def __init__(self, name, data_dir, ann_path, test_mode=True):
        super().__init__()

        self.name = name
        self.data_dir = data_dir
        self.ann_path = ann_path
        self.test_mode = test_mode

        self.get_data()

    def get_data(self):
        """Get data from an annotation file.
        """
        with open(self.ann_path, 'r') as f:
            lines = f.readlines()

        paths = set()
        for line in lines:
            path1 = line.rstrip()
            paths.add(path1)
        paths = list(paths)
        paths.sort()
        self.data_items = [{'path': path} for path in paths]

        if len(self.data_items) == 0:
            raise (RuntimeError('Found 0 files.'))

    def prepare(self, idx):
        # load image and pre-process (pipeline) from path
        path = self.data_items[idx]['path']
        item = {'path': osp.join(self.data_dir, path)}
        image = image_pipeline(item, self.test_mode)

        return image

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        return self.prepare(idx)
