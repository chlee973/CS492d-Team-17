import os
from multiprocessing.pool import Pool
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import h5py

import torch

def pen_state_to_binary(x):
    # x: [B, C, 3]
    result = x.clone()
    pen_state_probs = result[:, :, 2]
    binary_values = (pen_state_probs > 0.5).float()
    result[:, :, 2] = binary_values
    return result

def tensor_to_pil_image(tensor: torch.Tensor, canvas_size=(256, 256), padding=30):
    # tensor: [C, 3]
    assert tensor.ndim == 2 and tensor.shape[1] == 3
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = tensor.numpy()

    width, height = canvas_size

    sketch = scale_sketch(tensor, (width-padding, height-padding))  
    [start_x, start_y, _, _] = sketch_size(sketch=sketch)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    current_x, current_y = start_x + padding//2, start_y + padding//2

    for dx, dy, pen_state in sketch:
        next_x = current_x + dx
        next_y = current_y + dy

        if pen_state == 1:
            draw.line([current_x, current_y, next_x, next_y], fill="black", width=1)
        else:
            draw.line([current_x, current_y, next_x, next_y], fill="gray", width=1)
        current_x, current_y = next_x, next_y

    return image

def sketch_size(sketch):
    vertical_sum = np.cumsum(sketch[1:], axis=0)  

    xmin, ymin, _ = np.min(vertical_sum, axis=0)
    xmax, ymax, _ = np.max(vertical_sum, axis=0)

    w = xmax - xmin
    h = ymax - ymin
    start_x = -xmin - sketch[0][0]  
    start_y = -ymin - sketch[0][1]
    return [int(start_x), int(start_y), h, w]

def scale_sketch(sketch, size=(256, 256)):
    [_, _, h, w] = sketch_size(sketch)
    assert h !=0 and w != 0
    if h >= w:
        sketch_normalize = sketch / np.array([[h, h, 1]], dtype=float)
    else:
        sketch_normalize = sketch / np.array([[w, w, 1]], dtype=float)
    sketch_rescale = sketch_normalize * np.array([[size[0], size[1], 1]], dtype=float)
    return sketch_rescale.astype("int16")


def get_data_iterator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
    for i, data in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


class SketchDataset(Dataset):
    def __init__(self, data_path, categories, split, Nmax, label_offset = 1):
        """
        Args:
            h5_file (str): Path to the HDF5 file.
            categories (list of str): List of category names.
            split (str): Data split, e.g., 'train' or 'test'.
            Namx (int): maximum number of vectors.
            label_offset: Offset which will be added to category label.
        """
        self.data_path = data_path
        self.categories = categories
        self.split = split
        self.Nmax = Nmax
        self.label_offset = label_offset
        self.num_classes = len(categories)
        self.sketches = None
        self.sketches_normalized = None
        self.labels = None

        # Assign a unique label to each category
        self.category_to_label = {category: label_offset + idx for idx, category in enumerate(self.categories)}

        # Collect data information
        sketches = []
        labels = []

        with h5py.File(self.data_path, 'r') as h5file:
            for category in self.categories:
                group_path = f'{category}/{self.split}'
                if group_path in h5file:
                    data_group = h5file[group_path]
                    for dataset_name in data_group:
                        dataset = data_group[dataset_name]
                        data = dataset[:]
                        sketches.append(data)
                    labels.extend([self.category_to_label[category]] * len(data_group))
        
        # Purify & normalize sketches
        sketches, labels = self.purify(sketches, labels)

        self.sketches = sketches
        self.sketches_normalized = self.normalize(sketches)
        self.labels = labels

        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx) -> torch.Tensor:
        sketch = np.zeros((self.Nmax, 3))
        actual_length = self.sketches_normalized[idx].shape[0]
        sketch[:actual_length, :] = self.sketches_normalized[idx]
        label = np.array(self.labels[idx], dtype=np.int64)
        return sketch, label
    
    def max_size(self, sketches):
        sizes = [len(sketch) for sketch in sketches]
        return max(sizes)

    def purify(self, sketches, labels):
        data = []
        new_labels = []
        for i, sketch in enumerate(sketches):
            if 96 >= sketch.shape[0] > 0:  # remove small and too long sketches.
                sketch = np.minimum(sketch, 1000)  # remove large gaps.
                sketch = np.maximum(sketch, -1000)
                sketch = np.array(sketch, dtype=np.float32)  # change it into float32
                data.append(sketch)
                new_labels.append(labels[i])
        return data, new_labels
    
    def normalize(self, sketches):
        """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
        data = []
        scale_factor = 255
        for sketch in sketches:
            sketch_copy = sketch.copy()
            sketch_copy[:, 0:2] /= scale_factor
            data.append(sketch_copy)
        return data

class SketchDataModule(object):
    def __init__(
        self,
        data_path: str,
        categories: List[str],
        Nmax: int = 96,
        label_offset = 1,
        batch_size: int = 4,
        num_workers: int = 4,
    ):
        self.data_path = data_path
        self.categories = categories
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.Nmax = Nmax
        self.label_offset = label_offset

        self._set_dataset()

    def _set_dataset(self):
        self.train_ds = SketchDataset(
            self.data_path,
            split="train",
            categories=self.categories,
            Nmax=self.Nmax,
            label_offset=self.label_offset
        )
        self.test_ds = SketchDataset(
            self.data_path,
            split="test",
            categories=self.categories,
            Nmax=self.Nmax,
            label_offset=self.label_offset
        )
        self.num_classes = self.train_ds.num_classes

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )
