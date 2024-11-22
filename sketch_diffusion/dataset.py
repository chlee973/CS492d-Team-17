import os
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import h5py

from .sketch_util import scale_sketch, sketch_size

def pen_state_to_binary(x):
    # x: [B, C, 4] 
    # 마지막 차원은 (dx, dy, pen_state0, pen_state1)
    # 마지막 차원에 softmax를 취해야함
    assert x.shape[-1] == 4
    x_clone = x.clone()
    pen_states0 = x_clone[:, :, 2]
    ones = pen_states0 < 2
    binary_pen_states = torch.zeros_like(pen_states0, dtype=pen_states0.dtype, device=pen_states0.device)
    binary_pen_states[ones] = 1
    result = torch.cat((x_clone[:, :, :2], binary_pen_states[:, :, None]), dim=-1)
    assert result.shape[-1] == 3
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
    
    skip_line = False
    for dx, dy, pen_state in sketch:
        next_x = current_x + dx
        next_y = current_y + dy

        if skip_line:
            draw.line([current_x, current_y, next_x, next_y], fill="yellow", width=1)
            skip_line = False
        else:
            draw.line([current_x, current_y, next_x, next_y], fill="black", width=1)
            if pen_state == 1:
                skip_line = True
        current_x, current_y = next_x, next_y
    
    return image

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

    # def __getitem__(self, idx) -> torch.Tensor:
    #     sketch = np.zeros((self.Nmax, 3))
    #     actual_length = self.sketches_normalized[idx].shape[0]
    #     sketch[:actual_length, :] = self.sketches_normalized[idx]
    #     label = np.array(self.labels[idx], dtype=np.int64)
    #     return sketch, label

    def __getitem__(self, idx) -> torch.Tensor:
        sketch = self.sketches_normalized[idx]
        if len(sketch) < self.Nmax:
            sketch = self.resize_sketch(sketch, self.Nmax)
        elif len(sketch) > self.Nmax:
            sketch = sketch[:self.Nmax]
        # else: length is exactly Nmax, no action needed

        label = np.array(self.labels[idx], dtype=np.int64)
        return sketch, label

    def resize_sketch(self, sketch, Nmax):
        sketch = sketch.copy()
        while len(sketch) < Nmax:
            pen_down_indices = np.where(sketch[:, 2] != 1)[0]
            if len(pen_down_indices) == 0:
                break
            dx_dy = sketch[pen_down_indices, :2]
            lengths = np.sqrt(np.sum(dx_dy ** 2, axis=1))
            sorted_indices = np.argsort(-lengths)
            split_occurred = False
            for idx in pen_down_indices[sorted_indices]:
                dx, dy, pen_state = sketch[idx]
                if dx == 0 and dy == 0:
                    continue
                new_stroke1 = [dx / 2, dy / 2, pen_state]
                new_stroke2 = [dx / 2, dy / 2, pen_state]
                sketch[idx] = new_stroke1
                sketch = np.insert(sketch, idx + 1, new_stroke2, axis=0)
                split_occurred = True
                if len(sketch) >= Nmax:
                    break
            if not split_occurred:
                break
        if len(sketch) < Nmax:
            padding = np.zeros((Nmax - len(sketch), 3))
            sketch = np.vstack([sketch, padding])
        elif len(sketch) > Nmax:
            sketch = sketch[:Nmax]
        return sketch

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
