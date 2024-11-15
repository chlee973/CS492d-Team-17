import os
import ndjson
import numpy as np
import h5py
import json

def preprocess_sketch(sketch: list):
    sketch_pos = np.concatenate([np.array(stroke).astype(np.float32) for stroke in sketch], axis=1)
    sketch_vectors = sketch_pos[:, 1:] - sketch_pos[:, :-1]
    pen_state = np.full((sketch_vectors.shape[1]), 1).astype(np.float32)
    length_sum = 0
    stroke_lengths = [len(stroke[0]) for stroke in sketch]
    for stroke_length in stroke_lengths[:-1]:
        length_sum += stroke_length
        pen_state[length_sum-1] = 0
    result = np.transpose(np.concatenate((sketch_vectors, pen_state[np.newaxis, :]), axis=0))
    return result

def main():
    categories = ['cat', 'garden', 'helicopter']
    data_dir = './data'
    h5_file_path = os.path.join(data_dir, 'sketches.h5')

    with h5py.File(h5_file_path, 'w') as h5file:
        for category in categories:
            data_path = os.path.join(data_dir, f"{category}.ndjson")
            indices_path = os.path.join('sketch_data', category, 'train_test_indices.json')
            
            # Load train and test indices
            with open(indices_path, 'r') as f:
                indices = json.load(f)
                train_indices = indices['train']
                test_indices = indices['test']

            # Load and preprocess sketches
            with open(data_path, 'r') as f:
                data = ndjson.load(f)
            
            preprocessed_sketches = []
            for sketch in data:
                sketch = sketch['drawing']
                result = preprocess_sketch(sketch)
                preprocessed_sketches.append(result)
            
            # Create groups for category, train, and test
            category_group = h5file.create_group(category)
            train_group = category_group.create_group('train')
            test_group = category_group.create_group('test')
            
            # Save train data
            for i in train_indices:
                sketch_name = f'sketch_{i}'
                if sketch_name not in train_group:
                    sketch = preprocessed_sketches[i]
                    train_group.create_dataset(f'sketch_{i}', data=sketch, compression='gzip')
            
            # Save test data
            for i in test_indices:
                sketch_name = f'sketch_{i}'
                if sketch_name not in test_group:
                    sketch = preprocessed_sketches[i]
                    test_group.create_dataset(f'sketch_{i}', data=sketch, compression='gzip')
            
            print(f"{category}-train: {len(train_group)}")
            print(f"{category}-test: {len(test_group)}")

if __name__ == "__main__":
    main()