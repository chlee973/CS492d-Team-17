import os
import ndjson
import numpy as np
import h5py
import json
from tqdm import tqdm

## JSON 파일 없는 일반 Ndjson 데이터에 대해서, 모두 train으로 h5 파일 생성해주는 파이썬 파일

def preprocess_sketch(sketch: list):
    sketch_pos = np.concatenate([np.array(stroke).astype(np.float32) for stroke in sketch], axis=1)
    zero_pos = np.zeros((2,1), dtype=np.float32)
    sketch_pos = np.concatenate([zero_pos, sketch_pos], axis=1)
    sketch_vectors = sketch_pos[:, 1:] - sketch_pos[:, :-1]
    pen_state = np.ones((sketch_vectors.shape[1],)).astype(np.float32)
    pen_state[0] = 0
    length_sum = 0
    stroke_lengths = [len(stroke[0]) for stroke in sketch]
    for stroke_length in stroke_lengths[:-1]:
        length_sum += stroke_length
        pen_state[length_sum] = 0
    result = np.transpose(np.concatenate((sketch_vectors, pen_state[np.newaxis, :]), axis=0))
    return result

def main():
    category = 'basketball'
    data_dir = './data'
    h5_file_path = os.path.join(data_dir, f'sketches_{category}.h5')

    with h5py.File(h5_file_path, 'w') as h5file:
        data_path = os.path.join(data_dir, f"{category}.ndjson")
        # Load and preprocess sketches
        with open(data_path, 'r') as f:
            data = ndjson.load(f)
        
        preprocessed_sketches = []
        for sketch in tqdm(data):
            sketch = sketch['drawing']
            result = preprocess_sketch(sketch)
            preprocessed_sketches.append(result)
        
        # Create groups for category, train, and test
        category_group = h5file.create_group(category)
        train_group = category_group.create_group('train')
        
        for i, sketch in enumerate(preprocessed_sketches):
            sketch_name = f'sketch_{i}'
            if sketch_name not in train_group:
                train_group.create_dataset(sketch_name, data=sketch, compression='gzip')

        print(f"{category}-train: {len(train_group)}")

if __name__ == "__main__":
    main()