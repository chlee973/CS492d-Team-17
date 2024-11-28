import os
import ndjson
import numpy as np
import h5py
import json
from tqdm import tqdm
from rdp import rdp  # RDP 알고리즘 임포트

## 실제로 그리는 stroke일 때 pen_state가 1, 아닐 때 0
## 시작점의 위치 정보도 포함하여 첫 pen_state는 0이 된다.
def preprocess_sketch(sketch: list, epsilon=2.0):
    """
    sketch: list of strokes, where each stroke is [x, y] lists
    epsilon: float, RDP algorithm parameter to control the degree of simplification
    """
    # Apply RDP to each stroke
    simplified_sketch = []
    for stroke in sketch:
        points = np.array(stroke)  # Shape: (2, N)
        # Combine x and y into (N, 2) for rdp
        simplified_points = rdp(points.T, epsilon=epsilon).T
        # Transpose back to (2, M)
        simplified_sketch.append(simplified_points)

    # Flatten all simplified strokes into a single sequence
    sketch_pos = np.concatenate([stroke.astype(np.float32) for stroke in simplified_sketch], axis=1)
    
    # Add a starting zero position
    zero_pos = np.zeros((2,1), dtype=np.float32)
    sketch_pos = np.concatenate([zero_pos, sketch_pos], axis=1)
    
    # Compute vectors
    sketch_vectors = sketch_pos[:, 1:] - sketch_pos[:, :-1]
    
    # Initialize pen states to 1
    pen_state = np.ones((sketch_vectors.shape[1],)).astype(np.float32)
    pen_state[0] = 0  # 첫 pen_state는 0

    # Determine pen_state transitions (0 where pen is lifted)
    length_sum = 0
    stroke_lengths = [len(stroke[0]) for stroke in simplified_sketch]
    for stroke_length in stroke_lengths[:-1]:
        length_sum += stroke_length
        if length_sum < pen_state.shape[0]:
            pen_state[length_sum] = 0
    
    # Concatenate vectors and pen_state
    result = np.transpose(np.concatenate((sketch_vectors, pen_state[np.newaxis, :]), axis=0))
    return result

def main():
    categories = ['cat', 'garden', 'helicopter']
    data_dir = './data'
    h5_file_path = os.path.join(data_dir, 'sketches_rdp.h5')

    with h5py.File(h5_file_path, 'w') as h5file:
        for category in categories:
            data_path = os.path.join(data_dir, f"{category}.ndjson")
            indices_path = os.path.join('sketch_data', category, 'train_test_indices.json')
            
            # if category == "cat":
            #     indices_path_50 = os.path.join('sketch_data', category, 'train_50.json')
            #     indices_path = indices_path_50

            # Load train and test indices
            with open(indices_path, 'r') as f:
                indices = json.load(f)
                train_indices = indices['train']
                test_indices = indices['test']

            # Load and preprocess sketches
            with open(data_path, 'r') as f:
                data = ndjson.load(f)
            
            # Create groups for category, train, and test
            category_group = h5file.create_group(category)
            train_group = category_group.create_group('train')
            test_group = category_group.create_group('test')
            
            # Save train data
            for i in tqdm(train_indices):
                sketch_name = f'sketch_{i}'
                if sketch_name not in train_group:
                    sketch = data[i]['drawing']
                    sketch = preprocess_sketch(sketch, epsilon=2.0)
                    train_group.create_dataset(f'sketch_{i}', data=sketch, compression='gzip')
            
            # Save test data
            for i in tqdm(test_indices):
                sketch_name = f'sketch_{i}'
                if sketch_name not in test_group:
                    sketch = data[i]['drawing']
                    sketch = preprocess_sketch(sketch, epsilon=2.0)
                    test_group.create_dataset(f'sketch_{i}', data=sketch, compression='gzip')
            
            print(f"{category}-train: {len(train_group)}")
            print(f"{category}-test: {len(test_group)}")

if __name__ == "__main__":
    main()