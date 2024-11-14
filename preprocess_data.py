import os
import ndjson
import numpy as np
import pickle

def preprocess_sketch(sketch: list):
    sketch_pos = np.concatenate([np.array(stroke) for stroke in sketch], axis=1)
    sketch_vectors = sketch_pos[:, 1:] - sketch_pos[:, :-1]
    pen_state = np.full((sketch_vectors.shape[1]), 1)
    length_sum = 0
    stroke_lengths = [len(stroke[0]) for stroke in sketch]
    for stroke_length in stroke_lengths[:-1]:
        length_sum += stroke_length
        pen_state[length_sum-1] = 0
    result = np.transpose(np.concatenate((sketch_vectors, pen_state[np.newaxis, :]), axis=0))
    return result

categories = ['cat', 'garden', 'helicopter']
data_dir = './data'
save_dir = './data/preprocessed'

for category in categories:
    data_path = os.path.join(data_dir, f"{category}.ndjson")
    with open(data_path, 'r') as f:
        data = ndjson.load(f)
    preprocessed_sketches = []
    for sketch in data:
        sketch = sketch['drawing']
        result = preprocess_sketch(sketch)
        preprocessed_sketches.append(result)
    save_path = os.path.join(data_dir, f"{category}.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(preprocessed_sketches, f)
    