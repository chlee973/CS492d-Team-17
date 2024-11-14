import os
import pickle

categories = ['cat', 'garden', 'helicopter']
data_dir = './data'

data_path = os.path.join(data_dir, 'cat.pkl')

with open(data_path, 'rb') as f:
    loaded_arrays = pickle.load(f)

print(loaded_arrays[0])
print(loaded_arrays[0].shape)