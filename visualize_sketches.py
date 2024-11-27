# visualize_sketches.py

import os
import h5py
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm
from sketch_diffusion.dataset import pen_state_to_binary, tensor_to_pil_image

def main():
    # HDF5 파일 경로
    data_dir = './data'
    h5_file_path = os.path.join(data_dir, 'sketches_rdp.h5')
    
    # 시각화할 카테고리와 개수 설정
    categories = ['cat', 'garden', 'helicopter']  # 원하는 카테고리 선택
    num_samples = 5  # 각 카테고리당 시각화할 샘플 수
    
    # HDF5 파일 열기
    with h5py.File(h5_file_path, 'r') as h5file:
        for category in categories:
            print(f"Processing category: {category}")
            category_group = h5file.get(category)
            if category_group is None:
                print(f"Category '{category}' not found in HDF5 file.")
                continue
            
            # Train과 Test 그룹 가져오기
            train_group = category_group.get('train')
            test_group = category_group.get('test')
            
            if train_group is None or test_group is None:
                print(f"Train/Test groups not found for category '{category}'.")
                continue
            
            # 시각화를 위해 Train과 Test에서 일부 샘플 선택
            for split, group in [('train', train_group), ('test', test_group)]:
                print(f"  Processing split: {split}")
                sample_keys = list(group.keys())[:num_samples]  # 처음 num_samples개 샘플 선택
                
                for sketch_key in sample_keys:
                    sketch_data = torch.from_numpy(group[sketch_key][()])
                    image = tensor_to_pil_image(sketch_data, canvas_size=(256, 256), padding=30, show_hidden=False)
                    
                    # 이미지 저장
                    output_dir = os.path.join(data_dir, 'visualizations', category, split)
                    os.makedirs(output_dir, exist_ok=True)
                    image_path = os.path.join(output_dir, f"{sketch_key}.png")
                    image.save(image_path)
                    print(f"    Saved image: {image_path}")

    print("Visualization completed.")

if __name__ == "__main__":
    main()