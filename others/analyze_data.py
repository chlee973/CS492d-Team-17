from collections import Counter
import numpy as np
import h5py

data_path = './data/sketches_rdp.h5'

categories = ['cat', 'garden', 'helicopter']

with h5py.File(data_path, 'r') as h5file:
    for category in categories:
        group_path = f'{category}/train'
        sketches = []
        if group_path in h5file:
            data_group = h5file[group_path]
            for dataset_name in data_group:
                dataset = data_group[dataset_name]
                data = dataset[:]
                sketches.append(data)
        
        sketch_lengths = [len(sketch) for sketch in sketches]
        num_ones = [np.sum(sketch[:, -1]==1) for sketch in sketches]
        num_zeros = [np.sum(sketch[:, -1]==0) for sketch in sketches]
        num_ones = sum(num_ones)
        num_zeros = sum(num_zeros)
        print(num_ones)
        print(num_zeros)
        length_counts = Counter(sketch_lengths)
        sorted_lengths = sorted(length_counts.keys())  # 길이를 정렬
        cumulative_counts = np.cumsum([length_counts[length] for length in sorted_lengths])

        # 누적합의 비율 계산
        total_strings = len(sketches)  # 전체 문자열 개수
        cumulative_ratios = cumulative_counts / total_strings  # 비율 계산
        print(f"<{category}>")
        # 출력
        print("String Length | Cumulative Count | Cumulative Ratio")
        print("---------------------------------------------")
        for length, count, ratio in zip(sorted_lengths, cumulative_counts, cumulative_ratios):
            print(f"{length:>13} | {count:>16} | {ratio*100:.3f}%")