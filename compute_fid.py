import os
import argparse
from cleanfid import fid

parser = argparse.ArgumentParser()
parser.add_argument("--fdir1", type=str, default="./test_data")
parser.add_argument("--fdir2", type=str, default="./test_data_2")
parser.add_argument("--save_path", type=str, default="./fid_score.txt")
args = parser.parse_args()

# compute FID
score = fid.compute_fid(args.fdir1, args.fdir2)
print(f"FID score: {score}")