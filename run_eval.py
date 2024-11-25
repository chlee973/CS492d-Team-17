import os
import argparse
from cleanfid import fid

parser = argparse.ArgumentParser()
parser.add_argument("--fdir1", type=str, default="./sketch_data/cat/images_test")
parser.add_argument("--fdir2", type=str, default="./sketch_diffusion/results/diffusion-ddpm-11-19-132249 - ema,sch")
args = parser.parse_args()

# compute FID
score_fid = fid.compute_fid(args.fdir1, args.fdir2)

# compute KID
score_kid = fid.compute_kid(args.fdir1, args.fdir2)

print("========================")
print(f"- FID score: {score_fid}")
print(f"- KID score: {score_kid}")
print("========================")