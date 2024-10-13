import os
import json
import ndjson
import argparse
from PIL import Image, ImageDraw
import random
import numpy as np

def draw_strokes(strokes, height=256, width=256):
    """
    Make a new PIL image with the given strokes
    """
    image = Image.new("RGB", (width, height), "white")
    image_draw = ImageDraw.Draw(image)

    for stroke in strokes:
        # concat x and y coordinates
        points = list(zip(stroke[0], stroke[1]))

        # draw all points
        # image_draw.point(points, fill=0)
        image_draw.line(points, fill=0)

    return image

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./data")
parser.add_argument("--save_dir", type=str, default="./sketch_data")
parser.add_argument("--category", type=str, default="cat")
parser.add_argument("--num_train", type=int, default=10000)
parser.add_argument("--num_test", type=int, default=2000)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

save_dir = os.path.join(args.save_dir, args.category)
os.makedirs(save_dir, exist_ok=True)
os.makedirs(os.path.join(save_dir, "images_train"), exist_ok=True)
os.makedirs(os.path.join(save_dir, "images_test"), exist_ok=True)

# set random seed
random.seed(args.seed)
np.random.seed(args.seed)

# load data
with open(os.path.join(args.data_dir, f"{args.category}.ndjson"), "r") as f:
    data = ndjson.load(f)

# randomly sample indices
sampled_indices = np.random.randint(
    0, len(data) + 1, 
    size=min(args.num_train+args.num_test, len(data)))

train_indices = sampled_indices[:args.num_train]
test_indices = sampled_indices[args.num_train:]

# iterate over data
for idx in train_indices:
    item = data[idx]
    strokes = item["drawing"]
    image = draw_strokes(strokes)
    image.save(os.path.join(save_dir, "images_train", f"{idx:06d}.png"))

for idx in test_indices:
    item = data[idx]
    strokes = item["drawing"]
    image = draw_strokes(strokes)
    image.save(os.path.join(save_dir, "images_test", f"{idx:06d}.png"))

# save train/test indices
train_indices = [int(x) for x in train_indices]
test_indices = [int(x) for x in test_indices]

train_test_indices = {
    "train": train_indices,
    "test": test_indices
}

with open(os.path.join(save_dir, "train_test_indices.json"), "w") as f:
    json.dump(train_test_indices, f)

print("Done!")