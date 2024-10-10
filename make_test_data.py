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
parser.add_argument("--save_dir", type=str, default="./test_data")
parser.add_argument("--num_per_category", type=int, default=20)
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(os.path.join(args.save_dir, "images"), exist_ok=True)

# list of categories
categories = os.listdir(args.data_dir)

# set random seed
SEED = 2024
random.seed(SEED)
np.random.seed(SEED)

# iterate over categories
index_info = {}

for category in categories:
    category_name = category.replace(".ndjson", "")
    print(f"Processing {category_name}...")

    # load data
    with open(os.path.join(args.data_dir, category), "r") as f:
        data = ndjson.load(f)

    # randomly sample indices
    sampled_indices = np.random.randint(0, len(data) + 1, size=min(args.num_per_category, len(data)))

    # iterate over data
    for idx in sampled_indices:
        item = data[idx]
        strokes = item["drawing"]

        # draw strokes
        image = draw_strokes(strokes)
        image.save(os.path.join(args.save_dir, "images", f"{category_name}_{idx:03d}.png"))

    # save index information
    index_info[category_name] = list([int(x) for x in sampled_indices])

    
# save index information
with open(os.path.join(args.save_dir, "index_info.json"), "w") as f:
    json.dump(index_info, f, indent=4)

print("Done!")