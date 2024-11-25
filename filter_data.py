import os
import json
import ndjson
import argparse
from PIL import Image, ImageDraw

def draw_strokes(strokes, height=256, width=256):
    """
    주어진 strokes로 새로운 PIL 이미지를 만듭니다.
    """
    image = Image.new("RGB", (width, height), "white")
    image_draw = ImageDraw.Draw(image)

    for stroke in strokes:
        # x와 y 좌표를 결합합니다.
        points = list(zip(stroke[0], stroke[1]))
        # 포인트들을 연결하여 선을 그립니다.
        image_draw.line(points, fill=0)

    return image

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./data")
parser.add_argument("--save_dir", type=str, default="./sketch_data")
parser.add_argument("--category", type=str, default="cat")
args = parser.parse_args()

save_dir = os.path.join(args.save_dir, args.category)
os.makedirs(save_dir, exist_ok=True)
os.makedirs(os.path.join(save_dir, "images_train"), exist_ok=True)
os.makedirs(os.path.join(save_dir, "images_test"), exist_ok=True)

# 데이터 로드
with open(os.path.join(args.data_dir, f"{args.category}.ndjson"), "r") as f:
    data = ndjson.load(f)

# train/test 인덱스 로드
with open(os.path.join(save_dir, "train_test_indices.json"), "r") as f:
    train_test_indices = json.load(f)

train_indices = train_test_indices["train"]
test_indices = train_test_indices["test"]

# Train 데이터에 대한 이미지 생성 및 저장
for idx in train_indices:
    item = data[idx]
    strokes = item["drawing"]
    image = draw_strokes(strokes)
    image.save(os.path.join(save_dir, "images_train", f"{idx:06d}.png"))

# Test 데이터에 대한 이미지 생성 및 저장
for idx in test_indices:
    item = data[idx]
    strokes = item["drawing"]
    image = draw_strokes(strokes)
    image.save(os.path.join(save_dir, "images_test", f"{idx:06d}.png"))

print("완료되었습니다!")
