import os
import re
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from tqdm import tqdm

# 디바이스 설정: GPU가 사용 가능하면 'cuda', 아니면 'cpu'
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# CLIP 모델 및 프로세서 로드 후 GPU로 이동
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# CLIP Score 계산 함수
def calculate_clip_score(image_path, text="a sketch of a helicopter"):
    """
    이미지와 텍스트의 CLIP Score를 계산합니다.
    Args:
        image_path (str): 이미지 파일 경로.
        text (str): CLIP 모델에 입력할 텍스트 프롬프트.
    Returns:
        float: CLIP Score.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
        
        # 입력 데이터를 GPU로 이동
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # 이미지-텍스트 유사도
            score = logits_per_image.item()
        return score
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return float('-inf')  # 에러 발생 시 최소값 반환

# 디렉토리에서 이미지 파일 읽기
def get_image_files(directory):
    """
    디렉토리에서 이미지 파일 경로를 수집합니다.
    Args:
        directory (str): 이미지 파일이 저장된 디렉토리 경로.
    Returns:
        list: 이미지 파일 경로 리스트.
    """
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in supported_extensions
    ]

# 파일 이름으로부터 인덱스 추출 함수
def extract_index(filename):
    """
    파일 이름으로부터 인덱스를 추출합니다.
    파일 이름 형식에 따라 수정이 필요할 수 있습니다.
    예: 'image_12345.jpg'에서 12345 추출
    Args:
        filename (str): 파일 이름.
    Returns:
        str: 추출된 인덱스. 추출 실패 시 None 반환.
    """
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        return None

# CLIP Score 상위 1000개 인덱스 선택
def select_top_indices(image_dir, top_k=1000, text="a sketch of a helicopter", output_file="top_1000_indices.txt"):
    """
    이미지 디렉토리에서 CLIP Score 상위 K개의 이미지 인덱스를 선택합니다.
    Args:
        image_dir (str): 입력 이미지 디렉토리.
        top_k (int): 선택할 이미지 개수.
        text (str): CLIP 모델에 입력할 텍스트 프롬프트.
        output_file (str): 상위 K개 인덱스를 저장할 파일 경로.
    """
    image_files = get_image_files(image_dir)

    print(f"Found {len(image_files)} images in {image_dir}. Calculating CLIP Scores...")
    scores = []
    for image_path in tqdm(image_files, desc="Calculating CLIP Scores"):
        score = calculate_clip_score(image_path, text=text)
        scores.append((image_path, score))

    # 상위 K개 이미지 선택
    scores.sort(key=lambda x: x[1], reverse=True)
    top_images = scores[:top_k]

    # 인덱스 추출
    top_indices = []
    for image_path, score in top_images:
        filename = os.path.basename(image_path)
        index = extract_index(filename)
        if index is not None:
            top_indices.append(index)
        else:
            print(f"Could not extract index from filename: {filename}")

    # 인덱스를 파일로 저장
    with open(output_file, 'w') as f:
        for idx in top_indices:
            f.write(f"{idx}\n")

    print(f"Top {top_k} indices saved to {output_file}.")

# 실행
if __name__ == "__main__":
    category = "cat"
    input_directory = f"sketch_data/{category}/images_train"  # 스케치 이미지 디렉토리
    select_top_indices(input_directory, top_k=1000, text="a sketch of a {category}", output_file="top_1000_{category}_indices.txt")