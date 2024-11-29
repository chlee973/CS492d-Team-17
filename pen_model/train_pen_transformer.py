import torch
import torchvision
import h5py
from torch import nn
from pathlib import Path
from datetime import datetime
# from sketch_diffusion.image_datasets import load_data, SketchesDataset
from sketch_diffusion.dataset import SketchDataModule, get_data_iterator, pen_state_to_binary, tensor_to_pil_image

class PositionalEncoding(nn.Module):
    """포지셔널 인코딩을 구현한 클래스입니다."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 포지셔널 인코딩 초기화
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 인덱스
        pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 인덱스
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_length, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerPenet(nn.Module):
    """Transformer 기반의 Penet 모델입니다."""
    def __init__(self, input_dim=2, hidden_dim=256, num_layers=4, num_heads=8, dropout=0.1):
        super(TransformerPenet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # 입력 임베딩
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # 포지셔널 인코딩
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # Transformer 인코더
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout, dim_feedforward=hidden_dim*4, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 출력 레이어
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: (batch, seq_length, input_dim)
        x = self.embedding(x)  # (batch, seq_length, hidden_dim)
        x = self.pos_encoder(x)  # (batch, seq_length, hidden_dim)
        
        # Transformer는 입력을 (seq_length, batch, hidden_dim) 형태로 받습니다.
        x = x.transpose(0, 1)  # (seq_length, batch, hidden_dim)
        x = self.transformer_encoder(x)  # (seq_length, batch, hidden_dim)
        x = x.transpose(0, 1)  # (batch, seq_length, hidden_dim)
        
        out = self.fc_out(x)  # (batch, seq_length, 1)
        out = self.sigmoid(out)  # (batch, seq_length, 1)
        return out.squeeze(-1)  # (batch, seq_length)
    
    def save(self, file_path):
        # 저장할 항목 정의
        state = {
            "state_dict": self.state_dict(),  # 모델 가중치
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout": 0.1
        }
        torch.save(state, file_path)
    
    def load(self, file_path):
        # 저장된 체크포인트 로드
        dic = torch.load(file_path, map_location="cpu")
        state_dict = dic["state_dict"]
        
        # 모델 가중치 로드
        self.load_state_dict(state_dict)
        print(f"Model loaded from {file_path}")

if __name__ == "__main__":
    device = f"cuda:{0}"
    # data = SketchesDataset(
    #     "/root/sketchtext/sketch-diffusion-main/datasets_npz",
    #     # ["airplane.npz", "apple.npz", "bus.npz", "car.npz"],
    #     ["airplane.npz"],
    #     "train")

    ds_module = SketchDataModule(
        data_path="data/sketches_rdp.h5",
        categories=['cat'],
        Nmax=96,
        label_offset=1,
        batch_size=512,
        num_workers=4,
    )

    train_dl = ds_module.train_dataloader()
    train_it = get_data_iterator(train_dl)
    ##########################
    
    model = TransformerPenet().cuda()
    # criterion = nn.CrossEntropyLoss()
    noise_criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    # for step in range(100000):
    for step in range(100001):
        model.train()
        optimizer.zero_grad()
        img, label = next(train_it)
        img, label = img.to(device).to(torch.float32), label.to(torch.float32)
        outputs = model(img[:,:,:2])
        
        loss = noise_criterion(outputs, img[:,:,2])
        loss.backward()
        optimizer.step()
        if step %10000 == 0:
            print(f"[{step}]: {loss.item()}")
    
    path = f"results/pen-state-prediction-transformer-{datetime.now().strftime('%m-%d-%H%M%S')}"
    save_dir = Path(path)
    save_dir.mkdir(exist_ok=True, parents=True)
    save_path = f"results/pen-state-prediction-transformer-{datetime.now().strftime('%m-%d-%H%M%S')}/pen.ckpt"  # 확장자 추가
    model.save(save_path)