import torch
import torchvision
import h5py
from torch import nn
from pathlib import Path
from datetime import datetime
# from sketch_diffusion.image_datasets import load_data, SketchesDataset
from sketch_diffusion.dataset import SketchDataModule, get_data_iterator, pen_state_to_binary, tensor_to_pil_image


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

def normalization(channels):
    return GroupNorm32(32, channels)

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

class BasePenet(nn.Module):
    def __init__(self, dims, channels, output_channels, dropout=0.1):
        super(BasePenet, self).__init__()
        self.out_channels = output_channels
        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.out_layers = torch.nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            torch.nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

    def forward(self, x):  
        x = self.in_layers(x)
        x = self.out_layers(x)
        return x

class Penet(nn.Module):
    def __init__(self, dims, channels, dropout=0.1):
        super(Penet, self).__init__()
        self.out_channels = channels
        self.softmax = nn.Softmax(dim=-1)
        self.layer1 = BasePenet(dims, 1 * channels, 2 * channels, dropout)
        self.layer2 = BasePenet(dims, 2 * channels, 4 * channels, dropout)
        self.layer3 = BasePenet(dims, 4 * channels, 8 * channels, dropout)
        self.layer4 = BasePenet(dims, 8 * channels, 4 * channels, dropout)
        self.layer5 = BasePenet(dims, 4 * channels, 2 * channels, dropout)
        self.layer6 = BasePenet(dims, 2 * channels, 1 * channels, dropout)
        self.final_layer = nn.Linear(2, 1)

    def forward(self, x):
        x = self.softmax(x)
        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        h3 = self.layer3(h2)
        h4 = self.layer4(h3)
        h5 = self.layer5(h4)
        h6 = self.layer6(h5)
        h6 = self.final_layer(h6).squeeze(-1)  # (b x 96 x 1 -> b x 96)

        return h6
    def save(self, file_path):
        # 저장할 항목 정의
        state = {
            "state_dict": self.state_dict(),  # 모델 가중치
            "dims": 1,
            "channels": 96,
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


# class NewPenet(nn.Module):
#     def __init__(self, dims, channels, dropout=0.1):
#         super(NewPenet, self).__init__()
#         self.out_channels = channels
#         self.softmax = nn.Softmax(dim=-1)
#         self.layer1 = BasePenet(dims, channels, 32, dropout)
#         self.layer2 = BasePenet(dims, 32, 16, dropout)
#         self.layer3 = BasePenet(dims, 16, 8, dropout)
#         self.layer4 = BasePenet(dims, 8, 16, dropout)
#         self.layer5 = BasePenet(dims, 16, 32, dropout)
#         self.layer6 = BasePenet(dims, 32, channels, dropout)

#     def forward(self, x):
#         x = self.softmax(x)
#         h1 = self.layer1(x)
#         h2 = self.layer2(h1)
#         h3 = self.layer3(h2)
#         h4 = self.layer4(h3)
#         h5 = self.layer5(h4)
#         h6 = self.layer6(h5)
#         h6 = torch.sigmoid(h6)

#         return h6

#     def save(self, file_path):
#         # 저장할 항목 정의
#         state = {
#             "state_dict": self.state_dict(),  # 모델 가중치
#             "dims": 1,
#             "channels": 96,
#             "dropout": 0.1
#         }
#         torch.save(state, file_path)

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
    
    model = Penet(dims=1, channels=96).cuda()
    # criterion = nn.CrossEntropyLoss()
    noise_criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    # for step in range(100000):
    for step in range(100000):
        model.train()
        optimizer.zero_grad()
        # batch, _, _, _ = data.make_batch(512)
        # batch = batch.transpose(0, 1).contiguous()
        # batch = batch[:, :, :3]
        # pen_state = batch[:, :, 2]
        # pen_state = torch.unsqueeze(pen_state, 2)
        img, label = next(train_it)
        img, label = img.to(device).to(torch.float32), label.to(torch.float32)
        outputs = model(img[:,:,:2])
        
        loss = noise_criterion(outputs, img[:,:,2])
        loss.backward()
        optimizer.step()
        if step %10000 == 0:
            print(f"[{step}]: {loss.item()}")
    
    path = f"results/pen-state-prediction-{datetime.now().strftime('%m-%d-%H%M%S')}"
    save_dir = Path(path)
    save_dir.mkdir(exist_ok=True, parents=True)
    save_path = f"results/pen-state-prediction-{datetime.now().strftime('%m-%d-%H%M%S')}/pen.ckpt"  # 확장자 추가
    model.save(save_path)