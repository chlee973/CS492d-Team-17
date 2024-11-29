import torch
import torch.nn as nn
import math
from .module import Swish, TimeEmbedding

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=96):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.max_len = max_len
    
    def forward(self, x):
        """
        x: (B, N, d_model)
        """
        B, N, _ = x.size()
        if N > self.max_len:
            raise ValueError(f"Sequence length N={N} exceeds max_len={self.max_len}")
        positions = torch.arange(0, N, device=x.device).unsqueeze(0).expand(B, N)  # (B, N)
        pos_emb = self.pos_embedding(positions)  # (B, N, d_model)
        return x + pos_emb

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=96):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)  # Use max_len parameter
        self.max_len = max_len
    
    def forward(self, x):
        """
        x: (B, N, d_model)
        """
        B, N, _ = x.size()
        if N > self.max_len:
            raise ValueError(f"Sequence length N={N} exceeds max_len={self.max_len}")
        positions = torch.arange(0, N, device=x.device).unsqueeze(0).expand(B, N)  # (B, N)
        pos_emb = self.pos_embedding(positions)  # (B, N, d_model)
        return x + pos_emb

class TransformerPenModel(nn.Module):
    def __init__(self, d_model=128, num_layers=4, num_heads=8, T=1000, max_len=96):
        super(TransformerModel, self).__init__()
        self.Nmax = max_len
        self.d_model = d_model
        self.T = T
        
        # 타임스텝 임베딩
        self.time_embed = TimeEmbedding(d_model)
        
        # 데이터 임베딩
        self.input_proj = nn.Linear(3, d_model)
        self.input_layer_norm = nn.LayerNorm(d_model)  # 입력 임베딩 후 LayerNorm 추가
        
        # 포지셔널 인코딩
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_len)
        
        # Transformer 인코더
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=0.1, activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.transformer_layer_norm = nn.LayerNorm(d_model)  # Transformer 인코더 후 LayerNorm 추가
        
        # 출력 레이어
        self.output_proj = nn.Linear(d_model, 1)

        self._initialize_parameters()

    def _initialize_parameters(self):
        # Xavier 초기화 적용
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, timestep, class_label=None):
        """
        x: (B, N, 2)
        t: (B,) 타임스텝
        """
        B, N, _ = x.shape
        
        # 타임스텝 임베딩
        t_embed = self.time_embed(timestep)  # (B, d_model) assuming TimeEmbedding outputs d_model
        t_embed = t_embed.unsqueeze(1).repeat(1, N, 1)  # (B, N, d_model)
        
        # 데이터 임베딩
        x_embed = self.input_proj(x)  # (B, N, d_model)
        x_embed = self.input_layer_norm(x_embed)  # LayerNorm 적용
        
        # 포지셔널 인코딩 추가
        x = x_embed + t_embed  # (B, N, d_model)
        x = self.positional_encoding(x)  # (B, N, d_model)
        
        # Transformer 입력 형식 맞추기: (N, B, d_model)
        x = x.permute(1, 0, 2)  # (N, B, d_model)
        
        # Transformer 통과
        x = self.transformer(x)  # (N, B, d_model)
        
        # 다시 (B, N, d_model)
        x = x.permute(1, 0, 2)  # (B, N, d_model)
        x = self.transformer_layer_norm(x)  # LayerNorm 적용
        
        # 출력 예측
        epsilon = self.output_proj(x)  # (B, N, 2)
        return epsilon