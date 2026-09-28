"""Sequence classifiers for landmark sequences [B, T, F] -> logits [B, num_classes].

Both models are small (<1M params) so they run in real time in the browser
via onnxruntime-web (WASM / WebGPU).
"""
from __future__ import annotations

import torch
import torch.nn as nn

import config as C


class InputProjection(nn.Module):
    """Per-frame Linear -> LayerNorm -> GELU projection of the raw feature vector."""

    def __init__(self, feat_dim: int, d: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feat_dim, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class BiLSTMClassifier(nn.Module):
    """2-layer bidirectional LSTM (hidden 128) with mean+max temporal pooling."""

    def __init__(self, num_classes: int, feat_dim: int = C.FEAT_DIM, hidden: int = 128,
                 layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.proj = InputProjection(feat_dim, hidden, dropout)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(4 * hidden), nn.Dropout(dropout),
                                  nn.Linear(4 * hidden, num_classes))

    def forward(self, x):                       # x: [B, T, F]
        h, _ = self.lstm(self.proj(x))          # [B, T, 2H]
        pooled = torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=-1)
        return self.head(pooled)


class TransformerClassifier(nn.Module):
    """Small Transformer encoder: 2 layers, 4 heads, d=128, learned positions, CLS pooling."""

    def __init__(self, num_classes: int, feat_dim: int = C.FEAT_DIM, d_model: int = 128,
                 heads: int = 4, layers: int = 2, ff: int = 256, dropout: float = 0.2,
                 seq_len: int = C.SEQ_LEN):
        super().__init__()
        self.proj = InputProjection(feat_dim, d_model, dropout)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))   # +1 for CLS
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        # enable_nested_tensor=False keeps the graph simple/stable for ONNX export
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, num_classes))

    def forward(self, x):                       # x: [B, T, F], T == seq_len
        z = self.proj(x)
        cls = self.cls.expand(z.shape[0], -1, -1)
        z = torch.cat([cls, z], dim=1) + self.pos[:, : z.shape[1] + 1]
        z = self.encoder(z)
        return self.head(self.norm(z[:, 0]))


def build_model(name: str, num_classes: int, feat_dim: int = C.FEAT_DIM, **kw) -> nn.Module:
    if name == "lstm":
        return BiLSTMClassifier(num_classes, feat_dim, **kw)
    if name == "transformer":
        return TransformerClassifier(num_classes, feat_dim, **kw)
    raise ValueError(f"unknown model '{name}' (use lstm | transformer)")


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    for n in ("lstm", "transformer"):
        m = build_model(n, len(C.VOCAB))
        out = m(torch.randn(2, C.SEQ_LEN, C.FEAT_DIM))
        print(f"{n:12s} params={count_params(m):,}  out={tuple(out.shape)}")
