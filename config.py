"""Training configuration for PPG/GSR CLIP-style pretraining."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Dataset and DataLoader settings.

    `data_dir` should point to the folder that contains the 60 subjects x 5 days
    npy files. Files may be directly inside this folder or nested in subject
    folders. Each npy file is expected to have columns:
    ["timestamp", "GSR", "PPG"].
    """

    data_dir: Path = Path("data/npy")
    window_size: int = 400 # I remember that I used 600
    stride: int = 200 # and 300
    batch_size: int = 64
    num_workers: int = 0
    val_ratio: float = 0.2
    normalize: bool = True
    drop_last: bool = True


@dataclass
class ModelConfig:
    """Transformer encoder settings for both PPG and GSR branches."""

    input_channels: int = 1
    embed_dim: int = 128
    transformer_dim: int = 128
    patch_size: int = 40 # 10   6/3 must use patch_size = 10 if want to use pretrained weights from last week
    # must notice that PPG and GSR have different changing cycles, so it is better to use different patch sizes for them
    # better to do PPG_patch = 20? GSR_patch = 50-60?
    depth: int = 4
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    temperature: float = 0.1 # 0.07


@dataclass
class TrainConfig:
    """Optimization and checkpoint settings."""

    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "cuda"
    checkpoint_dir: Path = Path("checkpoints")
    save_every: int = 5
    log_every: int = 20


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


CFG = Config()