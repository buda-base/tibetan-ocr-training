from dataclasses import dataclass


@dataclass
class DatasetDistribution:
    train_samples: list[str]
    val_samples: list[str]
    test_samples: list[str]

@dataclass
class CTCModelConfig:
    checkpoint: str
    model_file: str
    architecture: str
    input_width: int
    input_height: int
    input_layer: str
    output_layer: str
    squeeze_channel: bool
    swap_hw: bool
    charset: list[str]

@dataclass
class VitConfig:
    in_ch: int = 512
    embed_dim: int = 256
    patch_kernel: int = 3
    patch_stride: int = 1
    num_layers: int = 2
    num_heads: int = 4
    mlp_ratio: float = 2.0