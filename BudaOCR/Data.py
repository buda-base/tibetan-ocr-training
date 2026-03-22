from enum import Enum
from dataclasses import dataclass
from pathlib import Path


class Encoding(Enum):
    UNICODE = 0
    WYLIE = 1


@dataclass
class DatasetDistribution:
    train_samples: list[str]
    val_samples: list[str]
    test_samples: list[str]

    
@dataclass
class CTCModelConfig:
    checkpoint: str
    onnx_file: str
    architecture: str
    input_width: int
    input_height: int
    input_layer: str
    output_layer: str
    squeeze_channel: bool
    swap_hw: bool
    add_blank: bool
    encoding: Encoding
    charset: str | list[str]


@dataclass
class VitConfig:
    in_ch: int = 512
    embed_dim: int = 256
    patch_kernel: int = 3
    patch_stride: int = 1
    num_layers: int = 2
    num_heads: int = 4
    mlp_ratio: float = 2.0


@dataclass
class KenLMConfig:
    kenlm_file: str | Path
    arpa_file: str | Path
    unigrams: list[str]
