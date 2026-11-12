from .bcam import EBO, ELO
from .WithSE import CEMO, EMA
from .fit import CNNSENormalWithFit
from .modular import ModularCNN as ModularCNN
from .blocks import ConvNormAct as ConvNormAct, ResidualBlock as ResidualBlock
from .attention import ECAAttention as ECAAttention, CoordinateAttention as CoordinateAttention
from .bcam import CBAMBlock as CBAMBlock
from .WithSE import SEBlock as SEBlock
from .registry import ModelRegistry as ModelRegistry, ModelConfigError as ModelConfigError
from .builder import ModelBuilder as ModelBuilder

# Frozen legacy wildcard contract. New components use explicit named imports.
__all__ = ["EBO", "ELO", "CEMO", "EMA", "CNNSENormalWithFit"]

from .architectures import ResNet, PreActResNet, WideResNet, ResNeXt, DenseNet
from .m3_residual import BasicResidualBlock, Bottleneck, PreActivationBlock, WideResidualBlock, ResNeXtBottleneck
from .m3_attention import SimAM, TripletAttention, SelectiveKernelAttention

from .mobile_architectures import MobileNetV2, MobileNetV3, EfficientNetV2, GhostNet, ShuffleNetV2
from .reparam_architectures import RepVGG, MobileOne
from .m4_blocks import InvertedResidual, MBConv, FusedMBConv, GhostConv, GhostBottleneck, ChannelShuffle, ShuffleUnit, DropBlock2D
from .m4_reparam import RepVGGBlock, MobileOneBlock, reparameterize_model
