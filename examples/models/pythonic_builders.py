"""Three explicit CPU examples: a CNN, a graph, and training preparation only."""
import torch
from torch import nn
from torch.utils.data import TensorDataset

from hyso import GraphModelBuilder, ModelBuilder, TrainingConfig, TrainingSession
from hyso.models.graph import GraphModel


def simple_cnn() -> nn.Module:
    """Build fresh CPU weights; do not run forward or training."""
    return (ModelBuilder((1, 8, 8), 2)
            .conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
            .relu()
            .max_pool2d(kernel_size=2)
            .build())


def graph_model() -> GraphModel:
    """Keep graph node names, input references and output selection explicit."""
    return (GraphModelBuilder({'image': [1, 8, 8]})
            .conv_norm_act('stem', ['image'], in_channels=1, out_channels=4,
                           norm=None, activation='relu')
            .max_pool2d('pool', ['stem'], kernel_size=2)
            .add('head', 'classification_head', ['pool'], in_features=4, num_classes=2)
            .outputs(logits='head')
            .build())


def prepare_training() -> TrainingSession:
    """Prepare the existing session/loaders; calling fit remains a separate decision."""
    generator=torch.Generator(device='cpu').manual_seed(71)
    inputs=torch.randn(4, 1, 8, 8, generator=generator, device='cpu')
    targets=torch.tensor([0, 1, 0, 1], dtype=torch.long, device='cpu')
    dataset=TensorDataset(inputs, targets)
    return TrainingConfig(epochs=1, batch_size=2, device='cpu').prepare_datasets(
        simple_cnn(), dataset)


def advanced_cnn() -> nn.Module:
    """Commercial-compatible depthwise, attention and pixel-rearrangement example."""
    return (ModelBuilder((16, 8, 8), 3)
            .depthwise_conv2d(in_channels=16)
            .pointwise_conv2d(in_channels=16, out_channels=16)
            .cbam(channels=16, reduction=4)
            .eca(channels=16)
            .pixel_shuffle(upscale_factor=2)
            .build())


def fused_graph() -> GraphModel:
    """Commercial-compatible explicit branches, weighted fusion and classifier head."""
    return (GraphModelBuilder({'image': [16, 8, 8]})
            .depthwise_conv2d('left', ['image'], in_channels=16)
            .pointwise_conv2d('right', ['image'], in_channels=16, out_channels=16)
            .weighted_sum('merge', ['left', 'right'], weights=[0.25, 0.75])
            .classification_head('head', ['merge'], in_features=16, num_classes=3)
            .outputs(logits='head')
            .build())


if __name__=='__main__':
    print(type(simple_cnn()).__name__)
    print(type(graph_model()).__name__)
    print(prepare_training().settings()['state'])
