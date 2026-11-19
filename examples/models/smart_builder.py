"""Short schema2 CNN using exact metadata inference; CPU build/forward only."""
import json

import torch

from hyso import ModelBuilder


def cnn_builder() -> ModelBuilder:
    """Describe a classifier without allocating weights, running forward or training."""
    return (ModelBuilder.classification((1,8,8), num_classes=3)
            .conv2d(out_channels=8)
            .batch_norm()
            .relu()
            .max_pool2d(kernel_size=2)
            .conv2d(out_channels=16)
            .batch_norm()
            .relu())


if __name__=='__main__':
    model=cnn_builder().build().eval()
    with torch.inference_mode():
        output=model(torch.ones(2,1,8,8,device='cpu'))
    print(json.dumps(dict(schema_version=2,output_shape=list(output.shape))))
