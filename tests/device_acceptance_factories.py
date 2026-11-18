"""Small deterministic CNN for actual CPU/GPU P34 acceptance; no downloads."""


def cnn():
    import torch
    from torch import nn
    model=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),
        nn.Conv2d(16,32,3,stride=2,padding=1),nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(32,10))
    return model,torch.randn(8,3,64,64)
