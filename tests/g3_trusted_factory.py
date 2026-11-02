"""Deliberately trivial trusted G3 CPU test recipe; not a user model."""

def build(*, width=2):
    import torch
    from torch import nn
    if type(width) is not int or not 1 <= width <= 8:
        raise ValueError('bounded_fixture_width')
    torch.manual_seed(42)
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, width), nn.ReLU(), nn.Linear(width, 2))
    x = torch.ones(2, 1, 2, 2, dtype=torch.float32)
    return model, x
