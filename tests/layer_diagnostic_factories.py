"""Small owned inference recipes for allocation/perturbation tests."""
import torch


class Temporary(torch.nn.Module):
    def forward(self,x):
        temporary=x.clone()
        result=temporary+1
        del temporary
        return result


class Memory(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.leaf=Temporary()

    def forward(self,x):
        return self.leaf(x)
