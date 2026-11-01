"""Trusted, importable recipes for isolated capacity validation."""
import os
import time
import torch


class Selective(torch.nn.Module):
    def __init__(self, reject=2, failure='oom'):
        super().__init__()
        self.reject = reject
        self.failure = failure
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.shape[0] == self.reject:
            if self.failure == 'oom':
                raise MemoryError('controlled CPU allocation failure fixture')
            if self.failure == 'exception':
                raise RuntimeError('controlled ordinary failure')
            if self.failure == 'crash':
                os._exit(73)
            if self.failure == 'timeout':
                time.sleep(90)
            if self.failure == 'cardinality':
                return x[:0]
        return x * self.weight


def invalid():
    return object()
