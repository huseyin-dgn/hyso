"""Explicit named tensor batches for bounded two-input Graph training/resume."""
import torch
from torch.utils.data import Dataset


class NamedTensorDataset(Dataset):
    """Two named CPU tensor inputs and one target tensor; no transforms or RNG.

    Names bind tensors explicitly. Dictionary insertion order is immaterial.
    Checkpoint admission additionally validates the exact Graph/task/dtype contract.
    """
    def __init__(self, inputs, targets):
        if (type(inputs) is not dict or len(inputs) != 2
                or any(type(name) is not str or not name for name in inputs)):
            raise ValueError('NamedTensorDataset requires exactly two explicit input names')
        self.input_names = tuple(sorted(inputs))
        self.tensors = tuple(inputs[name] for name in self.input_names) + (targets,)
        if (any(type(t) is not torch.Tensor or t.device.type != 'cpu' or t.layout != torch.strided
                or t.requires_grad or t.ndim == 0 for t in self.tensors)
                or not len(targets) or any(len(t) != len(targets) for t in self.tensors)):
            raise ValueError('NamedTensorDataset requires nonempty matching dense CPU tensors without gradients')

    def __len__(self):
        return len(self.tensors[-1])

    def __getitem__(self, index):
        return ({name: tensor[index] for name, tensor in zip(self.input_names, self.tensors[:-1])},
                self.tensors[-1][index])
