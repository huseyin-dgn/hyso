"""Explicit named tensor batches for bounded two-input Graph training/resume."""
import torch
from torch.utils.data import Dataset


class NamedTensorDataset(Dataset):
    """Two named CPU tensor inputs and one target tensor; no transforms or RNG.

    Names bind tensors explicitly. Dictionary insertion order is immaterial.
    Checkpoint admission additionally validates the exact Graph/task/dtype contract.

    Args:
        inputs: Tensor inputs matching the selected runtime input names, shapes and dtypes. Required.
        targets: Classification targets aligned with the named input samples; no automatic task/loss
            inference. Required.
    """
    def __init__(self, inputs, targets) -> 'None':
        """Initialize NamedTensorDataset from the supplied settings or evidence.

        Args:
            inputs: Tensor inputs matching the selected runtime input names, shapes and dtypes. Required.
            targets: Classification targets aligned with the named input samples; no automatic task/loss
                inference. Required.
        """
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


class NamedTargetDataset(Dataset):
    """One Tensor or two named CPU inputs; exactly two named classification targets.

    Automatic multi-loss resume requires the closed CPU fp32 loader/config profile.

    Args:
        inputs: Tensor inputs matching the selected runtime input names, shapes and dtypes. Required.
        targets: Classification targets aligned with the named input samples; no automatic task/loss
            inference. Required.
    """
    def __init__(self, inputs, targets) -> 'None':
        """Initialize NamedTargetDataset from the supplied settings or evidence.

        Args:
            inputs: Tensor inputs matching the selected runtime input names, shapes and dtypes. Required.
            targets: Classification targets aligned with the named input samples; no automatic task/loss
                inference. Required.
        """
        if (type(targets) is not dict or len(targets) != 2
                or any(type(k) is not str or not k for k in targets)):
            raise ValueError('two explicit target names required')
        self.target_names = tuple(sorted(targets))
        if type(inputs) is dict:
            if len(inputs) != 2 or any(type(k) is not str or not k for k in inputs):
                raise ValueError('exactly two explicit input names required')
            self.input_names = tuple(sorted(inputs))
            input_tensors = tuple(inputs[name] for name in self.input_names)
        else:
            input_tensors = (inputs,)
        self.tensors = input_tensors + tuple(targets[name] for name in self.target_names)
        if (any(type(t) is not torch.Tensor or t.device.type != 'cpu' or t.layout != torch.strided
                or t.requires_grad or t.ndim == 0 for t in self.tensors)
                or not len(self.tensors[0]) or any(len(t) != len(self.tensors[0]) for t in self.tensors)):
            raise ValueError('matching nonempty dense CPU tensors without gradients required')

    def __len__(self):
        return len(self.tensors[0])

    def __getitem__(self, index):
        named = hasattr(self, 'input_names')
        inputs = ({name:t[index] for name,t in zip(self.input_names,self.tensors[:2])}
                  if named else self.tensors[0][index])
        return inputs, {name:t[index] for name,t in zip(self.target_names,self.tensors[2 if named else 1:])}
