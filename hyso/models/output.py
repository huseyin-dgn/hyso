"""Explicit, parameter-sharing selection of one GraphModel Tensor output."""

import torch
from torch import nn

from .graph import GraphModel


class GraphTensorOutput(nn.Module):
    """Share a single-input graph; select one named output without copying weights.

    ``task`` is an optional classification training guard (multiclass/binary).
    Without it, selection is also suitable for inference/benchmark feature outputs.
    All graph nodes/outputs still execute. Adapter state_dict keys use ``graph.``;
    save/load graph.state_dict() when an unwrapped graph is the destination.
    Moving/training this wrapper also moves/trains its shared graph.
    """

    def __init__(self, graph, output_name, *, task=None):
        super().__init__()
        if not isinstance(graph, GraphModel):
            raise TypeError("graph must be a GraphModel")
        config = graph.to_config()
        if len(config["inputs"]) != 1:
            raise ValueError("Tensor workflows require exactly one graph input")
        if type(output_name) is not str or output_name not in config["outputs"]:
            raise ValueError("output_name must name an explicit graph output")
        if task is not None:
            if task not in ("multiclass", "binary"):
                raise ValueError(
                    "Only multiclass/binary classification training is supported"
                )
            node = next(
                (
                    n
                    for n in config["nodes"]
                    if n["id"] == config["outputs"][output_name]
                ),
                None,
            )
            if node is None or node["component"] != "classification_head":
                raise ValueError(
                    "Classification training requires a classification_head output"
                )
            classes = node["params"]["num_classes"]
            if (task == "multiclass" and classes < 2) or (
                task == "binary" and classes != 1
            ):
                raise ValueError("Classification head width does not match task")
        self.graph = graph
        self.output_name = output_name

    def forward(self, inputs):
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("GraphTensorOutput expects a single Tensor input")
        value = self.graph(inputs)[self.output_name]
        if not isinstance(value, torch.Tensor):
            raise TypeError("Selected graph output is not a Tensor")
        return value
