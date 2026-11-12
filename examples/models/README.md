# M3 CPU examples

Install HYSo first (`python -m pip install -e .`). From the checkout root:

```console
python examples/models/m3_workflows.py resnet
python examples/models/m3_workflows.py variant --depth 34
python examples/models/m3_workflows.py densenet
python examples/models/m3_workflows.py block
python examples/models/m3_workflows.py attention
python examples/models/m3_workflows.py builder
python examples/models/m3_workflows.py training
python examples/models/m3_workflows.py state_dict
```

All use CPU and tiny synthetic inputs. Training is one two-sample optimizer step;
it is an API example, not a model-quality result. DenseNet uses explicit growth=4
as a small width variant. ResNet examples retain full named topology with a
one-channel input and three-class head. Large depths only perform an eval forward.
The state_dict example uses a temporary directory and strict weights-only load;
no acceptance checkpoint is read or written. The default scenario is Builder.

See [architecture guide](../../docs/guides/model-architectures.md) for defaults,
normalization, JSON contracts and epoch-resume limitations.
