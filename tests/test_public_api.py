"""Fixed public contracts from P0.1-A, independent of the current source AST."""

import importlib
import inspect

import pytest


CANONICAL = {
    "EBO": "hyso.models.bcam",
    "ELO": "hyso.models.bcam",
    "CEMO": "hyso.models.WithSE",
    "EMA": "hyso.models.WithSE",
    "CNNSENormalWithFit": "hyso.models.fit",
    "build_transforms": "hyso.augmentation.augmentation",
    "get_loaders": "hyso.augmentation.augmentation",
    "EarlyStopping": "hyso.callbacks.callbacks",
    "ModelCheckpoint": "hyso.callbacks.callbacks",
    "LRSchedulerCallback": "hyso.callbacks.callbacks",
}

# Explicit snapshots: do not derive expected defaults from the implementation.
SIGNATURES = {
    "EBO.__init__": "(self, input_channels=3, num_classes=10, conv_channels=[32, 64, 128, 256], residual_per_block=2, p_drop=0.0, att_reduction=16, norm_type='auto', device=None, use_depthwise=False)",
    "EBO.callbacks": "(self, optimizer='Adam', optimizer_params=None, scheduler=None, scheduler_params=None, model_checkpoint=False, earlystop=None, lr=0.001, weight_decay=0.0001)",
    "EBO.fit": "(self, train_loader, val_loader=None, epochs=10, lr=0.001, optimizer_name='adamw', weight_decay=0.0001, use_ema=False, use_amp=True, grad_clip=None, accum_steps=1, scheduler_name=None, scheduler_params=None, max_lr=None)",
    "ELO.__init__": "(self, input_channels=3, num_classes=10, conv_channels=[64, 128, 256, 512], residual_per_block=2, p_drop=0.0, att_reduction=16, norm_type='auto', use_depthwise=False)",
    "ELO.forward": "(self, x)",
    "CEMO.__init__": "(self, input_channels: int = 3, num_classes: int = 10, conv_channels=[64, 128, 256], residuals_per_stage: int = 2, se_reduction: int = 8, p_drop: float = 0.1, dropout_fc: float = 0.3, use_ema: bool = False, ema_decay: float = 0.999, device: str = None)",
    "CEMO.forward": "(self, x)",
    "CEMO.callbacks": "(self, optimizer='Adam', lr=0.001, weight_decay=0.0001, model_checkpoint=False, earlystop=None, scheduler=None, scheduler_params=None)",
    "CEMO.fit": "(self, train_loader, val_loader=None, epochs=5, lr=0.001, weight_decay=0.0001, use_ema=False)",
    "EMA.__init__": "(self, model, decay=0.999, device=None)",
    "EMA.to": "(self, device)",
    "EMA.update": "(self, model)",
    "CNNSENormalWithFit.__init__": "(self, *args, **kwargs)",
    "CNNSENormalWithFit.callbacks": "(self, optimizer='Adam', optimizer_params=None, scheduler=None, scheduler_params=None, model_checkpoint=False, earlystop=None, lr=0.001, weight_decay=0.0001)",
    "CNNSENormalWithFit.fit": "(self, train_loader, val_loader=None, epochs=10, lr=0.001, use_ema=False)",
    "build_transforms": "(img_size=(224, 224), augment='basic', extra_ops=None, channels=3)",
    "get_loaders": "(train_path, val_path=None, batch_size=32, img_size=(224, 224), augment='basic', extra_ops=None, num_workers=0, channels=3)",
    "EarlyStopping.__init__": "(self, patience=7, delta=0.0, verbose=True, restore_best_weights=True)",
    "EarlyStopping.__call__": "(self, val_loss, model)",
    "EarlyStopping.restore_weights": "(self, model)",
    "ModelCheckpoint.__init__": "(self, filepath='best_model.pth', verbose=True)",
    "ModelCheckpoint.__call__": "(self, epoch, model, val_loss)",
    "LRSchedulerCallback.__init__": "(self, scheduler)",
    "LRSchedulerCallback.step": "(self, val_loss=None)",
}


def canonical_symbol(name):
    return getattr(importlib.import_module(CANONICAL[name]), name)


@pytest.mark.parametrize("name", CANONICAL)
def test_canonical_import(name):
    symbol = canonical_symbol(name)
    assert callable(symbol)
    assert symbol.__name__ == name
    assert symbol.__module__ == CANONICAL[name]


@pytest.mark.parametrize(
    "package,name",
    [("hyso.models", name) for name in (
        "EBO", "ELO", "CEMO", "EMA", "CNNSENormalWithFit"
    )] + [("hyso.augmentation", "build_transforms"),
         ("hyso.augmentation", "get_loaders")],
)
def test_documented_package_reexport(package, name):
    assert getattr(importlib.import_module(package), name) is canonical_symbol(name)


@pytest.mark.parametrize("path,expected", SIGNATURES.items(), ids=SIGNATURES)
def test_public_signature_and_defaults(path, expected):
    name, _, method = path.partition(".")
    symbol = canonical_symbol(name)
    target = getattr(symbol, method) if method else symbol
    assert str(inspect.signature(target)) == expected
    if method == "__init__":
        # Class-call signature must also expose the documented constructor.
        parameters = list(inspect.signature(target).parameters.values())[1:]
        assert inspect.signature(symbol) == inspect.Signature(parameters)
