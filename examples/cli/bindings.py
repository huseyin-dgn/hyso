"""Trusted retained binding factory. No model, forward, fit or device access."""
from pathlib import Path


def retained(*, path, sha256, model_id='model', features=('P22',)):
    from hyso.analysis import ArtifactSource, NativeBinding, RetainedSource
    from hyso.analysis._codec import read_json
    payload, digest, size = read_json(path, expected_sha256=sha256)
    return [NativeBinding('retained-example', (model_id,), tuple(features), RetainedSource(payload),
                          artifacts=(ArtifactSource(str(Path(path).resolve()), digest, size),))]
