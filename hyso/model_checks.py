"""Explicit bounded CPU checks of HYSo Builder configurations and isolated weights.

These are user-model checks, not execution of the project's pytest/native suite.
Preflight never builds a module. Runtime checks use synthetic zero inputs in eval
mode, not a dataset, classification loss, quality or performance measurement.
"""

import copy
import hashlib
import json
import math
import time

import torch

from .models import GraphModelBuilder, ModelBuilder


def config_digest(config):
    """Content identity for the exact supplied JSON architecture; no weight claim."""
    return hashlib.sha256(
        json.dumps(
            config, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def preflight_model(config, *, batch_size=1):
    """Dispatch existing allocation-free schema1/2/3 Core validation unchanged."""
    builder = GraphModelBuilder if config.get("schema_version") == 3 else ModelBuilder
    return builder.preflight(config, batch_size=batch_size)


def _admit(config, report, *, runtime):
    if not report["valid"]:
        raise ValueError(
            "Invalid architecture: " + report["diagnostics"][0]["message_en"]
        )
    bound = report.get("parameter_estimate")
    if type(bound) is not int or bound > 1_000_000:
        raise ValueError("CPU checks require a known Core parameter bound <=1000000")
    if runtime:
        shapes = (
            [s["shape"] for s in config["inputs"].values()]
            if config.get("schema_version") == 3
            else [config.get("input_shape")]
        )
        if any(s is None for s in shapes) or sum(math.prod(s) for s in shapes) > 65536:
            raise ValueError(
                "Synthetic fixture requires concrete inputs totaling <=65536 elements"
            )
        activation = report.get("activation_estimate", {})
        if (
            not activation
            or max(
                activation.get("peak_live_elements", 0),
                activation.get("peak_tensor_elements", 0),
            )
            > 1_000_000
        ):
            raise ValueError(
                "CPU checks require Core activation geometry <=1000000 elements"
            )


def _isolated_model(config, model=None):
    builder = GraphModelBuilder if config.get("schema_version") == 3 else ModelBuilder
    normalized = builder.from_config(config).to_config()
    if model is None:
        return builder.from_config(normalized).build()
    if not hasattr(model, "to_config") or model.to_config() != normalized:
        raise ValueError("Supplied model architecture differs from configuration")
    if any(t.device.type != "cpu" for t in [*model.parameters(), *model.buffers()]):
        raise ValueError(
            "CPU checks require a CPU source model; caller model is not moved"
        )
    candidate = copy.deepcopy(model)
    originals = [*model.modules(), *model.parameters(), *model.buffers()]
    if {id(t) for t in originals} & {
        id(t)
        for t in [*candidate.modules(), *candidate.parameters(), *candidate.buffers()]
    }:
        raise ValueError("Model deepcopy aliases original objects")
    storage = {
        t.untyped_storage().data_ptr()
        for t in [*model.parameters(), *model.buffers()]
        if t.numel()
    }
    if any(
        t.numel() and t.untyped_storage().data_ptr() in storage
        for t in [*candidate.parameters(), *candidate.buffers()]
    ):
        raise ValueError("Model deepcopy aliases original storage")
    return candidate


def count_parameters(config, *, model=None):
    """Explicit <=1M-bound CPU construction/copy and numel; no forward pass."""
    _admit(config, preflight_model(config), runtime=False)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        candidate = _isolated_model(config, model)
        return sum(p.numel() for p in candidate.parameters())


def run_model_checks(
    config,
    *,
    model=None,
    tests=("structure", "forward", "backward", "serialization"),
    model_id=None,
):
    """Run selected checks on a fresh model or isolated copy of supplied weights.

    Source parameters, buffers, gradients, modes and CPU RNG are not changed.
    CPU-only, <=1M Core parameter/activation bounds and <=65536 input elements.
    Backward differentiates sum(eval outputs), not a training loss. It may SKIP
    when no gradient exists. Forward prerequisites run even if not selected.
    Arbitrary custom Python side effects are not sandboxed by deepcopy.
    """
    allowed = ("structure", "forward", "backward", "serialization", "parameters")
    if (
        not isinstance(tests, (tuple, list))
        or not tests
        or any(t not in allowed for t in tests)
        or len(set(tests)) != len(tests)
    ):
        raise ValueError("Choose distinct supported checks: " + ", ".join(allowed))
    report = preflight_model(config)
    result = dict(
        format="hyso.model_checks",
        schema_version=1,
        config_sha256=config_digest(config),
        model_id=model_id,
        initialization="isolated_supplied_weights"
        if model is not None
        else "fresh_seed_0",
        fixture="SYNTHETIC_ZERO_INPUT_CPU_EVAL",
        selected_tests=list(tests),
        preflight=report,
        cases=[],
    )

    def case(name, status, reason, started=None):
        result["cases"].append(
            dict(
                name=name,
                status=status,
                reason=reason,
                seconds=0.0 if started is None else time.perf_counter() - started,
            )
        )

    if "structure" in tests:
        case(
            "structure",
            "PASS" if report["valid"] else "FAIL",
            "Core preflight passed; no forward executed"
            if report["valid"]
            else report["diagnostics"][0]["message_en"],
        )
    selected = [t for t in tests if t != "structure"]
    if not selected:
        return result
    if not report["valid"]:
        for t in selected:
            case(t, "SKIP", "Invalid architecture; inspect retained Core diagnostics")
        return result
    try:
        _admit(config, report, runtime=any(t != "parameters" for t in selected))
    except ValueError as error:
        for t in selected:
            case(t, "SKIP", str(error))
        return result
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        try:
            candidate = _isolated_model(config, model)
            result["state_sha256"] = _state_digest(candidate)
        except Exception as error:
            for t in selected:
                case(t, "FAIL", type(error).__name__ + ": " + str(error))
            return result
        if "parameters" in selected:
            result["parameter_count"] = sum(p.numel() for p in candidate.parameters())
            case(
                "parameters",
                "PASS",
                "Counted real nn.Module parameters; no forward needed",
            )
        runtime = [t for t in selected if t != "parameters"]
        if not runtime:
            return result
        candidate.eval()
        dtype = next(
            (
                t.dtype
                for t in [*candidate.parameters(), *candidate.buffers()]
                if t.is_floating_point()
            ),
            torch.float32,
        )

        def zeros(shape):
            return torch.zeros(1, *shape, dtype=dtype, device="cpu", requires_grad=True)

        inputs = (
            {n: zeros(s["shape"]) for n, s in config["inputs"].items()}
            if config.get("schema_version") == 3
            else zeros(config["input_shape"])
        )
        started = time.perf_counter()
        try:
            output = candidate(inputs)
            values = _outputs(output, config, report)
            if not all(torch.isfinite(v).all() for v in values):
                raise ValueError("Nonfinite forward output")
            if "forward" in runtime:
                case(
                    "forward",
                    "PASS",
                    "Finite outputs on synthetic zero inputs",
                    started,
                )
        except Exception as error:
            for t in runtime:
                case(
                    t,
                    "FAIL" if t == "forward" else "SKIP",
                    "Forward prerequisite failed: " + str(error),
                    started,
                )
            result["error"] = str(error)
            return result
        if "backward" in runtime:
            started = time.perf_counter()
            try:
                objective = sum(v.sum() for v in values)
                if objective.requires_grad:
                    objective.backward()
                gradients = [
                    p.grad for p in candidate.parameters() if p.grad is not None
                ]
                gradients += [
                    v.grad
                    for v in (inputs.values() if isinstance(inputs, dict) else [inputs])
                    if v.grad is not None
                ]
                if gradients and not all(torch.isfinite(g).all() for g in gradients):
                    raise ValueError("Nonfinite observed gradient")
                case(
                    "backward",
                    "PASS" if gradients else "SKIP",
                    "Observed finite gradients for sum of eval outputs; not loss validation"
                    if gradients
                    else "No differentiable gradient produced",
                    started,
                )
            except Exception as error:
                case("backward", "FAIL", str(error), started)
        if "serialization" in runtime:
            started = time.perf_counter()
            try:
                builder = (
                    GraphModelBuilder
                    if config.get("schema_version") == 3
                    else ModelBuilder
                )
                clone = builder.from_config(config).build().to(dtype=dtype)
                clone.load_state_dict(candidate.state_dict(), strict=True)
                clone.eval()
                with torch.no_grad():
                    repeated = _outputs(clone(inputs), config, report)
                if len(values) != len(repeated) or not all(
                    torch.equal(a, b) for a, b in zip(values, repeated)
                ):
                    raise ValueError("Strict state_dict roundtrip changed eval output")
                case(
                    "serialization",
                    "PASS",
                    "Strict state_dict load preserved every configured output on the fixture",
                    started,
                )
            except Exception as error:
                case("serialization", "FAIL", str(error), started)
    return result


def _outputs(output, config, report):
    if config.get("schema_version") == 3:
        if type(output) is not dict or set(output) != set(config["outputs"]):
            raise ValueError("Graph output names differ from configuration")
        values = [output[k] for k in sorted(output)]
    else:
        values = [output]
    if not values or any(not isinstance(v, torch.Tensor) for v in values):
        raise ValueError("Expected nonempty Tensor outputs")
    shapes = (
        [report["outputs"][key]["shape"] for key in sorted(output)]
        if config.get("schema_version") == 3
        else [report["output_shape"]]
    )
    for value, shape in zip(values, shapes):
        if (
            list(value.shape) != [1, *shape[1:]]
            or value.device.type != "cpu"
            or not value.is_floating_point()
        ):
            raise ValueError("Output shape/device/dtype violates the Core contract")
    return values


def _state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(
            json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode()
        )
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
