"""Bounded CPU QA fixtures over the existing HYSo performance APIs.

Prepare once, then execute in fresh child interpreters with an external timeout.
All generated files belong under the supplied external QA directory.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time
from datetime import datetime, timezone

import torch

from hyso import ModelBuilder, GraphModelBuilder, ExperimentManager, evaluate_classification
from hyso.models.output import GraphTensorOutput
from hyso.benchmark import BenchmarkRunner, compare_results
from hyso.checkpoint import save_training_checkpoint, load_training_checkpoint
from hyso.performance import (
    ThroughputProtocol, measure_throughput, MemoryProtocol, measure_memory,
    measure_footprint, analyze_latency,
)
from hyso.analysis import EvidenceRecord, EvidenceIndex, EvidenceScope, AnalysisContext
from hyso.reporting import create_report, report_experiment


SEED = 13516


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def builders():
    sequential = (
        ModelBuilder.sequential((1, 8, 8), num_classes=10, pooling="none")
        .add("conv2d", in_channels=1, out_channels=4, kernel_size=3, padding=1)
        .add("relu")
        .add("conv2d", in_channels=4, out_channels=8, kernel_size=3, padding=1)
        .add("relu")
        .add("flatten")
    )
    graph = (
        GraphModelBuilder({"image": [1, 8, 8]})
        .add("stem", "conv2d", ["image"], in_channels=1, out_channels=4, kernel_size=3, padding=1)
        .add("activation", "relu", ["stem"])
        .add("left", "conv2d", ["activation"], in_channels=4, out_channels=4, kernel_size=3, padding=1)
        .add("left_relu", "relu", ["left"])
        .add("right", "pointwise_conv2d", ["activation"], in_channels=4, out_channels=4)
        .add("right_relu", "relu", ["right"])
        .add("fusion", "concat", ["left_relu", "right_relu"], axis=1)
        .add("flat", "flatten", ["fusion"])
        .add("head", "classification_head", ["flat"], in_features=512, num_classes=10, pooling="none")
        .outputs(logits="head")
    )
    return dict(sequential=sequential, graph=graph)


def fresh(kind, config):
    builder = GraphModelBuilder if kind == "graph" else ModelBuilder
    model = builder.from_config(config).build()
    if kind == "graph":
        return GraphTensorOutput(model, "logits", task="multiclass")
    return model


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(str((name, tuple(tensor.shape), str(tensor.dtype))).encode())
        digest.update(tensor.detach().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def summarize(samples):
    if not samples or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError("positive finite nonempty millisecond samples required")
    ordered = sorted(samples)
    def quantile(q):
        position = (len(ordered) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return dict(count=len(samples), unit="ms", min=min(samples), p50=statistics.median(samples),
                p90=quantile(.9), p95=quantile(.95) if len(samples) >= 100 else None,
                mean=statistics.mean(samples), std=statistics.stdev(samples) if len(samples) > 1 else None,
                samples_ms=samples, quantile_method="linear_(n-1)*q",
                tail_limit="descriptive correlated samples; no confidence or superiority claim")


def timed(call):
    start = time.perf_counter_ns()
    value = call()
    return value, (time.perf_counter_ns() - start) / 1e6


def prepare(root, revision, hardware):
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    protocol = dict(
        id="hyso-13.5.16-cpu-fp32-v1", start_id="20261116-stage13-5-16",
        prepared_at_utc=datetime.now(timezone.utc).isoformat(), revision=revision,
        environment=dict(python=platform.python_version(), executable=os.sys.executable,
                         torch=str(torch.__version__), os=platform.platform(),
                         cuda_build=torch.version.cuda, hardware=hardware, torch_threads=1,
                         interop_threads=torch.get_num_interop_threads()),
        device="cpu", precision="fp32", seed=SEED, batches=[1, 8, 32], chw=[1, 8, 8], classes=10,
        configurations={k: b.to_config() for k, b in builders().items()},
        independent_child_runs=2, inference=dict(warmup=5, iterations=40, repeats=3),
        throughput=dict(warmup=5, batch_count=40, repeats=3),
        construction_samples=5, first_forward_samples=1,
        training=dict(batch=8, warmup=2, samples=20, optimizer="SGD", lr=.001),
        checkpoint_samples=5, evaluation_samples=3, evaluation_count=32,
        reporting_samples=3, comparison_samples=5,
        resource=dict(interval_seconds=.005, max_samples=1000, forwards=100, batch=32),
        clock="perf_counter_ns", synchronization="CPU synchronous host; CUDA excluded until separate admission",
        child_timeout_seconds=180, measurement_deadline_seconds=60,
        validity=["complete terminal summary and exit zero", "finite positive times",
                  "raw sample count equals protocol", "inference state and input hashes stable",
                  "strict checkpoint equality", "CUDA remains uninitialized"],
        limitations=["P95 requires at least 100 samples in QA summary; built-in descriptive P95 retained",
                     "warm repeats share a model and process; fresh children are separate observations",
                     "first forward uses a fresh model per batch in an already imported process",
                     "construction includes from_config validation, build and GraphTensorOutput wrapping",
                     "RSS is process-wide; sampled maximum and lifetime HWM have separate scopes",
                     "training mutates only a dedicated QA clone; no learning-quality claim",
                     "CUDA gate is evaluated after CPU correctness; no CUDA benchmark is implicit"],
    )
    write(root / "protocol.json", protocol)
    write(root / "protocol-lock.json", dict(sha256=sha(root / "protocol.json"), source_sha256=sha(__file__)))
    assert not torch.cuda.is_initialized()
    return protocol


def run(root, name):
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf8"))
    lock = json.loads((root / "protocol-lock.json").read_text())
    assert sha(protocol_path) == lock["sha256"] and sha(__file__) == lock["source_sha256"]
    torch.set_num_threads(protocol["environment"]["torch_threads"])
    torch.random.default_generator.manual_seed(SEED)
    output = root / name
    output.mkdir()
    started = time.perf_counter()
    summary = dict(name=name, pid=os.getpid(), started_at_utc=datetime.now(timezone.utc).isoformat(),
                   protocol_sha256=sha(protocol_path), source_sha256=sha(__file__), models={}, comparisons={})
    records, inference = [], {}
    for kind, config in protocol["configurations"].items():
        directory = output / kind
        directory.mkdir()
        construction = []
        for _ in range(protocol["construction_samples"]):
            torch.random.default_generator.manual_seed(SEED)
            model, elapsed = timed(lambda: fresh(kind, config))
            construction.append(elapsed)
        model.eval()
        initial_hash = state_hash(model)
        item = dict(config=config, architecture_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                    state_sha256=initial_hash, device="cpu", dtype="torch.float32",
                    parameters=sum(p.numel() for p in model.parameters()),
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                    construction=summarize(construction), batches={})
        generator = torch.Generator().manual_seed(SEED)
        for batch in protocol["batches"]:
            inputs = torch.randn(batch, 1, 8, 8, generator=generator)
            input_hash = hashlib.sha256(inputs.numpy().tobytes()).hexdigest()
            first_model = fresh(kind, config).eval()
            first_model.load_state_dict(model.state_dict(), strict=True)
            with torch.inference_mode():
                first_output, first_ms = timed(lambda: first_model(inputs))
            assert first_output.shape == (batch, 10) and torch.isfinite(first_output).all()
            settings = protocol["inference"]
            result = BenchmarkRunner(model, example_input=inputs, device="cpu", precision="fp32",
                                     warmup_iterations=settings["warmup"], measure_iterations=settings["iterations"],
                                     repeats=settings["repeats"], seed=SEED, model_name=kind, model_config=config).run()
            result.to_json(directory / f"inference-b{batch}.json")
            raw = result.to_dict()
            samples = sum(raw["timing"]["latency_samples_ms"], [])
            inference[kind, batch] = result
            tail = analyze_latency(result, quantiles=(.9, .95))
            tail.to_json(directory / f"tail-b{batch}.json")
            settings = protocol["throughput"]
            throughput = measure_throughput(model, inputs, protocol=ThroughputProtocol(
                warmup_iterations=settings["warmup"], batch_count=settings["batch_count"], repeats=settings["repeats"]))
            throughput.to_json(directory / f"throughput-b{batch}.json")
            assert state_hash(model) == initial_hash
            assert hashlib.sha256(inputs.numpy().tobytes()).hexdigest() == input_hash
            item["batches"][str(batch)] = dict(first_forward_ms=first_ms, input_sha256=input_hash,
                input_shape=list(inputs.shape), output_shape=list(first_output.shape), latency=summarize(samples),
                serial_service_images_per_second=raw["throughput"]["images_per_sec"],
                throughput=throughput.to_dict()["measurements"])
            for feature, filename in (("P01", f"inference-b{batch}.json"), ("P02", f"throughput-b{batch}.json")):
                path = directory / filename
                records.append(EvidenceRecord.from_json(path, feature_id=feature, expected_sha256=sha(path),
                    scope=EvidenceScope(model_id=kind, source_revision=protocol["revision"])))

        inputs = torch.randn(8, 1, 8, 8, generator=generator)
        labels = torch.arange(8) % 10
        training = fresh(kind, config).train()
        training.load_state_dict(model.state_dict(), strict=True)
        optimizer = torch.optim.SGD(training.parameters(), lr=.001)
        forward_backward, optimizer_step = [], []
        settings = protocol["training"]
        for index in range(settings["warmup"] + settings["samples"]):
            optimizer.zero_grad(set_to_none=True)
            def backward():
                loss = torch.nn.functional.cross_entropy(training(inputs), labels)
                loss.backward()
                return loss
            loss, elapsed = timed(backward)
            assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in training.parameters())
            _, step_ms = timed(optimizer.step)
            if index >= settings["warmup"]:
                forward_backward.append(elapsed)
                optimizer_step.append(step_ms)
        assert state_hash(training) != initial_hash and state_hash(model) == initial_hash
        item.update(forward_backward=summarize(forward_backward), optimizer_step=summarize(optimizer_step),
                    training_final_state_sha256=state_hash(training))

        saves, loads, sizes = [], [], []
        restored = fresh(kind, config)
        for index in range(protocol["checkpoint_samples"]):
            path = directory / f"checkpoint-{index}.pth"
            _, elapsed = timed(lambda: save_training_checkpoint(path, model, include_rng=False, graph_contract=kind == "graph"))
            saves.append(elapsed)
            _, elapsed = timed(lambda: load_training_checkpoint(path, restored, map_location="cpu", expected_sha256=sha(path)))
            loads.append(elapsed)
            assert state_hash(restored) == initial_hash
            restored.eval()
            with torch.inference_mode():
                assert torch.equal(restored(inputs), model(inputs))
            sizes.append(path.stat().st_size)
        item.update(checkpoint_save=summarize(saves), checkpoint_load=summarize(loads), checkpoint_bytes=sizes)
        footprint = measure_footprint(model, artifact_path=path, artifact_format="hyso.training_checkpoint", expected_sha256=sha(path))
        footprint.to_json(directory / "footprint.json")

        resource_inputs = torch.randn(32, 1, 8, 8, generator=generator)
        def workload():
            with torch.inference_mode():
                for _ in range(protocol["resource"]["forwards"]):
                    model(resource_inputs)
        memory = measure_memory(workload, protocol=MemoryProtocol(interval_seconds=.005, max_samples=1000))
        memory.to_json(directory / "memory.json")
        item["memory"] = memory.to_dict()["measurements"]
        item["cpu_utilization"] = dict(status="NOT_MEASURED", reason="short fixture window and process-time resolution do not support a reliable utilization estimate")
        for feature, filename in (("P10", "memory.json"), ("P11", "footprint.json")):
            path = directory / filename
            records.append(EvidenceRecord.from_json(path, feature_id=feature, expected_sha256=sha(path),
                scope=EvidenceScope(model_id=kind)))

        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(resource_inputs, torch.arange(32) % 10), batch_size=8, shuffle=False, num_workers=0)
        evaluations = []
        for _ in range(protocol["evaluation_samples"]):
            evaluation, elapsed = timed(lambda: evaluate_classification(model, loader, class_names=[str(x) for x in range(10)], split="synthetic-qa"))
            evaluations.append(elapsed)
        evaluation.to_json(directory / "evaluation.json")
        item["evaluation"] = summarize(evaluations)
        assert state_hash(model) == initial_hash
        summary["models"][kind] = item
        assert time.perf_counter() - started < protocol["measurement_deadline_seconds"]

    for batch in protocol["batches"]:
        samples = []
        for _ in range(protocol["comparison_samples"]):
            comparison, elapsed = timed(lambda: compare_results(inference["sequential", batch], inference["graph", batch]))
            samples.append(elapsed)
        assert comparison["status"] == "compatible"
        summary["comparisons"][str(batch)] = dict(result=comparison, overhead=summarize(samples))
    index = EvidenceIndex(records)
    index.to_json(output / "evidence-index.json")
    analysis = index.analyze(context=AnalysisContext(workload_id=protocol["id"], task="synthetic inference benchmark",
        model=dict(ids=list(summary["models"])), dataset=dict(kind="deterministic synthetic tensors"), source_revision=protocol["revision"]))
    analysis.to_json(output / "analysis.json")
    report_samples = []
    for iteration in range(protocol["reporting_samples"]):
        def export():
            report = create_report(analysis, language="en", level="expert")
            for suffix, method in (("json", report.to_json), ("md", report.to_markdown), ("html", report.to_html)):
                method(output / f"report-en-{iteration}.{suffix}")
        _, elapsed = timed(export)
        report_samples.append(elapsed)
    report = create_report(analysis, language="tr", level="expert")
    for suffix, method in (("json", report.to_json), ("md", report.to_markdown), ("html", report.to_html)):
        method(output / ("report-tr." + suffix))
    summary["reporting"] = summarize(report_samples)
    store = output / "store"
    manager = ExperimentManager(store)
    manager.close()
    ExperimentManager.migrate_store(store)
    manager = ExperimentManager(store)
    artifacts = []
    try:
        experiment = manager.create_experiment(protocol["id"])
        for kind, item in summary["models"].items():
            current = manager.create_run(experiment["id"], "benchmark", config=dict(protocol=protocol, model=item["architecture_sha256"]),
                                         provenance=dict(raw_samples=f"{kind}/inference-b*.json", limitations=protocol["limitations"]))
            current.start()
            for batch, data in item["batches"].items():
                current.log_metric(name=f"latency_b{batch}", value=data["latency"]["p50"], unit="ms", source="measured_cpu_qa", step=0, step_kind="observation")
                current.log_metric(name=f"service_rate_b{batch}", value=data["serial_service_images_per_second"], unit="images/s", source="derived_serial_service_rate", step=0, step_kind="observation")
            for path in sorted((output/kind).glob("*.json")):
                artifact = current.add_artifact(path, kind="benchmark_qa", ownership="copy")
                artifacts.append(dict(id=artifact["id"], sha256=sha(path)))
            for path in (root/"protocol.json", output/"analysis.json", output/"report-en-0.json", output/"report-tr.json"):
                artifact = current.add_artifact(path, kind="report", ownership="copy")
                artifacts.append(dict(id=artifact["id"], sha256=sha(path)))
            current.finish()
        before = [manager.verify_artifact(a["id"]) for a in artifacts]
    finally:
        manager.close()
    manager = ExperimentManager(store)
    try:
        after = [manager.verify_artifact(a["id"]) for a in artifacts]
        assert all(row["status"] == "verified" for row in before+after)
        assert all(manager.get_artifact(a["id"])["expected_sha256"] == a["sha256"] for a in artifacts)
        summary["experiment"] = dict(id=experiment["id"], runs=manager.list_runs(experiment_id=experiment["id"]),
                                      artifacts=artifacts, before=before, after=after)
        report_experiment(manager, experiment["id"], language="en", level="expert").to_json(output/"experiment-report.json")
    finally:
        manager.close()
    summary.update(status="complete", elapsed_seconds=time.perf_counter()-started,
                   finished_at_utc=datetime.now(timezone.utc).isoformat(), end_id=name+"-complete", cuda_initialized=torch.cuda.is_initialized())
    assert summary["elapsed_seconds"] < protocol["measurement_deadline_seconds"] and not summary["cuda_initialized"]
    write(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--name", default="cpu-1")
    parser.add_argument("--revision")
    parser.add_argument("--hardware", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.revision, json.loads(args.hardware.read_text(encoding="utf-8-sig")))
    else:
        result = run(args.root, args.name)
        print(json.dumps(dict(status=result["status"], pid=result["pid"], elapsed_seconds=result["elapsed_seconds"])))
