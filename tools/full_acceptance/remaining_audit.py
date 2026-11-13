"""Offline cross-run scope reconciliation for P13/P23/P30. Never executes stages."""
from pathlib import Path
import argparse
import collections
import csv
import json

from .core import read, record, sha, digest, publish, now


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def pinned(ref):
    actual = record(ref["path"])
    require(actual["sha256"] == ref["sha256"], "artifact changed: " + ref["path"])
    if "bytes" in ref:
        require(actual["bytes"] == ref["bytes"], "artifact size changed: " + ref["path"])
    return actual


def run_source(path):
    """Validate historical snapshot and retained raw artifacts, without live drift checks."""
    path = Path(path).resolve(strict=True)
    manifest = path / "manifest.json"
    mh = sha(manifest)
    require(manifest.with_suffix(".sha256").read_text(encoding="ascii").strip() == mh,
            "manifest seal mismatch")
    m, status, results, matrix = [read(path / name) for name in
        ("manifest.json", "run-status.json", "stage-results.json", "P01-P35-final-acceptance-matrix.json")]
    require(Path(m["output"]).resolve() == path and m["format"] == "hyso.full_acceptance.manifest",
            "manifest output/format mismatch")
    require(status.get("controller_active") is False and status.get("manifest_sha256") == mh
            and results.get("manifest_sha256") == mh and m["run_id"] == status.get("run_id") == results.get("run_id"),
            "historical controller identity/terminal mismatch")
    snapshot = Path(m["source_snapshot"]["root"]).resolve()
    files = m["source_snapshot"]["files"]
    require(snapshot.is_relative_to(path) and digest(files) == m["source_snapshot"]["tree_sha256"],
            "historical snapshot seal mismatch")
    require({p.relative_to(snapshot).as_posix() for p in snapshot.rglob("*") if p.is_file()} == set(files),
            "historical snapshot file set mismatch")
    for rel, expected in files.items():
        source = (snapshot / rel).resolve()
        require(source.is_relative_to(snapshot) and sha(source) == expected, "historical source changed: " + rel)
    stages = {x["id"]: x for x in results["experiments"]}
    experiments = {x['id']: x for x in m['experiments']}
    require(len(stages) == len(results["experiments"]), "duplicate historical stage")
    evidence = []
    for ident, row in stages.items():
        if row.get("attempted") is not True:
            continue
        sup_path = path / "stages" / ident / "supervisor-receipt.json"
        sup = read(sup_path)
        require(sup == row.get("supervisor") and sup.get("manifest_sha256") == mh
                and sup.get("experiment") == ident, "historical supervisor identity mismatch: " + ident)
        if row["status"] in ("PASS", "PARTIAL"):
            require(experiments[ident]['authorization']['approved'] is True
                    and sup.get('command') == experiments[ident]['command'], 'stage approval/command mismatch')
            terminal_path = path / 'stages' / ident / 'result.json'
            terminal = read(terminal_path)
            require(terminal.get('manifest_sha256') == mh and terminal.get('id') == ident
                    and all(row.get(k) == v for k,v in terminal.items()), 'worker/controller result mismatch: '+ident)
            evidence.append(record(terminal_path))
            require(type(sup.get("exit_code")) is int and sup["exit_code"] == 0
                    and sup.get("native_crash") is False and sup.get("owned_job_only") is True
                    and type(sup.get("cleanup", {}).get("active_processes")) is int
                    and sup["cleanup"]["active_processes"] == 0, "unclean observed stage: " + ident)
        evidence.append(record(sup_path))
        evidence.extend(pinned(ref) for ref in row.get("evidence", []))
    features = {x["id"]: x for x in matrix["features"]}
    require(len(matrix["features"]) == 35 and set(features) == {f"P{i:02}" for i in range(1, 36)},
            "exactly P01-P35 required")
    assets = [pinned(ref) for ref in [*m['assets']['models'].values(), m['assets']['dataset'],
        m['assets']['split_indices'], m['assets']['split_manifest']]]
    return dict(path=str(path), manifest=record(manifest), state=status,
        source_files_verified=len(files), assets=assets, stages=stages, features=features,
        artifacts=[record(path / name) for name in ("run-status.json", "stage-results.json",
            "P01-P35-final-acceptance-matrix.json")], evidence=evidence)


def feature_rows(base, long, roofline):
    require(long["stages"]["10.stability"]["criteria"].get("P31_uninterrupted_7200") is True
            and long["stages"]["10.stability"]["criteria"].get("no_sampling_gaps") is True
            and long["stages"]["10.stability"]["status"] == "PASS"
            and long["stages"]["11.memory"]["status"] == "PASS"
            and long["stages"]["11.memory"]["criteria"].get("actual_long_run_series") is True,
            "P31/P32 same-run criteria missing")
    work, = [r["value"] for r in long["stages"]["10.stability"]["metrics"] if r["name"] == "actual_work_seconds"]
    require(type(work) in (float, int) and work >= 7200, "P31 actual work below 7200 seconds")
    require(roofline["stages"]["06.roofline"]["status"] == "PASS"
            and roofline["stages"]["06.roofline"]["criteria"].get("P18_calibration_screen") is True,
            "P18 recorded calibration criterion missing")
    rows = []
    for number in range(1, 36):
        ident = f"P{number:02}"
        source = long if ident in ("P31", "P32") else roofline if ident == "P18" else base
        feature = source["features"][ident]
        expected = "PARTIAL" if ident in ("P13", "P23", "P30") else "PASS"
        require(feature["decision"] == expected, ident + " differs from the evidence-backed expected scope")
        rows.append(dict(id=ident, name=feature["name"], decision=feature["decision"], source_run=source["path"],
            manifest_sha256=source["manifest"]["sha256"], current_execution=feature["current_execution"],
            experiments=feature["experiments"], decision_basis=feature["decision_basis"],
            retained_scope=feature["retained_scope"], prior_decision_source=feature["prior_decision_source"],
            source_preservation=source["state"].get("preservation_status"),
            scope="cross-run scoped evidence; historical retention is not a new execution"))
    return rows


def training_archives(index):
    """Reopen pinned historical P23 payloads; no factories, training or workers."""
    from hyso.performance import TrainingMemoryResult, IsolatedTrainingMemoryResult
    from unittest.mock import patch
    from contextlib import ExitStack
    from hyso.performance import isolated_training_memory
    import torch
    import subprocess
    calls = []
    def forbidden(*a, **k):
        calls.append(True)
        raise AssertionError('offline reader attempted execution')
    root = Path(__file__).resolve().parents[2]
    rows = []
    with ExitStack() as guards:
        for obj, name in ((torch, 'load'), (torch.nn.Module, '_call_impl'), (torch.cuda, '_lazy_init'),
                          (subprocess, 'Popen'), (isolated_training_memory, '_launch')):
            guards.enter_context(patch.object(obj, name, forbidden))
        for item in read(index)['checks']:
            if item['format'] not in ('hyso.performance.training_memory', 'hyso.performance.isolated_training_memory'):
                continue
            artifact = (root / item['path']).resolve()
            pinned(dict(path=str(artifact), sha256=item['artifact_sha256']))
            payload = read(artifact)
            for key in item['json_pointer'].split('/')[1:]:
                payload = payload[int(key)] if isinstance(payload, list) else payload[key]
            import hashlib
            require(hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                ensure_ascii=False).encode()).hexdigest() == item['payload_sha256'], 'saved payload hash mismatch')
            cls = IsolatedTrainingMemoryResult if 'isolated' in item['format'] else TrainingMemoryResult
            result = cls(payload)
            require(result.to_dict() == payload, 'saved P23 export changed')
            result.diagnostics()
            rows.append(dict(path=str(artifact), json_pointer=item['json_pointer'],
                sha256=item['payload_sha256'], status='PASS'))
    require(not calls, 'offline execution guard reached')
    return dict(passed=len(rows), failed=0, execution_calls=0, index=record(index), checks=rows)


def build(args):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("audit requires a new directory")
    for source in (args.base_run, args.long_run, args.roofline_run, args.p13_recovery):
        source = source.resolve()
        require(output != source and source not in output.parents and output not in source.parents,
                "output overlaps historical evidence")
    base, long, roofline = [run_source(p) for p in (args.base_run, args.long_run, args.roofline_run)]
    rows = feature_rows(base, long, roofline)
    from hyso.performance.result import LayerOverheadResult
    calibration_path = args.base_run / 'stages/06.layer/ownership-calibration.json'
    calibration = LayerOverheadResult(read(calibration_path)).to_dict()
    require(calibration['measurements']['status'] == 'estimated', 'historical CPU calibration not estimated')
    archived_training = training_archives(args.p23_reader_index)
    # Successful hook-free CUDA control never closes the crashing activation path.
    refs = read(args.p23_identities)
    identities = [pinned(ref) for ref in refs]
    crash_ref, = [r for r in identities if Path(r["path"]).name == "c-native-crash.json"]
    from hyso.performance import IsolatedTrainingMemoryResult
    crash = IsolatedTrainingMemoryResult(read(crash_ref["path"])).to_dict()
    require(crash["outcome"] == "native_crash" and crash["measurement"] is None
            and crash["receipt"]["exit_code"] & 0xffffffff == 0xC0000409, "historical native crash identity differs")
    recovery = read(args.p13_recovery / "p13-recovery-audit.json")
    require(recovery["status"] == "PARTIAL" and not recovery["errors"]
            and recovery["inputs_unchanged"] is True and recovery["capture_receipt_created"] is False,
            "P13 recovery audit not verified")
    pinned(recovery['manifest'])
    capture_manifest = read(recovery['manifest']['path'])
    base_manifest = read(base['manifest']['path'])
    require(capture_manifest['assets']['models']['B']['sha256'] == base_manifest['assets']['models']['B']['sha256']
            and capture_manifest['hardware']['gpu_uuid'] == base_manifest['hardware']['gpu_uuid'],
            'historical CPU/CUDA Model B/device binding differs')
    p30 = read(args.p30_config)
    assets = {key:pinned(p30[key]) for key in ("factory_source", "checkpoint", "video")}
    result = dict(created_at=now(), scope="UNION_OF_RETAINED_SCOPES_AND_DISTINCT_RUNS", suite_complete=False,
        counts=dict(collections.Counter(x["decision"] for x in rows)), features=rows,
        sources=dict(base=base, long=long, roofline=roofline),
        p13=dict(decision="PARTIAL", audit=record(args.p13_recovery / "p13-recovery-audit.json"),
                 controller_terminal_verified=recovery["controller_terminal_verified"],
                 historical_CPU_calibration=record(calibration_path),
                 CPU_CUDA_link="same Model B checkpoint; separate manifests; current 06.layer acceptance absent"),
        p23=dict(decision="PARTIAL", identities=identities, exit_code="0xC0000409",
            completed_measurement=None, archived_reader_validation=archived_training,
            cleanup_active_processes=crash["receipt"]["job"]["active_processes"],
            proven="native abnormal exit with saved_activation_lifetime enabled; no completed measurement",
            hypotheses=["native hook/autograd/runtime interaction; no causal stack or dump establishes it"],
            root_cause="UNRESOLVED; Python version-check fix and hook-free success are not a causal proof"),
        p30=dict(decision="PARTIAL", config=record(args.p30_config), assets=assets,
            model_id=p30["model_id"], label_semantics=p30["label_semantics"],
            real_video_opened=False, physical_camera_opened=False,
            camera_provenance=p30.get("camera_provenance"), accuracy_claim=False),
        limitations=["base run preservation FAIL and ABORTED remain visible; independently hashed stage evidence retained",
            "source snapshots are historical; none certifies the newly edited implementation natively",
            "hashes are integrity checks, not signed provenance or proof of physical sensor identity"])
    output.mkdir(parents=True, exist_ok=False)
    publish(output / "remaining-scope-audit.json", result, new=True)
    with (output / "P01-P35-scope.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k:json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                         for k,v in row.items()} for row in rows)
    lines = ["# P01–P35 evidence reconciliation", "", "32 PASS / 3 PARTIAL across retained scopes and separate runs.",
        "No completed single-run acceptance. Base ABORTED/preservation FAIL is retained explicitly.", "",
        "| Feature | Decision | Evidence run |", "|---|---|---|"]
    lines += [f"| {r['id']} | {r['decision']} | {r['source_run']} |" for r in rows]
    (output / "P01-P35-scope.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-run", "long-run", "roofline-run", "p13-recovery", "p23-identities", "p30-config", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument('--p23-reader-index', type=Path,
        default=Path(__file__).resolve().parents[2] / 'outputs/p23-reader-p18-offline-20261028-r01/p23-reader-verification.json')
    result = build(parser.parse_args(argv))
    print(json.dumps(dict(counts=result["counts"], suite_complete=False)))


if __name__ == "__main__":
    main()
