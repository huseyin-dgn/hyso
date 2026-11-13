"""Offline stdlib policy tests; unittest discovery avoids Torch/conftest entirely."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tools/forensics/p23_preflight.py"
spec = importlib.util.spec_from_file_location("p23_preflight", SOURCE)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)
NOW = 100 * 10**9
RUN = "offline-test-only"
SHA = "a" * 64


def capture():
    inventory = dict(process_query_succeeded=True, compute_query_succeeded=True,
                     processes=[dict(pid=42, name="WhatsApp.Root.exe", path=None)],
                     compute_pids=[42], raw_query=["42, [Insufficient Permissions], [N/A]"])
    return dict(policy_id=policy.POLICY_ID, run_id=RUN, manifest_sha256=SHA,
                collection_start_time_ns=NOW-4*10**9, checked_time_ns=NOW-10**9,
                host_free_bytes=4*1024**3, disk_free_bytes=12*1024**3,
                gpu_samples=[dict(time_ns=NOW-(4-i)*10**9, uuid=policy.GPU_UUID,
                                  utilization_percent=x, free_mib=2048)
                             for i, x in enumerate([35, 80, 100])],
                inventory=inventory,
                workload_review=dict(status="complete", reviewer="offline-test",
                    inventory_sha256=policy.canonical_hash(inventory), conflicting_workloads=[],
                    gpu_processes=[dict(pid=42, category="desktop", reason="fixture desktop identity reviewed")]))


def evaluate(value):
    return policy.evaluate(value, run_id=RUN, manifest_sha256=SHA, now_ns=NOW)


class PreflightPolicyTests(unittest.TestCase):
    def test_high_utilization_and_exact_memory_thresholds_are_eligible(self):
        result = evaluate(capture())
        self.assertTrue(result["eligible"])
        self.assertEqual(result["utilization_percent_samples"], [35, 80, 100])
        self.assertEqual(result["performance_comparison"]["status"], "unsupported")
        self.assertIsNone(result["performance_comparison"]["value"])
        self.assertFalse(result["cuda_workload_started"])

    def test_missing_utilization_is_explicit_telemetry_null(self):
        value = capture()
        value["gpu_samples"][0]["utilization_percent"] = None
        self.assertTrue(evaluate(value)["eligible"])
        self.assertIsNone(evaluate(value)["utilization_percent_samples"][0])

    def test_no_input_mutation(self):
        value = capture()
        original = copy.deepcopy(value)
        evaluate(value)
        self.assertEqual(value, original)

    def test_each_memory_budget_blocks_independently(self):
        for field in ("host_free_bytes", "disk_free_bytes"):
            with self.subTest(field=field):
                value = capture()
                value[field] -= 1
                self.assertFalse(evaluate(value)["eligible"])
        for index in range(3):
            with self.subTest(gpu_sample=index):
                value = capture()
                value["gpu_samples"][index]["free_mib"] -= 1
                self.assertIn("GPU_free_below_2048MiB", evaluate(value)["blockers"])

    def test_freshness_uses_capture_start_and_rejects_future(self):
        for stamp in (NOW-30*10**9-1, NOW+1):
            with self.subTest(stamp=stamp):
                value = capture()
                value["collection_start_time_ns"] = stamp
                self.assertFalse(evaluate(value)["eligible"])
        value = capture()
        value["collection_start_time_ns"] = NOW-30*10**9
        self.assertTrue(evaluate(value)["eligible"])

    def test_every_sample_must_match_gpu_and_be_ordered(self):
        for field, replacement in (("uuid", "GPU-other"), ("time_ns", NOW+1)):
            with self.subTest(field=field):
                value = capture()
                value["gpu_samples"][1][field] = replacement
                self.assertFalse(evaluate(value)["eligible"])
        value = capture()
        value["gpu_samples"][1]["time_ns"] = value["gpu_samples"][0]["time_ns"]
        self.assertFalse(evaluate(value)["eligible"])

    def test_identity_and_policy_are_bound(self):
        for key in ("policy_id", "run_id", "manifest_sha256"):
            with self.subTest(key=key):
                value = capture()
                value[key] = "other"
                self.assertFalse(evaluate(value)["eligible"])

    def test_unknown_or_compute_blocks_even_when_idle(self):
        for category in ("unknown", "compute"):
            with self.subTest(category=category):
                value = capture()
                for sample in value["gpu_samples"]:
                    sample["utilization_percent"] = 0
                value["workload_review"]["gpu_processes"][0]["category"] = category
                self.assertFalse(evaluate(value)["eligible"])

    def test_conflict_and_incomplete_review_block(self):
        for key, replacement in (("conflicting_workloads", ["independent training"]),
                                 ("status", "pending"), ("gpu_processes", []),
                                 ("inventory_sha256", "b"*64)):
            with self.subTest(key=key):
                value = capture()
                value["workload_review"][key] = replacement
                self.assertFalse(evaluate(value)["eligible"])

    def test_inventory_failures_and_hidden_pids_block(self):
        for key, replacement in (("process_query_succeeded", False),
                                 ("compute_query_succeeded", False),
                                 ("raw_query", ["43, unknown, N/A"]),
                                 ("processes", [])):
            with self.subTest(key=key):
                value = capture()
                value["inventory"][key] = replacement
                value["workload_review"]["inventory_sha256"] = policy.canonical_hash(value["inventory"])
                self.assertFalse(evaluate(value)["eligible"])

    def test_malformed_or_missing_data_never_becomes_eligible(self):
        for key in capture():
            with self.subTest(missing=key):
                value = capture()
                del value[key]
                with self.assertRaises((KeyError, ValueError, TypeError)):
                    evaluate(value)
        for invalid in (True, -1, float("nan"), 101, "35"):
            with self.subTest(busy=invalid):
                value = capture()
                value["gpu_samples"][0]["utilization_percent"] = invalid
                with self.assertRaises(ValueError):
                    evaluate(value)
        for invalid in (True, -1, None):
            with self.subTest(memory=invalid):
                value = capture()
                value["host_free_bytes"] = invalid
                with self.assertRaises(ValueError):
                    evaluate(value)

    def test_only_three_samples_no_duplicate_review_or_empty_reason(self):
        value = capture()
        value["gpu_samples"].append(value["gpu_samples"][-1])
        with self.assertRaises(ValueError):
            evaluate(value)
        value = capture()
        value["workload_review"]["gpu_processes"] *= 2
        self.assertFalse(evaluate(value)["eligible"])
        value = capture()
        value["workload_review"]["gpu_processes"][0]["reason"] = ""
        with self.assertRaises(ValueError):
            evaluate(value)

    def test_cli_rejects_duplicate_json_and_oversize_without_launching(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            for payload in ('{"run_id":"one","run_id":"two"}', "x"*(1024**2+1)):
                path.write_text(payload, encoding="utf-8")
                result = subprocess.run([sys.executable, "-I", "-S", "-B", str(SOURCE),
                                         str(path), "--run-id", RUN, "--manifest-sha256", SHA],
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2, result.stderr)
                output = json.loads(result.stdout)
                self.assertFalse(output["eligible"])
                self.assertFalse(output["cuda_workload_started"])

    def test_module_has_no_workload_or_sensor_imports_and_retains_budgets(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        imports = [node.names[0].name if isinstance(node, ast.Import) else node.module
                   for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertTrue(set(imports) <= {"argparse", "hashlib", "json", "math", "pathlib", "time"})
        self.assertEqual(policy.LIMITS["child_timeout_s"], 60)
        self.assertEqual(policy.LIMITS["outer_timeout_s"], 90)
        self.assertEqual(policy.LIMITS["cuda_allocator_fraction"], .15)
        self.assertTrue(policy.LIMITS["kill_on_close"])


if __name__ == "__main__":
    unittest.main()
