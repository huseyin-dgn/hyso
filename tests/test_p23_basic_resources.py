"""Stdlib policy tests: workload budgets replace forensic admission gates."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
import p23_basic_resources as resources

MiB = resources.MiB


class BasicResources(unittest.TestCase):
    def setUp(self):
        self.profile = dict(measurement=dict(rss=dict(summary=dict(current_lifetime_hwm_bytes=dict(value=640*MiB)))),
                            receipt=dict(job=dict(peak_job_commit_bytes=1700*MiB)))
        self.budget = resources.budget(self.profile, 19890)
        self.capture = dict(host=dict(total_physical=16*1024*MiB, available_physical=2800*MiB,
            commit_limit=28*1024*MiB, commit_headroom=8*1024*MiB, low_memory=False),
            gpu=dict(total_bytes=6*1024*MiB, free_bytes=5*1024*MiB), disk_free_bytes=10*1024*MiB)

    def test_less_than_four_GiB_can_be_eligible(self):
        result = resources.evaluate(self.capture, self.budget)
        self.assertTrue(result["eligible"])
        self.assertIsNone(result["fixed_host_free_threshold"])
        self.assertIsNone(result["forensic_freshness_threshold"])
        self.assertFalse(result["forensic_PID_classification_required"])

    def test_each_real_resource_shortage_stops(self):
        for section, key, value in (("host", "available_physical", 1024*MiB),
                ("host", "commit_headroom", 1024*MiB), ("gpu", "free_bytes", 512*MiB),
                ("host", "low_memory", True)):
            with self.subTest(key=key):
                capture = copy.deepcopy(self.capture)
                capture[section][key] = value
                self.assertFalse(resources.evaluate(capture, self.budget)["eligible"])

    def test_post_import_only_budgets_remaining_memory(self):
        capture = copy.deepcopy(self.capture)
        capture["host"]["available_physical"] -= 500*MiB
        capture["host"]["commit_headroom"] -= 1500*MiB
        result = resources.evaluate(capture, self.budget,
            resident=dict(rss_bytes=500*MiB, private_commit_bytes=1500*MiB))
        self.assertTrue(result["eligible"])
        self.assertEqual(result["remaining_host_bytes"], self.budget["expected_host_bytes"]-500*MiB)

    def test_model_need_is_part_of_the_gpu_budget(self):
        large = resources.budget(self.profile, 100_000_000)
        self.assertGreater(large["expected_gpu_bytes"], self.budget["expected_gpu_bytes"])
        self.assertFalse(resources.evaluate(self.capture, large)["eligible"])

    def test_process_limit_is_not_a_preallocation_or_allocator_fraction(self):
        self.assertGreater(self.budget["process_limit_bytes"], self.profile["receipt"]["job"]["peak_job_commit_bytes"])
        self.assertIsNone(self.budget["cuda_memory_fraction"])
        self.assertEqual(self.budget["concurrent_interpreters"], 2)


if __name__ == "__main__":
    unittest.main()
