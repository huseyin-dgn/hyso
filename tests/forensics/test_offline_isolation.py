"""The no-Torch harness must propagate failure and leave the parent untouched."""
import sys
import unittest
import offline_isolation_cases as cases


class OfflineIsolationTests(unittest.TestCase):
    def test_clean_child_leaves_parent_modules_untouched(self):
        before=dict(sys.modules)
        cases.CleanCase('check').check()
        self.assertEqual(before,sys.modules)

    def test_contaminated_child_is_a_failure_not_a_pass(self):
        before=dict(sys.modules)
        with self.assertRaises(AssertionError):
            cases.ContaminatedCase('check').check()
        self.assertEqual(before,sys.modules)
