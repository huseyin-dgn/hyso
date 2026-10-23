"""Synthetic positive/negative harness controls, not discovered test modules."""
import sys
import unittest
from offline_test_isolation import offline_test


class CleanCase(unittest.TestCase):
    @offline_test
    def check(self):
        self.assertNotIn('torch',sys.modules)


class ContaminatedCase(unittest.TestCase):
    @offline_test
    def check(self):
        sys.modules['torch']=object()  # Intentional synthetic import-contamination control.
        self.assertNotIn('torch',sys.modules)
