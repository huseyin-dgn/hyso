"""Offline P18 capture/locale contracts: no native sessions, GPU or NCU."""
import ast
from pathlib import Path
import unittest

from full_acceptance.handlers import _p18_capture_contract
from full_acceptance import core


class P18CaptureContractTests(unittest.TestCase):
    def test_target_is_single_observed_kernel_interval(self):
        self.assertEqual(_p18_capture_contract('target', {'number_format': 'decimal_comma'}), (1, 'decimal_comma'))

    def test_compute_and_memory_keep_three_independent_samples(self):
        self.assertEqual(_p18_capture_contract('compute', {'number_format': 'decimal_comma'}), (3, 'decimal_comma'))
        self.assertEqual(_p18_capture_contract('memory', {'number_format': 'decimal_point'}), (3, 'decimal_point'))

    def test_ambiguous_or_invalid_locale_fails_closed(self):
        for plan in ({}, {'number_format': None}, {'number_format': 'auto'}, {'number_format': 'decimal_both'}):
            with self.assertRaisesRegex(ValueError, 'explicit ncu_plan.number_format'):
                _p18_capture_contract('target', plan)
        with self.assertRaisesRegex(ValueError, 'invalid P18 capture role'):
            _p18_capture_contract('unknown', {'number_format': 'decimal_comma'})

    def test_source_target_receipt_is_exactly_one(self):
        path = core.ROOT / 'tools/full_acceptance/counter_worker.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'target_case')
        target_loops = [node for node in ast.walk(fn) if isinstance(node, ast.For)
                        and isinstance(node.target, ast.Name) and node.target.id == 'i']
        self.assertEqual(len(target_loops), 1)
        self.assertEqual(ast.literal_eval(target_loops[0].iter.args[0]), 1)
        keywords = [node for node in ast.walk(fn) if isinstance(node, ast.keyword) and node.arg == 'targeted_launches']
        self.assertEqual(len(keywords), 1)
        self.assertEqual(ast.literal_eval(keywords[0].value), 1)
        self.assertIn('targeted_launches=1', ast.get_source_segment(path.read_text(encoding='utf-8'), fn))
