"""Stage 13.5.14 probes selecting explicit strict context policy.

The original assertion bodies are retained in the closure evidence ledger.
"""

import pytest

from test_evaluation_comparison_acceptance import (
    bounded_cpu_work, classifier, graph_capture, produce, source,
)
from hyso.analysis import compare_models as _compare_models
from hyso.analysis.graph_evidence import compare_graph_evaluations as _compare_graph_evaluations


def compare_models(*args, **kwargs):
    return _compare_models(*args, context_policy="strict", **kwargs)


def compare_graph_evaluations(*args, **kwargs):
    return _compare_graph_evaluations(*args, context_policy="strict", **kwargs)


@pytest.mark.parametrize("field", ["dataset_id", "preprocessing_id"])
def test_requested_generic_declared_mismatch_rejection(field):
    left = source("A", produce(classifier(1.5)), **{field: "left"})
    right = source("B", produce(), **{field: "right"})
    result = compare_models(left, right, features="P26").to_dict()
    actual = [{"status": r["status"], "difference": r["difference"]} for r in result["metrics"]]
    assert all(r["status"] == "INCOMPATIBLE" and r["difference"] is None
               for r in result["metrics"]), actual


@pytest.mark.parametrize("field", ["dataset_id", "preprocessing_id"])
def test_requested_graph_incompatibility_has_no_nested_numeric_difference(tmp_path, field):
    left_model, left = graph_capture(tmp_path)
    folder = tmp_path / "right"
    folder.mkdir()
    context = dict(dataset_id="small-qa", preprocessing_id="identity", split="test")
    context[field] = "other"
    right_model, right = graph_capture(folder, context=context)
    result = compare_graph_evaluations(left, right).to_dict()
    assert result["status"] == "INCOMPATIBLE"
    assert all(r["difference"] is None for r in result["metrics"])
    actual = [{"status": r["status"], "difference": r["difference"]}
              for r in result["legacy"]["metrics"]]
    assert all(r["difference"] is None for r in result["legacy"]["metrics"]), actual
