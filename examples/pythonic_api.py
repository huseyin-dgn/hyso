"""CPU Python API example. Import/preparation never starts training or analysis.

Run ``python examples/pythonic_api.py --execute`` to explicitly fit one tiny
epoch and collect two P11 logical-footprint observations. This is not acceptance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from hyso import Device, GraphModelBuilder, ModelBuilder, TrainingConfig
from hyso.analysis import (AnalysisContext, LiveCPUModel, ModelComparisonSource,
                           ModelContext, compare_models, execute_analysis,
                           plan_analysis, preflight_analysis)
from hyso.advisor import recommend
from hyso.reporting import create_report


def run(*, execute: bool = False, output: Path | None = None) -> dict:
    """Prepare CPU objects; execute only when explicitly requested by the caller."""
    builder = ModelBuilder.classification((1, 2, 2), num_classes=2)
    graph = (GraphModelBuilder({'image': [1, 2, 2]})
             .classification_head('head', ['image'], in_features=1, num_classes=2)
             .outputs(logits='head'))
    generator = torch.Generator(device='cpu').manual_seed(71)
    inputs = torch.randn(4, 1, 2, 2, generator=generator)
    dataset = TensorDataset(inputs, torch.tensor([0, 1, 0, 1]))
    config = (TrainingConfig(epochs=1, batch_size=2, device=Device.cpu())
              .with_sgd(learning_rate=0.01, momentum=0.9)
              .with_step_scheduler(step_size=1, gamma=0.5))
    model = builder.build()
    session = config.prepare_datasets(model, dataset)
    plan = plan_analysis((ModelContext('trained', input_contract=True),
                          ModelContext('fresh', input_contract=True)),
                         features='P11', device=Device.cpu())
    summary = dict(executed=False, graph=graph.preview(), plan=plan.to_dict(),
                   training=session.settings())
    if execute:
        training = session.fit()
        review = preflight_analysis(plan, (LiveCPUModel('trained', model, inputs[:2]),
                                          LiveCPUModel('fresh', builder.build(), inputs[:2])))
        result = execute_analysis(review, authorized_features=('P11',), explicit_authorization=True)
        comparison = compare_models(ModelComparisonSource.from_execution(result, 'trained'),
                                    ModelComparisonSource.from_execution(result, 'fresh'), features='P11')
        recommendations = recommend(comparison)
        context = AnalysisContext(workload_id='ux05-cpu-example', task='classification',
                                  model={}, dataset={'id': 'synthetic-example'})
        report = create_report(comparison, recommendations, context=context, language='tr', level='technical')
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            report.to_json(output / 'report.json')
            report.to_markdown(output / 'report.md')
            report.to_html(output / 'report.html')
        summary.update(executed=True, training=training.to_dict(), analysis=result.to_dict(),
                       comparison=comparison.to_dict(), recommendations=recommendations.to_dict(),
                       report=report.to_dict())
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run(execute=args.execute, output=args.output)
    print(json.dumps({'executed': result['executed']}, ensure_ascii=True))
