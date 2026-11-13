"""Create offline CLI demo files in a fresh directory. Never capture or train.

Usage: python examples/cli/prepare_demo.py --output /existing/parent/fresh-demo
The training timing fixture is synthetic retained test data, not a user result.
"""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(output):
    from hyso import ModelBuilder, TrainingConfig
    from hyso.analysis import (ModelContext, NativeBinding, RetainedSource, plan_analysis,
                               preflight_analysis, execute_analysis, AnalysisContext)
    fixture = Path(__file__).with_name('training-timing.json')
    payload = json.loads(fixture.read_text(encoding='utf-8'))
    output.mkdir(exist_ok=False)
    ModelBuilder((1, 4, 4), 2).add('conv2d', in_channels=1, out_channels=2).to_json(output / 'architecture.json')
    TrainingConfig(epochs=1, batch_size=2).to_json(output / 'training.json')
    (output / 'p19.json').write_bytes(fixture.read_bytes())
    sha = hashlib.sha256(fixture.read_bytes()).hexdigest()
    for label in ('A', 'B'):
        plan = plan_analysis(ModelContext(label), features=('P19', 'P22'), execution_mode='native')
        review = preflight_analysis(plan, [NativeBinding('retained-demo', (label,), ('P19', 'P22'), RetainedSource(payload))])
        result = execute_analysis(review, authorized_features=('P19', 'P22'), explicit_authorization=True)
        result.to_json(output / (label + '-execution.json'))
        result.to_unified_report(context=AnalysisContext(workload_id='synthetic-offline-demo',
            task='classification', model={'label': label}, dataset={'origin': 'synthetic retained fixture; no training'})).to_json(output / (label + '-analysis.json'))
    print(json.dumps(dict(directory=str(output.resolve()), p19_sha256=sha,
                          scope='synthetic retained fixture; no training, capture or hardware acceptance'), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='Fresh directory under an existing parent.')
    prepare(parser.parse_args().output)
