"""Executable regression examples, preserving every original Python block."""
BLOCKS = ('import json\n'
 'from pathlib import Path\n'
 'import tempfile\n'
 'from hyso.performance import UtilizationProtocol, UtilizationResult, measure_utilization\n'
 '\n'
 'utilization = measure_utilization(protocol=UtilizationProtocol(\n'
 '    duration_seconds=.12, interval_seconds=.02, gpu=False))\n'
 'utilization_data = utilization.to_dict()\n'
 "assert utilization_data['actual']['cpu_samples'] >= 2\n"
 "assert utilization_data['measurements']['gpu']['tensor_core_utilization']['value'] is None\n"
 "assert utilization_data['actual']['gpu_queries'] == 0\n"
 '\n'
 "export_root = Path(tempfile.mkdtemp(prefix='hyso-transfer-utilization-guide-'))\n"
 "utilization.to_json(export_root / 'utilization.json')\n"
 "utilization.to_csv(export_root / 'utilization.csv')\n"
 'restored_utilization = UtilizationResult(json.loads(\n'
 "    (export_root / 'utilization.json').read_text(encoding='utf-8')))\n"
 'assert restored_utilization.to_dict() == utilization_data\n',
 'from hyso.experiments import ExperimentManager\n'
 "store = export_root / 'runs'\n"
 'manager = ExperimentManager(store)\n'
 "experiment = manager.create_experiment('transfer-utilization')\n"
 'manager.close()\n'
 'ExperimentManager.migrate_store(store)\n'
 'manager = ExperimentManager(store)\n'
 "run = manager.create_run(experiment['id'], 'benchmark', config={'feature': 'P16'})\n"
 'run.start()\n'
 "artifact = run.add_artifact(export_root / 'utilization.json',\n"
 "    kind='performance_result', ownership='copy')\n"
 "run.log_metric(name='cpu.samples', value=utilization_data['actual']['cpu_samples'],\n"
 "    unit='count', step=0, step_kind='observation', source='performance.utilization')\n"
 'run_id = run.id\n'
 'run.finish()\n'
 'manager.close()\n'
 'manager = ExperimentManager(store)\n'
 "assert manager.verify_artifact(artifact['id'])['status'] == 'verified'\n"
 'assert manager.get_run(run_id).read_metrics()\n'
 'manager.close()\n')
