"""Current explicit P12 v3 example; the archived v2 blocks remain unchanged."""
BLOCKS = ('from pathlib import Path\n'
 'import tempfile\n'
 'from hyso.performance import (ColdStartRecipe, ProfilingProtocol,\n'
 '    profile_operators, profile_layers, analyze_flops)\n'
 '\n'
 "output_dir = Path(tempfile.mkdtemp(prefix='hyso-profiling-guide-'))\n"
 "recipe = ColdStartRecipe('hyso.models.modular:ModularCNN',\n"
 "    model_kwargs={'input_channels': 1, 'num_classes': 2, 'widths': [3],\n"
 "                  'repeats': [1], 'strides': [1], 'attention': 'eca'},\n"
 "    input_shape=(2, 1, 5, 7), device='cpu', precision='fp32')\n"
 'protocol = ProfilingProtocol(warmup_iterations=1, measure_iterations=2,\n'
 '    timeout_seconds=45, cpu_memory_limit_bytes=3 * 1024**3,\n'
 '    input_limit_bytes=32 * 1024**2, max_events=20000)\n'
 'operators = profile_operators(recipe, protocol=protocol,\n'
 "    trace_path=output_dir / 'model-trace.json')\n"
 'layers = profile_layers(operators)\n'
 'counts = analyze_flops(operators)\n'
 'operator_data = operators.to_dict()\n'
 'layer_data = layers.to_dict()\n'
 'count_data = counts.to_dict()\n'
 "assert operator_data['actual']['measured_forwards'] == 2\n"
 "assert count_data['protocol']['operation_model_version'] == 3\n"
 "assert count_data['measurements']['status'] == 'complete'\n"
 "assert count_data['measurements']['total_flops_per_batch'] is not None\n"
 "assert count_data['measurements']['coverage']['unknown_calls'] == 0\n"
 "assert count_data['measurements']['transcendental_ops_per_batch']['exp'] > 0\n"
 "top_layers = [row for row in layer_data['measurements']['modules']\n"
 "              if row['cpu_exclusive_us'] is not None]\n"
 "top_layers.sort(key=lambda row: row['cpu_exclusive_us'], reverse=True)\n"
 "print('Instrumented CPU layers:', [(row['path'] or '<root>', row['cpu_exclusive_us'])\n"
 '                                   for row in top_layers[:3]])\n'
 "print('Instrumented CPU operators:', [(row['name'], row['self_cpu_us'])\n"
 '                                     for row in '
 "operator_data['measurements']['operators'][:3]])\n",
 "tiny_recipe = ColdStartRecipe('torch.nn:Conv2d',\n"
 "    model_kwargs={'in_channels': 2, 'out_channels': 4, 'kernel_size': 3,\n"
 "                  'padding': 1, 'groups': 2, 'bias': True},\n"
 "    input_shape=(2, 2, 5, 7), precision='fp64')\n"
 'tiny_counts = analyze_flops(tiny_recipe, protocol=protocol)\n'
 "tiny = tiny_counts.to_dict()['measurements']\n"
 "assert tiny['total_macs_per_batch'] == 2520\n"
 "assert tiny['total_flops_per_batch'] == 5320\n"
 "assert tiny['total_flops_per_sample'] == 2660\n"
 "assert tiny['coverage']['unknown_calls'] == 0\n"
 "assert tiny_counts.to_dict()['provenance']['input']['dtype'] == 'torch.float64'\n",
 "for name, result in [('operators', operators), ('layers', layers), ('counts', counts)]:\n"
 "    result.to_json(output_dir / (name + '.json'))\n"
 "    result.to_csv(output_dir / (name + '.csv'))\n"
 'from hyso.experiments import ExperimentManager\n'
 "root = output_dir / 'runs'\n"
 'manager = ExperimentManager(root)\n'
 "experiment = manager.create_experiment('profiling')\n"
 'manager.close()\n'
 'ExperimentManager.migrate_store(root)\n'
 'manager = ExperimentManager(root)\n'
 "run = manager.create_run(experiment['id'], 'benchmark', config={'features': ['P12', 'P13', "
 "'P14']},\n"
 "    provenance={'operator_result_id': operator_data['result_id']})\n"
 'run.start()\n'
 'artifact_ids = []\n'
 "for filename in ['operators.json', 'layers.json', 'counts.json', 'model-trace.json']:\n"
 "    record = run.add_artifact(output_dir / filename, kind='performance_result', "
 "ownership='copy')\n"
 "    artifact_ids.append(record['id'])\n"
 'metric_count = 0\n'
 "for index, row in enumerate(operator_data['measurements']['operators']):\n"
 "    if row['self_cpu_us'] is not None:\n"
 "        run.log_metric(name='profile.cpu_self', value=row['self_cpu_us'], unit='us',\n"
 "            step=index, step_kind='observation', source='performance.operator_profile_v1')\n"
 '        metric_count += 1\n'
 "run.log_metric(name='profile.collector_wall',\n"
 "    value=operator_data['measurements']['overhead']['collector_lifecycle_wall_ms'], unit='ms',\n"
 "    step=0, step_kind='observation', source='performance.operator_profile_v1')\n"
 'metric_count += 1\n'
 'run.finish()\n'
 'run_id = run.id\n'
 'manager.close()\n'
 'manager = ExperimentManager(root)\n'
 "assert manager.get_run(run_id).to_dict()['status'] == 'completed'\n"
 'assert len(manager.get_run(run_id).read_metrics()) == metric_count\n'
 "assert all(manager.verify_artifact(identity)['status'] == 'verified' for identity in "
 'artifact_ids)\n'
 'manager.close()\n')
