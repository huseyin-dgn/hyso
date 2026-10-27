"""Executable regression examples, preserving every original Python block."""
BLOCKS = ('from pathlib import Path\n'
 'import tempfile\n'
 'import torch\n'
 'from hyso.benchmark import BenchmarkRunner\n'
 'from hyso.performance import analyze_latency, measure_throughput, ThroughputProtocol\n'
 '\n'
 'torch.set_num_threads(1)\n'
 'model = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 3, padding=1),\n'
 '                            torch.nn.AdaptiveAvgPool2d(1)).eval()\n'
 'inputs = torch.ones(2, 1, 5, 7)\n'
 'benchmark = BenchmarkRunner(model, example_input=inputs, warmup_iterations=2,\n'
 '                            measure_iterations=500, repeats=2).run()\n'
 'latency = analyze_latency(benchmark, quantiles=(.50, .95, .99))\n'
 "assert latency.to_dict()['actual']['samples'] == 1000\n"
 "assert latency.to_dict()['measurements'][-1]['status'] == 'derived'\n"
 '\n'
 'throughput = measure_throughput(model, inputs, protocol=ThroughputProtocol(\n'
 "    device='cpu', precision='fp32', warmup_iterations=2, batch_count=20,\n"
 '    repeats=2, queue_depth=8))\n'
 "assert throughput.to_dict()['actual']['completed_images'] == 80\n"
 "output_dir = Path(tempfile.mkdtemp(prefix='hyso-performance-'))\n"
 "latency.to_json(output_dir / 'latency.json')\n"
 "latency.to_csv(output_dir / 'latency.csv')\n"
 "throughput.to_json(output_dir / 'throughput.json')\n"
 "throughput.to_csv(output_dir / 'throughput.csv')\n",
 'def cuda_example():\n'
 '    if not torch.cuda.is_available():\n'
 "        raise RuntimeError('Native CUDA is required; no CPU fallback')\n"
 "    gpu_model = torch.nn.Conv2d(1, 2, 3, padding=1).to('cuda:0').eval()\n"
 "    gpu_input = torch.ones(2, 1, 5, 7, device='cuda:0')\n"
 '    stream = torch.cuda.Stream(device=0)\n'
 '    # Producer setup is complete before another stream consumes these tensors.\n'
 '    torch.cuda.synchronize(0)\n'
 '    with torch.cuda.stream(stream):\n'
 '        return measure_throughput(gpu_model, gpu_input, protocol=ThroughputProtocol(\n'
 "            device='cuda:0', precision='amp_fp16', warmup_iterations=2,\n"
 '            batch_count=33, queue_depth=8, repeats=2))\n',
 'from hyso.experiments import ExperimentManager\n'
 '\n'
 "root = output_dir / 'experiment-store'\n"
 'manager = ExperimentManager(root)\n'
 "experiment = manager.create_experiment('P01-P02 resident inference')\n"
 'manager.close()\n'
 'ExperimentManager.migrate_store(root)\n'
 'manager = ExperimentManager(root)\n'
 "run = manager.create_run(experiment['id'], 'benchmark',\n"
 "    config={'protocol': throughput.to_dict()['protocol']},\n"
 "    provenance={'throughput_result_id': throughput.to_dict()['result_id'],\n"
 "                'source_benchmark_id': benchmark.to_dict()['identity']['run_id'],\n"
 "                'model_input_runtime': throughput.to_dict()['provenance']})\n"
 'run.start()\n'
 "artifact = run.add_artifact(output_dir / 'throughput.json', kind='performance_result',\n"
 "                            ownership='copy', role='output')\n"
 "run.add_artifact(output_dir / 'latency.json', kind='performance_result', ownership='copy')\n"
 "run.log_metric(name='perf.throughput.images', "
 "value=throughput.to_dict()['measurements']['value'],\n"
 "               unit='images/s', source='performance.throughput_v1', step=0, "
 "step_kind='observation')\n"
 "p99 = latency.to_dict()['measurements'][-1]\n"
 "if p99['value'] is not None:\n"
 "    run.log_metric(name='perf.latency.p99', value=p99['value'], unit='ms',\n"
 "                   source='performance.latency_v1', step=0, step_kind='observation')\n"
 "assert manager.verify_artifact(artifact['id'])['status'] == 'verified'\n"
 'run.finish()\n'
 "run_id, artifact_id = run.id, artifact['id']\n"
 'manager.close()\n'
 'manager = ExperimentManager(root)\n'
 "assert manager.get_run(run_id).to_dict()['status'] == 'completed'\n"
 "assert manager.verify_artifact(artifact_id)['status'] == 'verified'\n"
 'manager.close()\n')
