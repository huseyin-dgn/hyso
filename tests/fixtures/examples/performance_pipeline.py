"""Executable regression examples, preserving every original Python block."""
BLOCKS = ('from pathlib import Path\n'
 'import tempfile\n'
 'import torch\n'
 'from hyso.performance import (PipelineSample, PipelineStage, PipelineProtocol,\n'
 '    measure_pipeline, analyze_stages, ColdStartRecipe, ColdStartProtocol,\n'
 '    measure_cold_start, measure_footprint)\n'
 '\n'
 'torch.set_num_threads(1)\n'
 "output_dir = Path(tempfile.mkdtemp(prefix='hyso-pipeline-'))\n"
 'counter = iter(range(100))\n'
 'def read():\n'
 '    index = next(counter)\n'
 '    return PipelineSample(str(index), torch.full((1, 1, 5, 7), float(index)))\n'
 'def preprocess(frame):\n'
 '    resized = torch.nn.functional.interpolate(frame.payload, size=(4, 6)) / 255.\n'
 '    return PipelineSample(frame.sample_id, resized)\n'
 'def postprocess(frame):\n'
 '    return PipelineSample(frame.sample_id, frame.payload.mean((2, 3), keepdim=True))\n'
 '\n'
 'model = torch.nn.Conv2d(1, 2, 3, padding=1).eval()\n'
 'pipeline = measure_pipeline(model, read,\n'
 "    preprocess=PipelineStage('preprocess', preprocess),\n"
 "    postprocess=PipelineStage('postprocess', postprocess),\n"
 '    protocol=PipelineProtocol(samples=4, warmup_samples=1, queue_depth=2))\n'
 "stages = analyze_stages(pipeline)  # raw stage evidence'dan; model tekrar çalışmaz\n"
 "assert pipeline.to_dict()['actual']['completed'] == 4\n"
 "assert stages.to_dict()['measurements']['inference']['count'] == 4\n"
 "pipeline.to_json(output_dir / 'pipeline.json')\n"
 "pipeline.to_csv(output_dir / 'pipeline.csv')\n"
 "stages.to_json(output_dir / 'stages.json')\n"
 "stages.to_csv(output_dir / 'stages.csv')\n",
 'def cuda_pipeline_example():\n'
 '    if not torch.cuda.is_available():\n'
 "        raise RuntimeError('Native CUDA required; no CPU fallback')\n"
 "    gpu_model = torch.nn.Conv2d(1, 2, 3, padding=1).to('cuda:0').eval()\n"
 '    ids = iter(range(20))\n'
 '    def source():\n'
 '        return PipelineSample(str(next(ids)), torch.ones(1, 1, 5, 7))\n'
 '    def transfer(frame):\n'
 "        return PipelineSample(frame.sample_id, frame.payload.to('cuda:0', non_blocking=True))\n"
 '    def post(frame):\n'
 '        return PipelineSample(frame.sample_id, frame.payload.mean((2, 3), keepdim=True))\n'
 '    stream = torch.cuda.Stream(device=0)\n'
 '    torch.cuda.synchronize(0)\n'
 '    with torch.cuda.stream(stream):\n'
 '        return measure_pipeline(gpu_model, source,\n'
 "            transfer=PipelineStage('transfer', transfer, device='cuda'),\n"
 "            postprocess=PipelineStage('postprocess', post, device='cuda'),\n"
 "            protocol=PipelineProtocol(device='cuda:0', precision='amp_fp16', samples=3))\n",
 "model_path = output_dir / 'state.pt'\n"
 'artifact = measure_footprint(model, serialize_to=model_path, verify_roundtrip=True)\n'
 "info = artifact.to_dict()['raw_evidence']['artifact']\n"
 "recipe = ColdStartRecipe('torch.nn:Conv2d',\n"
 "    model_kwargs={'in_channels': 1, 'out_channels': 2, 'kernel_size': 3, 'padding': 1},\n"
 "    input_shape=(1, 1, 5, 7), artifact_path=model_path, artifact_sha256=info['sha256'])\n"
 'cold = measure_cold_start(recipe,\n'
 '    protocol=ColdStartProtocol(trials=1, warmup_iterations=1, warm_samples=2))\n'
 "assert cold.to_dict()['actual']['measured'] == 1\n"
 "cold.to_json(output_dir / 'cold.json')\n"
 "cold.to_csv(output_dir / 'cold.csv')\n",
 'from hyso.experiments import ExperimentManager\n'
 "root = output_dir / 'runs'\n"
 'manager = ExperimentManager(root)\n'
 "experiment = manager.create_experiment('P03-P04-P05')\n"
 'manager.close()\n'
 'ExperimentManager.migrate_store(root)  # explicit offline artifact capability\n'
 'manager = ExperimentManager(root)\n'
 "run = manager.create_run(experiment['id'], 'benchmark',\n"
 "    config={'pipeline_protocol': pipeline.to_dict()['protocol'], 'cold_protocol': "
 "cold.to_dict()['protocol']},\n"
 "    provenance={'pipeline_result_id': pipeline.to_dict()['result_id'],\n"
 "                'cold_result_id': cold.to_dict()['result_id'], 'model_artifact_sha256': "
 "info['sha256']})\n"
 'run.start()\n'
 'artifact_ids = []\n'
 "for name in ('pipeline', 'stages', 'cold'):\n"
 "    record = run.add_artifact(output_dir / (name + '.json'), kind='performance_result', "
 "ownership='copy')\n"
 "    artifact_ids.append(record['id'])\n"
 "    assert manager.verify_artifact(record['id'])['status'] == 'verified'\n"
 "for name, value in [('perf.pipeline.p50', "
 "pipeline.to_dict()['measurements']['p50_ms']['value']),\n"
 "                    ('perf.stage.inference.host_mean', "
 "stages.to_dict()['measurements']['inference']['host_mean_ms']),\n"
 "                    ('perf.cold.first_mean', "
 "cold.to_dict()['measurements']['first_inference_mean_ms']['value'])]:\n"
 '    if value is not None:\n'
 "        run.log_metric(name=name, value=value, unit='ms', source='performance.pipeline_v1', "
 "step=0, step_kind='observation')\n"
 'run.finish()\n'
 'run_id = run.id\n'
 'manager.close()\n'
 'manager = ExperimentManager(root)\n'
 "assert manager.get_run(run_id).to_dict()['status'] == 'completed'\n"
 "assert all(manager.verify_artifact(identity)['status'] == 'verified' for identity in "
 'artifact_ids)\n'
 'manager.close()\n')
