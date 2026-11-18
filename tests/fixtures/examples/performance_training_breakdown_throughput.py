"""Executable regression examples, preserving every original Python block."""
BLOCKS = ('import torch\n'
 'from torch import nn\n'
 'from hyso.training import Trainer\n'
 'from hyso.performance import (TrainingStageProtocol, TrainingTimingProtocol,\n'
 '    measure_training_detailed, analyze_training_throughput)\n'
 '\n'
 'model = nn.Linear(3, 2)\n'
 'trainer = Trainer(model, optimizer=torch.optim.AdamW(model.parameters(), lr=.01),\n'
 '    loss_fn=nn.CrossEntropyLoss(), accum_steps=2, grad_clip=.5, ema_decay=.8)\n'
 'train = list(zip(torch.ones(5, 3).split(2), torch.tensor([0, 1, 0, 1, 0]).split(2)))\n'
 'result = measure_training_detailed(trainer, train, epochs=2,\n'
 '    protocol=TrainingTimingProtocol(event_buffer=32, max_records=1000),\n'
 '    stage_protocol=TrainingStageProtocol(max_records=1000))\n'
 'timing = result.timing.to_dict()\n'
 'stages = result.stages.to_dict()\n'
 'throughput = result.throughput.to_dict()\n'
 "assert [r['samples'] for r in timing['observations'] if r['kind'] == 'window'] == [4, 1, 4, 1]\n"
 "assert throughput['segment']['completed']['samples'] == 10\n"
 "assert throughput['segment']['completed']['microbatches'] == 6\n"
 "assert throughput['segment']['completed']['attempts'] == 4\n"
 "assert throughput['segment']['completed']['applied'] == 4\n"
 '# Saf analiz; ikinci fit, timer veya model çağrısı yoktur.\n'
 'assert analyze_training_throughput(result.timing).to_dict() == throughput\n',
 'import hashlib\n'
 'from pathlib import Path\n'
 'import tempfile\n'
 'from hyso.experiments import ExperimentManager\n'
 '\n'
 '# Fit bitti. Tracking hatası bu ölçümü yeniden çalıştırmaz.\n'
 "with tempfile.TemporaryDirectory(prefix='hyso-b9-') as directory:\n"
 '    root = Path(directory)\n'
 "    manager = ExperimentManager(root / 'store')\n"
 '    manager.close()\n'
 "    ExperimentManager.migrate_store(root / 'store')  # mevcut artifact store akışı\n"
 "    manager = ExperimentManager(root / 'store')\n"
 "    run = manager.create_run(manager.create_experiment('training-B9')['id'], 'training', {})\n"
 '    run.start()\n'
 "    for name, component in [('P20', result.stages), ('P22', result.throughput)]:\n"
 "        path = root / (name + '.json')\n"
 '        component.to_json(path)\n'
 "        component.to_csv(root / (name + '.csv'))\n"
 "        artifact = run.add_artifact(path, kind='report', ownership='copy',\n"
 "            metadata={'source_segment_id': timing['segment_id']})\n"
 "        assert artifact['expected_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()\n"
 "        assert manager.verify_artifact(artifact['id'])['status'] == 'verified'\n"
 "        csv_path = root / (name + '.csv')\n"
 "        csv_artifact = run.add_artifact(csv_path, kind='report', ownership='copy',\n"
 "            metadata={'source_segment_id': timing['segment_id']})\n"
 "        assert csv_artifact['expected_sha256'] == "
 'hashlib.sha256(csv_path.read_bytes()).hexdigest()\n'
 "        assert manager.verify_artifact(csv_artifact['id'])['status'] == 'verified'\n"
 "    for epoch in throughput['epochs']:\n"
 "        metric = epoch['rates']['samples_per_second']\n"
 "        if metric['value'] is not None:\n"
 "            run.log_metric(name='training.samples', value=metric['value'], unit=metric['unit'],\n"
 "                           source='P22', epoch=epoch['epoch_id'])\n"
 "    for row in stages['summary']:\n"
 "        if row['host_ms'] is not None:\n"
 "            run.log_metric(name='training.stage.' + row['stage'], value=row['host_ms'],\n"
 "                           unit='ms', source='P20', step=0, step_kind='observation')\n"
 '    run.finish()\n'
 '    identity = run.id\n'
 '    manager.close()\n'
 "    manager = ExperimentManager(root / 'store')\n"
 "    assert len(manager.get_run(identity).read_metrics()) >= len(throughput['epochs'])\n"
 '    manager.close()\n')
