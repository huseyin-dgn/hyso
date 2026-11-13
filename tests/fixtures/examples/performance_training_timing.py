"""Executable regression examples, preserving every original Python block."""
BLOCKS = ('import torch\n'
 'from torch import nn\n'
 'from torch.utils.data import DataLoader, TensorDataset\n'
 'from hyso.training import Trainer\n'
 'from hyso.performance import TrainingTimingProtocol, measure_training\n'
 '\n'
 'model = nn.Linear(3, 2)\n'
 'optimizer = torch.optim.SGD(model.parameters(), lr=0.01)\n'
 'trainer = Trainer(model, optimizer=optimizer, loss_fn=nn.CrossEntropyLoss(), accum_steps=2)\n'
 'train = DataLoader(TensorDataset(torch.ones(5, 3), torch.tensor([0, 1, 0, 1, 0])), '
 'batch_size=2)\n'
 'result = measure_training(trainer, train, epochs=2,\n'
 '    protocol=TrainingTimingProtocol(max_records=1000, event_buffer=32, warmup_epochs=0))\n'
 'data = result.to_dict()\n'
 "assert [e['samples'] for e in data['epochs']] == [5, 5]\n"
 "assert [e['microbatches'] for e in data['epochs']] == [3, 3]\n"
 "assert [r['samples'] for r in data['observations'] if r['kind'] == 'window'] == [4, 1, 4, 1]\n",
 'import hashlib\n'
 'import tempfile\n'
 'from pathlib import Path\n'
 'from hyso.experiments import ExperimentManager\n'
 '\n'
 "# measure_training tamamlandı; bundan sonra tracking hatası fit'i tekrarlamaz.\n"
 "with tempfile.TemporaryDirectory(prefix='hyso-p19-') as directory:\n"
 '    root = Path(directory)\n'
 "    result.to_json(root / 'timing.json')\n"
 "    result.to_csv(root / 'timing.csv')\n"
 "    manager = ExperimentManager(root / 'store')\n"
 '    manager.close()\n'
 "    ExperimentManager.migrate_store(root / 'store')  # mevcut artifact store akışı\n"
 "    manager = ExperimentManager(root / 'store')\n"
 "    run = manager.create_run(manager.create_experiment('P19')['id'], 'training', {})\n"
 '    run.start()\n'
 "    artifact = run.add_artifact(root / 'timing.json', kind='report', ownership='copy')\n"
 "    assert artifact['expected_sha256'] == hashlib.sha256((root / "
 "'timing.json').read_bytes()).hexdigest()\n"
 "    assert manager.verify_artifact(artifact['id'])['status'] == 'verified'\n"
 "    for epoch in data['epochs']:\n"
 "        run.log_metric(name='training.epoch.wall', value=epoch['training_s'],\n"
 "                       unit='s', source='P19', epoch=epoch['epoch_id'])\n"
 '    run.finish()\n'
 '    identity = run.id\n'
 '    manager.close()\n'
 "    manager = ExperimentManager(root / 'store')\n"
 "    assert len(manager.get_run(identity).read_metrics()) == data['actual_epochs']\n"
 '    manager.close()\n')
