"""Executable regression examples, preserving every original Python block."""
BLOCKS = ('import torch\n'
 'from torch import nn\n'
 'from torch.utils.data import DataLoader, TensorDataset\n'
 'from hyso.training import Trainer\n'
 'from hyso.performance import DataLoaderAnalysisProtocol, measure_training_with_loader\n'
 '\n'
 'dataset = TensorDataset(torch.ones(5, 3), torch.tensor([0, 1, 0, 1, 0]))\n'
 'train = DataLoader(dataset, batch_size=2, shuffle=True,\n'
 '                   generator=torch.Generator().manual_seed(10), num_workers=0)\n'
 'model = nn.Linear(3, 2)\n'
 'trainer = Trainer(model, optimizer=torch.optim.AdamW(model.parameters(), lr=.01),\n'
 '                  loss_fn=nn.CrossEntropyLoss(), accum_steps=2)\n'
 'result = measure_training_with_loader(trainer, train, epochs=2,\n'
 '    loader_protocol=DataLoaderAnalysisProtocol(max_records=1000))\n'
 'data = result.loader.to_dict()\n'
 "assert [s['samples'] for s in data['summaries']] == [5, 5]\n"
 "assert [s['batches'] for s in data['summaries']] == [3, 3]\n"
 "assert [r['samples'] for r in data['observations'] if r['kind'] == 'batch'] == [2, 2, 1, 2, 2, "
 '1]\n'
 "assert result.throughput.to_dict()['segment']['completed']['samples'] == 10\n"
 "assert data['worker_service']['value'] is None\n",
 'def spawn_example():\n'
 '    loader = DataLoader(dataset, batch_size=2, num_workers=2,\n'
 "        multiprocessing_context='spawn', prefetch_factor=2, persistent_workers=True,\n"
 '        generator=torch.Generator().manual_seed(10))\n'
 '    model = nn.Linear(3, 2)\n'
 '    trainer = Trainer(model, optimizer=torch.optim.SGD(model.parameters(), lr=.01),\n'
 '                      loss_fn=nn.CrossEntropyLoss())\n'
 '    return measure_training_with_loader(trainer, loader, epochs=2)\n'
 '\n'
 "if __name__ == '__main__':\n"
 '    spawned_result = spawn_example()\n',
 'import hashlib\n'
 'from pathlib import Path\n'
 'import tempfile\n'
 'from hyso.experiments import ExperimentManager\n'
 '\n'
 '# Ölçüm bitti; export/tracking hatasında fit tekrar çağrılmaz.\n'
 "with tempfile.TemporaryDirectory(prefix='hyso-p21-') as directory:\n"
 '    root = Path(directory)\n'
 "    manager = ExperimentManager(root / 'store')\n"
 '    manager.close()\n'
 "    ExperimentManager.migrate_store(root / 'store')  # mevcut artifact store akışı\n"
 "    manager = ExperimentManager(root / 'store')\n"
 "    run = manager.create_run(manager.create_experiment('P21')['id'], 'training', {})\n"
 '    run.start()\n'
 "    for suffix in ('json', 'csv'):\n"
 "        path = root / ('loader.' + suffix)\n"
 "        getattr(result.loader, 'to_' + suffix)(path)\n"
 "        artifact = run.add_artifact(path, kind='report', ownership='copy',\n"
 "            metadata={'source_segment_id': result.timing.to_dict()['segment_id']})\n"
 "        assert artifact['expected_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()\n"
 "        assert manager.verify_artifact(artifact['id'])['status'] == 'verified'\n"
 "    for summary in data['summaries']:\n"
 "        if summary['phase'] == 'training':\n"
 "            run.log_metric(name='loader.next_wait', value=summary['next_wait_ms'], unit='ms',\n"
 "                           source='P21', epoch=summary['epoch_id'])\n"
 '    run.finish()\n'
 '    identity = run.id\n'
 '    manager.close()\n'
 "    manager = ExperimentManager(root / 'store')\n"
 '    assert len(manager.get_run(identity).read_metrics()) == 2\n'
 '    assert len(manager.list_artifacts(identity)) == 2\n'
 '    manager.close()\n')
