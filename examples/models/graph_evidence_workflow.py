"""Explicit bounded CPU provenance workflow; synthetic data, no performance claim."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from hyso import GraphModelBuilder, TrainingConfig, ExperimentManager, analysis, advisor
from hyso.training import NamedTensorDataset, NamedTargetDataset
from hyso.architecture_comparison import compare_architectures
from hyso.models.output import GraphTensorOutput
from hyso.checkpoint import save_training_checkpoint, load_training_checkpoint
from hyso.analysis.graph_evidence import (
    GraphEvaluationResult, GraphEvidenceError, evaluate_graph_classification, compare_graph_evaluations,
)
from hyso.reporting import create_report, report_experiment


def write(path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def output_directory(output):
    output = Path(output).resolve()
    repo = Path(__file__).resolve().parents[2]
    if output == repo or output.is_relative_to(repo):
        raise ValueError("Use fresh isolated QA outside the repository")
    output.mkdir(parents=True, exist_ok=False)
    return output


def train_named(output, *, resume_from=None):
    """Two named inputs, one selected head; resume keeps the total two-epoch budget.

    Rerun into a fresh output directory after interruption, passing the last saved
    checkpoint. Resuming an already completed budget performs no new updates.
    """
    output = output_directory(output)
    torch.set_num_threads(1)
    torch.random.default_generator.manual_seed(13618)
    builder = (GraphModelBuilder({'image': [1, 4, 4], 'hint': [1, 4, 4]})
        .add('merge', 'add', ['image', 'hint'])
        .add('head', 'classification_head', ['merge'], in_features=1, num_classes=2)
        .outputs(logits='head'))
    model = GraphTensorOutput(builder.build(), 'logits', task='multiclass', named_inputs=True)
    x = torch.arange(64, dtype=torch.float32).reshape(4, 1, 4, 4) / 64
    dataset = NamedTensorDataset({'hint': x.flip(-1), 'image': x}, torch.tensor([0, 1, 0, 1]))
    checkpoint = output / 'last.pth'
    settings = TrainingConfig(epochs=2, batch_size=2, optimizer='adam', learning_rate=.01,
        scheduler='step', scheduler_params={'step_size': 1, 'gamma': .8},
        validation=False, save_last_to=str(checkpoint),
        resume_from=None if resume_from is None else str(resume_from))
    loader, _ = settings.make_loaders(dataset)
    session = settings.prepare(model, loader, model_config=builder.to_config())
    result = session.fit()
    write(output / 'training.json', result.to_dict())
    summary = dict(history=result.history, applied_updates=result.to_dict()['applied_updates'],
        checkpoint=str(checkpoint if checkpoint.exists() else Path(resume_from)),
        completed_epochs=len(result.history['train_loss']), total_epoch_budget=2)
    write(output / 'summary.json', summary)
    return summary


def train_heads(output, *, resume_from=None, two_inputs=False, variant='A'):
    """One or two explicit inputs; two heads, strict resume and saved evaluation."""
    if variant not in ('A','B'):
        raise ValueError('Use variant A/B')
    output=output_directory(output)
    torch.set_num_threads(1)
    torch.random.default_generator.manual_seed(13619)
    width=2 if variant=='A' else 3
    inputs={'image':[1,4,4]}
    if two_inputs:inputs['hint']=[1,4,4]
    builder=(GraphModelBuilder(inputs)
        .add('left','conv2d',['image'],in_channels=1,out_channels=width)
        .add('right','conv2d',['hint' if two_inputs else 'image'],in_channels=1,out_channels=width)
        .add('disease_head','classification_head',['left'],in_features=width,num_classes=2)
        .add('severity_head','classification_head',['right'],in_features=width,num_classes=3)
        .outputs(disease='disease_head',severity='severity_head'))
    graph=builder.build()
    x=torch.arange(64,dtype=torch.float32).reshape(4,1,4,4)/64
    targets={'disease':torch.tensor([0,1,0,1]),'severity':torch.tensor([0,2,1,2])}
    data=NamedTargetDataset({'hint':x.flip(-1),'image':x} if two_inputs else x,targets)
    path=output/'last.pth'
    config=TrainingConfig(epochs=2,batch_size=2,optimizer='adam',learning_rate=.01,validation=False,
        save_last_to=str(path),resume_from=None if resume_from is None else str(resume_from))
    train,_=config.make_loaders(data)
    session=config.prepare(graph,train,model_config=builder.to_config(),
        loss_fn={name:torch.nn.CrossEntropyLoss() for name in targets},loss_weights={'disease':.3,'severity':1.2})
    training=session.fit()
    write(output/'training.json',training.to_dict())
    checkpoint=path if path.exists() else Path(resume_from)
    heads={name:dict(target=name,task='multiclass',class_names=labels,split='test',
        dataset_id='synthetic-two-heads',preprocessing_id='fixed-fp32')
        for name,labels in [('disease',['absent','present']),('severity',['low','medium','high'])]}
    sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    load_training_checkpoint(checkpoint,graph,optimizer=session.trainer.optimizer,
        scaler=session.trainer.scaler,capture_evidence=True,expected_sha256=sha)
    evaluation=analysis.evaluate_graph_heads(graph,DataLoader(data,batch_size=2),heads=heads,
        capture_structure=True,expected_checkpoint_sha256=sha)
    evaluation.to_json(output/'heads.json')
    store=output/'experiments'
    manager=ExperimentManager(store);manager.close();ExperimentManager.migrate_store(store)
    manager=ExperimentManager(store)
    try:
        experiment=manager.create_experiment('Multi-head Graph '+variant)
        record=manager.create_workflow_run(experiment['id'],'training',model_ids=[variant],
            architectures={variant:builder},training_config=config,protocol={'heads':heads,'association':'caller_declared'})
        record.start();training.log_to(record);record.record_result(training)
        cp=record.add_artifact(checkpoint,kind='checkpoint',ownership='reference',retain_locator=True)
        artifact=record.add_artifact(output/'heads.json',kind='graph_multi_head_evaluation',ownership='copy',
            metadata={'checkpoint_artifact_id':cp['id']})
        record.finish()
        opened=manager.open_result(artifact['id'])
        if opened['status']!='verified':raise ValueError('Saved evaluation failed integrity/result validation')
        opened=opened['result']
        summary=dict(model_variant=variant,history=training.history,applied_updates=training.to_dict()['applied_updates'],
            checkpoint=str(checkpoint),experiment_id=experiment['id'],run_id=record.id,
            evaluation_artifact_id=artifact['id'],
            per_head_metrics={name:opened.head(name).evaluation.to_dict()['metrics'] for name in heads},
            imported_graph_trust=opened.to_dict()['verification']['graph_v2'],
            unsupported=None)
        write(output/'summary.json',summary)
        return summary
    finally:manager.close()


def compare_saved_heads(paths, output, *, head_pairs):
    """Compare two saved statements; no training, inference or P12 capture."""
    if len(paths)!=2:raise ValueError('Select exactly two saved multi-head results')
    output=output_directory(output)
    results=[analysis.MultiHeadGraphEvaluationResult.from_json(path) for path in paths]
    comparisons=analysis.compare_graph_heads(*results,head_pairs=head_pairs)
    recommendations=[advisor.recommend(comparison) for comparison in comparisons.values()]
    for (head,comparison),recommendation in zip(comparisons.items(),recommendations):
        comparison.to_json(output/(head+'-comparison.json'))
        recommendation.to_json(output/(head+'-advisor.json'))
    for language in ('tr','en'):
        report=create_report(*results,*comparisons.values(),*recommendations,language=language)
        for suffix,export in [('json',report.to_json),('md',report.to_markdown),('html',report.to_html)]:
            export(output/(language+'.'+suffix))
    summary={head:dict(metrics=c.to_dict()['metrics'],structure=c.to_dict()['structure']['status'])
             for head,c in comparisons.items()}
    write(output/'summary.json',summary)
    return summary


def compare_saved(manager, experiment_id, artifact_ids, output):
    """Interpret saved bytes only; architecture metadata never proves trained state."""
    if len(artifact_ids) != 2:
        raise ValueError('Select exactly two Graph evaluation artifacts')
    output = Path(output)
    output.mkdir()
    opened = [manager.open_result(identity) for identity in artifact_ids]
    if not all(value['status'] == 'verified' for value in opened):
        raise ValueError('Both saved result artifacts must pass integrity checks')
    results = [value['result'] for value in opened]
    sources = [result.analysis_source(label) for label, result in zip(('A', 'B'), results)]
    comparison = analysis.compare_models(*sources, features=('P26', 'P12'),
        include_structure=True, context_policy='strict')
    comparison.to_json(output / 'comparison.json')
    recommendations = advisor.recommend(comparison)
    recommendations.to_json(output / 'advisor.json')
    configs = [result.to_dict()['source_receipt']['contract']['configuration'] for result in results]
    write(output / 'architectures.json', compare_architectures(configs, names=['A', 'B']))
    record = manager.create_workflow_run(experiment_id, 'comparison',
        model_ids=['A', 'B'], comparison=comparison)
    record.start()
    saved = record.record_result(comparison, input_artifact_ids=artifact_ids)
    for label, source in zip(('A', 'B'), sources):
        path = output / (label + '-analysis.json')
        analysis.UnifiedAnalysisReport(source.to_dict()['report']).to_json(path)
        record.add_artifact(path, kind='analysis_report', ownership='copy', retain_locator=True)
    record.finish()
    assert manager.open_result(saved['id'])['result'].to_dict() == comparison.to_dict()
    for language in ('tr', 'en'):
        report = report_experiment(manager, experiment_id, language=language, level='expert')
        for suffix, export in [('json', report.to_json), ('md', report.to_markdown), ('html', report.to_html)]:
            export(output / (language + '.' + suffix))
    data = comparison.to_dict()
    return dict(metrics=data['metrics'], structure=data['structure']['status'],
        saved_comparison_id=saved['id'], comparison_run_id=record.id, reports=str(output),
        imported_graph_trust=[r.to_dict()['verification']['graph_v2'] for r in results])


def run(output, *, capture_receiver_runtime=False):
    output = output_directory(output)
    torch.set_num_threads(1)
    train_x = torch.arange(64, dtype=torch.float32).reshape(4, 1, 4, 4) / 64
    test_x = torch.arange(64, 128, dtype=torch.float32).reshape(4, 1, 4, 4) / 128
    labels = torch.tensor([0, 1, 0, 1])
    train = DataLoader(TensorDataset(train_x, labels), batch_size=2, shuffle=False)
    test = DataLoader(TensorDataset(test_x, labels), batch_size=2, shuffle=False)
    context = dict(dataset_id="synthetic-stage13-5-11-closure" if capture_receiver_runtime else "synthetic-stage13-5-10",
        split="test", preprocessing_id="fixed_fp32_identity")
    settings = TrainingConfig(epochs=1, batch_size=2, shuffle=False, validation=False,
        device="cpu", precision="fp32", optimizer="sgd", learning_rate=0.01)
    models, associations, trainings, paths, initial_views = [], [], [], [], []
    store = output / "experiments"
    manager = ExperimentManager(store)
    manager.close()
    ExperimentManager.migrate_store(store)
    manager = ExperimentManager(store)
    artifacts, checks, run_ids, evaluation_artifact_ids = [], [], [], []
    try:
        experiment = manager.create_experiment("Graph v2 source/evaluation scope")
        for index, activation in enumerate(["silu", "relu"]):
            directory = output / ("model-" + str(index))
            directory.mkdir()
            torch.random.default_generator.manual_seed(13610 + index)
            b = (GraphModelBuilder({"image": [1, 4, 4]})
                .add("stem", "conv_norm_act", ["image"], in_channels=1, out_channels=2,
                     norm="batch", activation=activation)
                .add("left", "conv2d", ["stem"], in_channels=2, out_channels=2, kernel_size=1, padding=0)
                .add("merge", "add", ["left", "stem"])
                .add("head", "classification_head", ["merge"], in_features=2, num_classes=2)
                .outputs(logits="head", features="merge"))
            label = ('A', 'B')[index]
            mr = manager.create_workflow_run(experiment['id'], 'model', model_ids=[label],
                architectures={label: b}, protocol={'adapter_output': 'logits', 'state_dict_prefix': 'graph.'})
            mr.start()
            write(directory / 'architecture.json', b.to_config())
            architecture_artifact = mr.add_artifact(directory / 'architecture.json', kind='graph_architecture', ownership='copy')
            artifacts.append((architecture_artifact, directory / 'architecture.json'))
            mr.finish()
            run_ids.append(mr.id)
            def fresh():
                return GraphTensorOutput(b.build(), "logits", task="multiclass").float().cpu()
            source = fresh()
            training = settings.prepare(source, train).fit()
            source.eval()
            metadata = dict(seed=13610 + index, architecture=b.to_config(), settings=settings.to_dict(),
                fixture="explicit_synthetic_not_real_dataset", evaluation_context=context)
            cp = directory / "model.pth"
            save_training_checkpoint(cp, source, graph_contract=True, include_rng=False,
                metrics=metadata, history=training.history, completed_epochs=1)
            sha = hashlib.sha256(cp.read_bytes()).hexdigest()
            restored = fresh().eval()
            def validator(saved):
                if saved["metrics"] != metadata:
                    raise ValueError("checkpoint metadata mismatch")
            load_training_checkpoint(cp, restored, capture_evidence=True, expected_sha256=sha,
                validate_metadata=validator)
            with torch.no_grad():
                assert torch.equal(source(test_x), restored(test_x))
            result = evaluate_graph_classification(restored, test, class_names=["a", "b"], split="test",
                source_id=context["dataset_id"], context=context, expected_checkpoint_sha256=sha,
                capture_receiver_runtime=capture_receiver_runtime, capture_structure=True)
            assert result.to_dict()["verification"]["graph_v2"] == "verified"
            assert result.to_dict()["relation"]["status"] == ("verified" if capture_receiver_runtime else "limited")
            initial_views.append(result.to_dict()["verification"])
            assert result.evaluation.to_dict()["producer_evidence"]["checkpoint"]["status"] == "missing"
            models.append(restored)
            associations.append(result)
            trainings.append(training)
            paths.append(cp)
            write(directory / "training.json", training.to_dict())
            result.to_json(directory / "association.json")
            result.evaluation.to_json(directory / "evaluation.json")
            tr = manager.create_run(experiment["id"], "training", config=metadata)
            tr.start()
            training.log_to(tr)
            inputs = []
            for file, kind in [(cp, "checkpoint"), (directory / "training.json", "training_result")]:
                artifact = tr.add_artifact(file, kind=kind, ownership="reference")
                inputs.append(artifact)
                artifacts.append((artifact, file))
            tr.finish()
            er = manager.create_run(experiment["id"], "evaluation", config=metadata,
                provenance={"source": "native evaluation; separate source v2; " + (
                    "Core-observed receiver binding" if capture_receiver_runtime else "limited copy binding")})
            er.start()
            for artifact in inputs:
                manager.link_artifact(er.id, artifact["id"], role="input")
            for file, kind in [(directory / "association.json", "graph_evaluation_association"),
                               (directory / "evaluation.json", "evaluation")]:
                artifact = er.add_artifact(file, kind=kind, ownership="reference", retain_locator=True,
                    metadata={"receipt_scope": "retained_statement_bytes_not_process_local_trust"})
                artifacts.append((artifact, file))
                if kind == 'graph_evaluation_association':
                    evaluation_artifact_ids.append(artifact['id'])
            er.finish()
            run_ids.extend([tr.id, er.id])
        compared = compare_graph_evaluations(*associations)
        assert compared.to_dict()["status"] == "COMPARABLE"
        compared.to_json(output / "comparison.json")
        compared.legacy_comparison.to_json(output / "legacy-comparison.json")
        report = create_report(*trainings, *associations, compared, language="en", level="technical")
        for suffix, method in [("json", report.to_json), ("md", report.to_markdown), ("html", report.to_html)]:
            method(output / ("report." + suffix))
        rr = manager.create_run(experiment["id"], "evaluation", config=compared.legacy_comparison.tracking_config())
        rr.start()
        for artifact, file in artifacts:
            manager.link_artifact(rr.id, artifact["id"], role="input")
        for name in ["comparison.json", "legacy-comparison.json", "report.json", "report.md", "report.html"]:
            file = output / name
            artifacts.append((rr.add_artifact(file, kind="report", ownership="reference"), file))
        rr.finish()
        run_ids.append(rr.id)
        saved_comparison = compare_saved(manager, experiment['id'], evaluation_artifact_ids, output / 'saved-results')
        run_ids.append(saved_comparison['comparison_run_id'])
        for artifact, file in artifacts:
            checks.append(manager.verify_artifact(artifact["id"], path=file))
        assert all(v["status"] == "verified" for v in checks)
        imported = GraphEvaluationResult.from_json(output / "model-0/association.json")
        assert imported.to_dict()["verification"]["graph_v2"] == "unverified"
        bad = output / "negative-artifact-copy.json"
        bad.write_text("{}", encoding="utf8")
        probe_artifact, probe_file = next((artifact, file) for artifact, file in artifacts
                                         if artifact['kind'] == 'checkpoint')
        negative = manager.verify_artifact(probe_artifact['id'], path=bad)
        assert negative["status"] == "mismatch"
        assert manager.verify_artifact(probe_artifact['id'], path=probe_file)["status"] == "verified"
        try:
            evaluate_graph_classification(models[0], test, class_names=["a", "b"], split="test",
                expected_checkpoint_sha256="0" * 64)
        except GraphEvidenceError:
            wrong_identity_rejected = True
        else:
            raise AssertionError("wrong expected checkpoint identity accepted")
        with torch.no_grad():
            next(models[0].parameters()).add_(1)
        mutation = associations[0].to_dict()
        assert mutation["verification"]["graph_v2"] == "missing"
        write(output / "mutation.json", mutation)
        summary = dict(seeds=[13610, 13611], device="cpu", precision="fp32", epochs_per_model=1,
            applied_updates=[t.to_dict()["applied_updates"] for t in trainings],
            accuracy=[v.evaluation.to_dict()["metrics"]["accuracy"] for v in associations],
            comparison_status=compared.to_dict()["status"], claims=compared.legacy_comparison.to_dict()["claims"],
            legacy_p26=[v.evaluation.to_dict()["producer_evidence"]["checkpoint"]["status"] for v in associations],
            graph_v2_at_report_creation=[v["graph_v2"] for v in initial_views],
            cross_binding="verified;supported quiescent CPU receiver" if capture_receiver_runtime else "limited;isolated runtime unobserved",
            imported_json_status="unverified", mutation_status="missing", wrong_identity_rejected=wrong_identity_rejected,
            experiment_id=experiment["id"], run_ids=run_ids, artifact_verification=checks,
            saved_comparison=saved_comparison,
            negative_artifact_check=negative, checkpoint_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths])
        if capture_receiver_runtime:
            summary.update(receiver_runtime_at_report_creation=[v["receiver_runtime"] for v in initial_views],
                inference_binding_at_report_creation=[v["inference_binding"] for v in initial_views],
                mutation_receiver_runtime=mutation["verification"]["receiver_runtime"])
        write(output / "summary.json", summary)
        return summary
    finally:
        manager.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-receiver-runtime", action="store_true")
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--named-training', action='store_true', help='Two-input CPU training/checkpoint example')
    modes.add_argument('--multi-head', action='store_true', help='Two-head CPU training, checkpoint and saved evaluation')
    modes.add_argument('--compare-heads', type=Path, nargs=2, metavar=('A','B'), help='Compare two saved heads.json files')
    parser.add_argument('--two-inputs', action='store_true', help='Use two named inputs in the full --multi-head workflow')
    parser.add_argument('--variant', choices=('A','B'), default='A')
    parser.add_argument('--resume-from', type=Path, help='Last compatible training checkpoint; same total budget')
    args = parser.parse_args()
    if args.capture_receiver_runtime and (args.named_training or args.multi_head or args.compare_heads):
        parser.error('--capture-receiver-runtime belongs to the original evaluation example')
    if (args.two_inputs or args.variant!='A') and not args.multi_head:
        parser.error('--two-inputs/--variant require --multi-head')
    if args.named_training:
        result = train_named(args.output, resume_from=args.resume_from)
    elif args.multi_head:
        result = train_heads(args.output,resume_from=args.resume_from,two_inputs=args.two_inputs,variant=args.variant)
    elif args.compare_heads:
        if args.resume_from is not None:parser.error('--resume-from requires a training example')
        result = compare_saved_heads(args.compare_heads,args.output,
            head_pairs={'disease':'disease','severity':'severity'})
    else:
        if args.resume_from is not None:
            parser.error('--resume-from requires --named-training or --multi-head')
        result = run(args.output, capture_receiver_runtime=args.capture_receiver_runtime)
    print(json.dumps(result))
