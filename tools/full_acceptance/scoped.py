"""Single-stage preparation and authorization; all unrelated approvals remain closed.

Only newly assembled, unconsumed manifests are narrowed and resealed. Historical
manifests/results are read-only. Authorize and execute require separate commands.
"""
from pathlib import Path
import argparse
import json
import shutil
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'tools')]
from full_acceptance import core, cli, reporting


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scope_ids(stage):
    # Keep P13's existing strict regression gate. Its CPU prerequisite is part
    # of the reviewed plan; no dependency or acceptance threshold is relaxed.
    return ['02.regression',stage] if stage=='06.p13_capture' else [stage]


def check_scope(m, stage, approved):
    selected=scope_ids(stage)
    require(m.get('execution_scope') == selected, 'sealed scope mismatch')
    actual = [e['id'] for e in m['experiments'] if e['authorization']['approved']]
    require(actual == (selected if approved else []), 'unrelated stage authorization present')
    require(m['execution_policy'] == 'continue_independent', 'independent source-pinned policy required')
    require(m['optional']['scoped_driver']['configuration']['stage'] == stage, 'driver scope mismatch')
    return actual


def seal_fresh(m, stage, approved):
    out = Path(m['output'])
    require(not (out/'STARTED.json').exists() and not (out/'preparation').exists()
            and not (out/'stages').exists(), 'cannot reseal a consumed manifest')
    previous = core.record(out/'manifest.json')
    selected=scope_ids(stage)
    for e in m['experiments']:
        if e['id'] not in selected or not approved:
            e['authorization'] = dict(approved=False, approved_at=None,
                basis='NOT_AUTHORIZED: task explicitly excludes this stage' if e['id'] not in selected
                      else 'NOT_AUTHORIZED: awaiting separate explicit operator approval')
    m['execution_scope'] = selected
    m['optional']['scoped_driver'] = dict(file=core.record(__file__), configuration=dict(stage=stage,
        policy='all other experiments NOT_AUTHORIZED; unchanged acceptance thresholds'))
    check_scope(m, stage, approved)
    core.publish(out/'manifest.json', m)
    (out/'manifest.sha256').write_text(core.sha(out/'manifest.json')+'\n', encoding='ascii')
    core.publish(out/'scope-assembly.json', dict(previous_unconsumed_draft=previous,
        manifest=core.record(out/'manifest.json'), stage=stage,
        approved_experiments=check_scope(m, stage, approved), historical_manifest_modified=False), new=True)
    return core.verify(out/'manifest.json', expected=core.sha(out/'manifest.json'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare','authorize','execute','audit'))
    parser.add_argument('--stage', required=True, choices=('08.camera','06.p13_capture','06.layer'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--sha256')
    parser.add_argument('--approval-text')
    parser.add_argument('--camera-config', type=Path)
    parser.add_argument('--p13-nsys-plan', type=Path)
    parser.add_argument('--p13-evidence', type=Path)
    args = parser.parse_args(argv)
    if args.mode == 'prepare':
        require(args.output is not None and not args.output.exists(), 'new exclusive preparation directory required')
        m = core.prepare(args.output, camera_config=args.camera_config, p13_nsys_plan=args.p13_nsys_plan,
                         p13_evidence=args.p13_evidence, continue_independent=True)
        m = seal_fresh(m, args.stage, False)
        manifest = args.output/'manifest.json'
        code = cli.preflight(SimpleNamespace(manifest=manifest, approved_sha256=core.sha(manifest),
                                             continue_independent=False))
        require(code == 0, 'metadata preflight failed')
    else:
        require(args.manifest is not None and args.sha256, 'exact manifest and SHA256 required')
        m = core.verify(args.manifest, expected=args.sha256)
        require(not (Path(m['output'])/'STARTED.json').exists(), 'consumed manifests cannot be authorized or executed again')
        if args.mode == 'authorize':
            check_scope(m, args.stage, False)
            require(args.approval_text is not None and bool(args.approval_text.strip()), 'new explicit approval required')
            require(args.output is not None and not args.output.exists(), 'new authorized directory required')
            parent_preflight = core.read(Path(m['output'])/'preflight.json')
            require(parent_preflight['status'] == 'PASS', 'parent metadata preflight not PASS')
            m = core.authorize(args.manifest, args.output, scope_ids(args.stage), expected=args.sha256,
                               basis=args.approval_text)
            m = seal_fresh(m, args.stage, True)
            reporting.write_reports(m, reporting.initial_results(m))
            cli.commands(m)
            cli.preserve(m)
            core.publish(args.output/'run-status.json', dict(state='AUTHORIZED_PREPARED',
                run_id=m['run_id'], manifest_sha256=core.sha(args.output/'manifest.json'),
                controller_active=False, suite_complete=False, updated_at=core.now()), new=True)
        elif args.mode == 'execute':
            check_scope(m, args.stage, True)
            if args.stage == '08.camera':
                from hyso.performance._windows_camera import verify_binding
                cfg = m['optional']['camera_adapter']['configuration']
                verify_binding(cfg.get('camera_binding'),cfg.get('camera_index',0))
            if args.stage == '06.layer':
                cfg = m.get('optional',{}).get('p13_evidence',{}).get('configuration')
                require(cfg is not None and cfg.get('capture'), 'new finalized P13 evidence required before layer acceptance')
            return cli.execute(SimpleNamespace(manifest=args.manifest, approved_sha256=args.sha256, cuda_subtasks=False))
        else:
            actual = [e['id'] for e in m['experiments'] if e['authorization']['approved']]
            check_scope(m, args.stage, bool(actual))
    print(json.dumps(dict(output=m['output'], stage=args.stage,
        manifest_sha256=core.sha(Path(m['output'])/'manifest.json'),
        approved_experiments=[e['id'] for e in m['experiments'] if e['authorization']['approved']],
        native_execution_started=False)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
