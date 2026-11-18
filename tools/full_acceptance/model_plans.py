"""Prepare two independent, unexecuted P01-P35 model manifests after P23 PASS."""
import argparse
import shutil
from pathlib import Path

from . import core,cli,reporting,registry


def prepare_pair(output,reference_manifest,p23_evidence,*,p13_plan,ncu_number_format):
    evidence=core.read(p23_evidence)
    if evidence.get('decision')!='PASS' or any(evidence['models'][arm]['cuda']['decision']!='PASS'
            or evidence['models'][arm]['cpu']['decision']!='PASS' for arm in ('A','B')):
        raise ValueError('both-model P23 Basic CPU/CUDA PASS evidence required')
    base=core.read(reference_manifest)
    if any(evidence['models'][arm]['cuda']['checkpoint_sha256']!=base['assets']['models'][arm]['sha256'] for arm in ('A','B')):
        raise ValueError('P23 acceptance and full-plan checkpoint identities differ')
    output=Path(output).resolve();output.mkdir()
    configs=output/'configuration';configs.mkdir()
    camera=None
    if 'camera_adapter' in base['optional']:
        camera=configs/'camera.json';core.publish(camera,base['optional']['camera_adapter']['configuration'],new=True)
    ncu=dict(base['optional']['ncu_plan']['configuration'],number_format=ncu_number_format,
             number_format_basis='explicit declared plan choice; validate fresh CSV schema at capture')
    ncu_path=configs/'ncu.json';core.publish(ncu_path,ncu,new=True)
    nsys=core.read(p13_plan);plans=[]
    for arm in ('A','B'):
        nsys_path=configs/('nsys-'+arm+'.json')
        core.publish(nsys_path,dict(nsys,protocol='trained_'+arm+'_3_forward_module_nvtx'),new=True)
        destination=output/('Model-'+arm)
        if plans:
            # Same unchanged workspace; preserve the complete original baseline.
            destination.mkdir()
            shutil.copyfile(Path(plans[0]['directory'])/'preservation-before.json',destination/'preservation-before.json')
        manifest=core.prepare(destination,model_arm=arm,gpu_uuid=base['hardware']['gpu_uuid'],
            camera_config=camera,ncu_plan=ncu_path,p13_nsys_plan=nsys_path,
            ort_cpu=base['hardware'].get('ort_cpu'),ort_cuda=base['hardware'].get('ort_cuda'))
        path=destination/'manifest.json'
        cli.commands(manifest)
        rows=reporting.initial_results(manifest)
        reporting.write_reports(manifest,rows,state='PREPARED_NOT_EXECUTED')
        prerequisites={e['id']:reporting.prerequisites(manifest,e) for e in manifest['experiments']}
        feature_plan=[dict(id=ident,model_arm=arm,experiments=ids,
            devices=['cpu'] if ident=='P24' else ['cpu','cuda:0'] if ident in ('P19','P20','P21','P22','P23','P25') else 'explicit experiment devices',
            execution='NOT_RUN',decision='NOT_EVALUATED',historical_results_reused=False)
            for ident,ids in registry.FEATURE_EXPERIMENTS.items()]
        assert len(feature_plan)==35
        plan=dict(format='hyso.per_model.full_acceptance_preparation',model_arm=arm,
            manifest=core.record(path),checkpoint=manifest['assets']['models'][arm],
            p23_basic_reference=core.record(p23_evidence),suite_complete=False,fresh_pass_count=0,
            features=feature_plan,prerequisites=prerequisites,
            native_authorization='not granted by preparation; no execute was invoked',
            scope_notes=['Comparison features intentionally construct both fresh A/B arms as peer context.',
                'P13/layer, training, energy and long-run checkpoint routes select this manifest model.',
                'Camera/video uses the explicitly bound task-appropriate trained adapter, a separate pipeline reference.',
                'P24 target API is CPU-only; no unsupported CUDA target measurement is invented.',
                'Legacy reconciliation matrix retains historical decisions; it is not a fresh 35/35 result.'])
        core.publish(destination/'per-model-P01-P35-plan.json',plan,new=True)
        lines=['# Model '+arm+' — bağımsız P01–P35 koşu planı','',
            '**PREPARED / NOT EXECUTED. Yeni PASS: 0/35.**','',
            'Checkpoint SHA256: `'+plan['checkpoint']['sha256']+'`. Manifest: `manifest.json`.',
            '05.training bu modelde CPU ve CUDA için ayrı gözlemci fit’leri çalıştırır; P23 CUDA peak reset yalnız owned worker içindedir. P24 target API CPU kapsamındadır.',
            'Karşılaştırma özellikleri A/B peer bağlamını açıkça içerir; tüm yeni çıktılar bu manifestin kendi dizinine yazılır.',
            'Native aşamalar bu hazırlıkla yetkilendirilmedi ve çalıştırılmadı. `commands.ps1` mevcut güvenli authorization/execute akışını gösterir.',
            'Eski kabul kararları yeni 35/35 başarı yerine kullanılmaz. Her gerekli aşamanın gerçek terminal/cleanup/ölçüm kanıtı gerekir.','',
            '| Özellik | Deneyler | Yeni koşu |','| --- | --- | --- |']
        lines += ['| '+f['id']+' | '+', '.join(f['experiments'])+' | NOT_RUN |' for f in feature_plan]
        lines += ['', 'Metadata önkoşulları: '+str(prerequisites),
            'Execution-time GPU/tool privileges, actual camera availability, OS resources and native outcomes remain unverified.','']
        (destination/'per-model-plan.md').write_text('\n'.join(lines),encoding='utf-8')
        core.verify(path,expected=core.sha(path),live=True)
        plans.append(dict(arm=arm,directory=str(destination),manifest=core.record(path),
            features=35,executed=0,fresh_pass=0,known_missing={k:v for k,v in prerequisites.items() if v}))
    core.publish(output/'plans-index.json',dict(p23_basic=core.record(p23_evidence),plans=plans,
        native_full_runs_started=False,new_35_pass=False),new=True)
    return plans


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference-manifest',type=Path,required=True)
    p.add_argument('--p23-evidence',type=Path,required=True)
    p.add_argument('--p13-plan',type=Path,required=True)
    p.add_argument('--ncu-number-format',choices=('decimal_point','decimal_comma'),required=True)
    args=p.parse_args()
    print(prepare_pair(args.output,args.reference_manifest,args.p23_evidence,
        p13_plan=args.p13_plan,ncu_number_format=args.ncu_number_format))


if __name__=='__main__':main()
