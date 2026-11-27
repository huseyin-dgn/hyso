"""Explicit deployment conversion and read-only retained export presentation."""
from pathlib import Path
from .common import Outcome,command,output_check,export_text,json_text,CLIError


def register(sub):
    p=command(sub,'export','Explicit deployment artifacts, CPU validation and saved evidence; optional dependencies are never installed.',run,export=False)
    actions=p.add_subparsers(dest='action',required=True)
    command(actions,'dependencies','Discover optional dependency metadata without importing native runtimes.',run,export=False)
    for action in ('inspect','advise','report'):
        q=command(actions,action,'Read saved export evidence without loading a model.',run)
        q.add_argument('input',type=Path);q.add_argument('--language',choices=('tr','en','de','es'),default='en')
        q.add_argument('--format',choices=('json','markdown','html'),default='json' if action=='inspect' else 'markdown')
    for action in ('fp16','quantize'):
        q=command(actions,action,'Create a separate ONNX precision artifact from a verified export manifest.',run)
        q.add_argument('input',type=Path);q.add_argument('--example',type=Path,help='Named CPU input arrays in .npz; no pickle.')
        q.add_argument('--mode',choices=('dynamic','static'),default='dynamic')
        q.add_argument('--calibration',type=Path,nargs='+',help='User calibration .npz batches, bounded to 64.')
        q.add_argument('--dataset-id');q.add_argument('--representative',action='store_true')
        q.add_argument('--change-io-dtype',action='store_true')
    for action in ('benchmark','validate'):
        q=command(actions,action,'Explicit small CPU runtime execution; native GPU requests use the owned worker API.',run)
        q.add_argument('input',type=Path);q.add_argument('--example',type=Path,required=True)
        q.add_argument('--execute',action='store_true');q.add_argument('--trust-artifact',action='store_true')
        q.add_argument('--iterations',type=int,default=10);q.add_argument('--warmup',type=int,default=2)
        q.add_argument('--candidate',type=Path);q.add_argument('--labels',type=Path,help='Integer labels in .npy; no pickle.')
        q.add_argument('--precision',choices=('fp32','fp16','int8'),default='fp32')
        q.add_argument('--device',choices=('cpu','cuda'),default='cpu',help='CUDA selects the owned TensorRT runtime, never resident execution.')
        q.add_argument('--native-config',type=Path);q.add_argument('--evidence-directory',type=Path)
    q=command(actions,'compare','Compare compatible saved deployment benchmark records.',run)
    q.add_argument('baseline',type=Path);q.add_argument('candidate',type=Path)
    q=command(actions,'convert','Explicit trusted Python factory to ONNX/TorchScript/ExportedProgram.',run)
    q.add_argument('--factory',required=True,help='Trusted module:callable returning an nn.Module.')
    q.add_argument('--example',type=Path,required=True);q.add_argument('--backend',choices=('onnx','libtorch','torch_export'),default='onnx')
    q.add_argument('--execute',action='store_true');q.add_argument('--trust-factory',action='store_true')
    q=command(actions,'tensorrt','Explicit owned TensorRT engine build with pre-admission DLL/UUID/budget manifest.',run)
    q.add_argument('input',type=Path);q.add_argument('--example',type=Path,required=True)
    q.add_argument('--native-config',type=Path,required=True);q.add_argument('--profiles',type=Path)
    q.add_argument('--precision',choices=('fp32','fp16','int8'),default='fp32')
    q.add_argument('--execute',action='store_true');q.add_argument('--trust-artifact',action='store_true')


def _load(path):
    import json
    p=Path(path)
    if not p.is_file() or p.is_symlink() or p.stat().st_size>16*1024**2:raise CLIError('Bounded regular saved result required.')
    row=json.loads(p.read_text(encoding='utf-8'))
    if row.get('format')=='hyso.export.record':
        from hyso.export import load
        return load(str(p))
    if row.get('format')=='hyso.export':
        from hyso.export import ExportResult
        return ExportResult.from_json(p)
    return row


def _feed(path):
    import zipfile,numpy as np,torch
    p=Path(path)
    if p.stat().st_size>16*1024**2:raise CLIError('Input archive exceeds 16 MiB.')
    with zipfile.ZipFile(p) as z:
        if len(z.infolist())>64 or sum(r.file_size for r in z.infolist())>16*1024**2:raise CLIError('Expanded input archive exceeds bounded profile.')
    with np.load(p,allow_pickle=False) as f:
        return {n:torch.from_numpy(f[n].copy()) for n in f.files}


def run(args):
    from hyso import export as ex
    action=args.action
    if action=='dependencies':
        value=ex.availability();return Outcome(value,text=json_text(value))
    sources=[getattr(args,n,None) for n in ('input','candidate','baseline','example','labels','native_config','profiles')]
    sources+=getattr(args,'calibration',None) or []
    output_check(args,inputs=sources)
    if action in ('inspect','advise','report'):
        value=_load(args.input)
        if action=='advise':value=ex.advise(value,language=args.language)
        raw=ex.render(value,format=args.format,language=args.language)
    elif action in ('fp16','quantize'):
        if args.output is None:raise CLIError('Specify a separate --output artifact path.')
        source=ex.ExportResult.from_json(args.input);feed=_feed(args.example) if args.example else None
        if action=='fp16':value=ex.fp16(source,args.output,example_input=feed,keep_io_types=not args.change_io_dtype,config=ex.ExportConfig(overwrite=args.overwrite,atol=1e-3,rtol=1e-2))
        else:
            if args.calibration and len(args.calibration)>64:raise CLIError('At most 64 calibration batches.')
            calibration=(_feed(p) for p in args.calibration) if args.calibration else None
            value=ex.quantize(source,args.output,mode=args.mode,calibration=calibration,dataset_id=args.dataset_id,
                representative=args.representative,example_input=feed,config=ex.ExportConfig(overwrite=args.overwrite,atol=.1,rtol=.1))
        return Outcome(value.to_dict(),code=0 if value.to_dict()['deployment_verified'] else 6,text=value.to_json())
    elif action in ('benchmark','validate'):
        if not args.execute or not args.trust_artifact:raise CLIError('Runtime execution requires --execute --trust-artifact.')
        native=None
        if args.device=='cuda':
            if args.native_config is None or args.evidence_directory is None:raise CLIError('CUDA requires --native-config and --evidence-directory.')
            from .common import read_json
            native=ex.NativeConfig(**read_json(args.native_config,limit=1024*1024))
        source=ex.Runtime(ex.ExportResult.from_json(args.input),device=args.device if action=='benchmark' else 'cpu',
                          native=native,evidence_directory=args.evidence_directory);feed=_feed(args.example)
        # ONNX keeps named feeds. Torch exporters default to positional inputs.
        inputs=feed if source.info['backend']=='onnx' else tuple(feed.values())
        if action=='benchmark':value=ex.benchmark(source,inputs,warmup=args.warmup,iterations=args.iterations)
        else:
            if args.candidate is None:raise CLIError('Specify --candidate manifest.')
            candidate=ex.Runtime(ex.ExportResult.from_json(args.candidate),device=args.device,native=native,evidence_directory=args.evidence_directory)
            if args.labels:
                import numpy as np
                if args.labels.stat().st_size>16*1024**2:raise CLIError('Label file exceeds budget.')
                cohort=[(inputs,np.load(args.labels,allow_pickle=False))]
            else:cohort=[inputs]
            value=ex.validate(source,candidate,cohort,precision=args.precision,labelled=bool(args.labels))
        raw=json_text(value)
    elif action=='compare':
        value=ex.compare(_load(args.baseline),_load(args.candidate));raw=json_text(value)
    elif action=='convert':
        if not args.execute or not args.trust_factory or args.output is None:raise CLIError('Conversion requires --execute --trust-factory and --output.')
        from importlib import import_module
        if ':' not in args.factory:raise CLIError('Expected module:callable factory.')
        module,name=args.factory.split(':',1);model=getattr(import_module(module),name)();feed=_feed(args.example)
        value=getattr(ex,args.backend)(model,tuple(feed.values()),args.output,config=ex.ExportConfig(overwrite=args.overwrite))
        return Outcome(value.to_dict(),code=0 if value.to_dict()['deployment_verified'] else 6,text=value.to_json())
    elif action=='tensorrt':
        if not args.execute or not args.trust_artifact or args.output is None:raise CLIError('Native conversion requires --execute --trust-artifact and --output.')
        from .common import read_json
        native=ex.NativeConfig(**read_json(args.native_config,limit=1024*1024))
        profiles=read_json(args.profiles) if args.profiles else None
        tolerance=ex.Tolerance.for_precision(args.precision)
        value=ex.tensorrt(ex.ExportResult.from_json(args.input),args.output,example_input=_feed(args.example),native=native,
            precision=args.precision,profiles=profiles,config=ex.ExportConfig(overwrite=args.overwrite,atol=tolerance.atol,rtol=tolerance.rtol))
        return Outcome(value.to_dict(),code=0 if value.to_dict()['deployment_verified'] else 6,text=value.to_json())
    else:raise CLIError('Unknown export action.')
    export_text(args,raw)
    failed=isinstance(locals().get('value'),dict) and (value.get('status')=='FAIL' or value.get('numerical',{}).get('status')=='FAIL' or value.get('quality',{}).get('status')=='FAIL')
    data=value.to_dict() if isinstance(value,ex.ExportResult) else value
    return Outcome(data,code=5 if failed else 6 if isinstance(data,dict) and data.get('status')=='BLOCKED' else 0,text=raw)
