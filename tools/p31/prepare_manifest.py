"""Pin a fresh integration snapshot; preparation never executes a workload."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

import launch_p31 as launcher


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    args=parser.parse_args()
    launcher.configure_output(args.directory)
    if Path(sys.executable).resolve()!=launcher.PYTHON.resolve():raise RuntimeError('Use pinned Windows Python')
    reference=launcher.read(args.reference);identity=reference['evidence']['header']['workload']
    if identity['kind']!='model_inference' or not identity['device_id'].startswith('GPU-'):
        raise RuntimeError('Native GPU reference required')
    manifest=dict(schema_version=2,commit=launcher.COMMIT,repository=str(launcher.REPO),
        source_status=launcher.git('status','--porcelain=v1','--untracked-files=all'),
        source_sha256=launcher.source_inventory(),python_version=sys.version,
        python_executable_sha256=launcher.sha(launcher.PYTHON),
        launcher_sha256=launcher.sha(launcher.__file__),
        packages={name:importlib.metadata.version(name) for name in ('torch','torchvision','numpy','Pillow')},
        gpu_uuid=identity['device_id'],model_sha256=identity['model_sha256'],
        reference_result=str(args.reference.resolve()),reference_sha256=launcher.sha(args.reference),
        factory='runtime_support:make_gpu',input_shape=[2,3,16,16],dtype='torch.float32',seed=0,
        protocol=launcher.fixed_protocol(),preparation_only=True,long_run_started=False)
    launcher.ROOT.mkdir(parents=False,exist_ok=False)
    launcher.publish_new(launcher.ROOT/'manifest.json',manifest)
    print(json.dumps(dict(status='MANIFEST_PREPARED',directory=str(launcher.ROOT),
        sources=len(manifest['source_sha256']),long_run_started=False)))


if __name__=='__main__':main()
