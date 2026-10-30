"""Local recorded-video P30 audit. CPU only; no camera/network/GPU access.

Run from repo root: python tools/p30/acceptance.py --video ABSOLUTE_PATH --output NEW_DIRECTORY
The model is the existing untrained technical fixture, never a quality benchmark.
"""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def write_json(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def write_csv(path,rows):
    with path.open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def make_decode_probe(video,receipt):
    """Independent read-to-EOF inventory, executed inside the owned probe Job."""
    import cv2
    capture=cv2.VideoCapture(video);assert capture.isOpened()
    hashes=[];declared=capture.get(cv2.CAP_PROP_FRAME_COUNT)
    try:
        while True:
            ok,frame=capture.read()
            if not ok:break
            hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
    finally:capture.release()
    assert hashes
    write_json(Path(receipt),dict(decoded_to_EOF=True,actual_decoded_frames=len(hashes),
        declared_frames=declared,frame_sha256=hashes,file_sha256=digest(video),
        scope='independent_owned_OpenCV_file_decode_not_sensor_capture'))
    sys.path.insert(0,str(ROOT/'tests'))
    from runtime_support import make_frames
    return make_frames(delay=0)


def lifecycle(e,late_after):
    frames={};conservation=[];pending=set();processed=dropped=0
    for row in e['rows']:
        frame=row.get('frame_id');kind=row['event'];stamp=row['timestamp_ns']
        if kind=='received':
            assert frame not in frames
            pending.add(frame)
            frames[frame]=dict(frame_id=frame,clock_id=e['header']['clock_id'],
                read_start_ns=row['read_start_ns'],read_end_ns=row['read_end_ns'],
                frame_assigned_ns=row['frame_assigned_ns'],received_ns=stamp,
                frame_shape='x'.join(map(str,row['frame_shape'])),frame_nbytes=row['frame_nbytes'],
                content_sha256=row['content_sha256'],enqueued_ns=None,dequeued_ns=None,
                process_start_ns=None,preprocess_start_ns=None,preprocess_end_ns=None,
                inference_start_ns=None,inference_end_ns=None,postprocess_start_ns=None,postprocess_end_ns=None,
                process_end_ns=None,completion_ns=None,drop_ns=None,drop_reason=None,late=None,disposition='pending')
        elif kind in ('enqueued','dequeued'):frames[frame][kind+'_ns']=stamp
        elif kind in ('processed','dropped'):
            pending.remove(frame);f=frames[frame];f['disposition']=kind
            if kind=='processed':
                processed+=1;f.update(process_start_ns=row['process_start_ns'],process_end_ns=row['process_end_ns'],
                    completion_ns=stamp,late=(stamp-f['received_ns'])>late_after*1e9)
                for stage in row['stages']:
                    if stage['name'] in ('preprocess','inference','postprocess'):
                        f[stage['name']+'_start_ns']=stage['start_ns'];f[stage['name']+'_end_ns']=stage['end_ns']
            else:dropped+=1;f.update(drop_ns=stamp,drop_reason=row['reason'])
        assert len(frames)==processed+dropped+len(pending)
        conservation.append(dict(event_index=row['index'],timestamp_ns=stamp,event=kind,frame_id=frame,
            received=len(frames),processed=processed,dropped=dropped,pending=len(pending),conserved=True,queue_depth=row['queue_depth']))
    assert not pending
    assert set(frames)=={str(i) for i in range(len(frames))}
    return list(frames.values()),conservation


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--video',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    source_path=args.video.resolve(strict=True);out=args.output.resolve();out.mkdir(exist_ok=False)
    from hyso.performance import StreamSource,StreamProtocol,WorkloadRecipe,measure_streaming,RuntimeResult
    from hyso.performance.runtime_result import _result
    from hyso.performance.stability import _read_journal
    source=StreamSource('video_file',str(source_path),'existing_local_OpenCV_project_recorded_parking_video_origin_sensor_unverified')
    recipe=WorkloadRecipe('runtime_support:make_frames',dict(delay=.005),search_path=str(ROOT/'tests'))
    source_hash=digest(source_path)
    write_json(out/'source.json',dict(path=str(source_path),sha256=source_hash,bytes=source_path.stat().st_size,
        scope='existing_local_recorded_video_no_download_no_live_camera',
        model='seed30_untrained_three_channel_fixture_real_CPU_forward_not_MNIST_or_quality_acceptance'))
    inventory_recipe=WorkloadRecipe('acceptance:make_decode_probe',
        dict(video=str(source_path),receipt=str(out/'decode-inventory.json')),search_path=str(Path(__file__).parent))
    probe=measure_streaming(source,inventory_recipe,out/'probe',protocol=StreamProtocol(max_frames=1))
    probe.to_json(out/'probe.json');pd=probe.to_dict()
    assert pd['analysis']['status']=='completed',pd['analysis']
    inventory=json.loads((out/'decode-inventory.json').read_text(encoding='utf-8'))
    count=inventory['actual_decoded_frames'];assert count>0
    assert count==pd['evidence']['header']['capture']['declared_frames']==inventory['declared_frames']
    assert inventory['file_sha256']==source_hash
    summary={};previous_frames=None;previous_processor=None
    for policy in ('block','drop_newest'):
        # Both policies decode the entire same file. Plan deadlines are explicit,
        # generous relative to the controlled service wait, not hardware limits.
        protocol=StreamProtocol(max_frames=count,queue_capacity=1,queue_policy=policy,
            late_after_seconds=.01,max_runtime_seconds=max(30.,count*.025+10))
        write_json(out/(policy+'-protocol.json'),dict(source=asdict(source),recipe=asdict(recipe),protocol=asdict(protocol)))
        r=measure_streaming(source,recipe,out/policy,protocol=protocol);d=r.to_dict();a=d['analysis']
        c=d['evidence']['controller'];code=c['exit_code'] & 0xffffffff
        if code>=0x80000000:raise RuntimeError('Native crash: stop all subsequent P30 workloads; preserve evidence')
        r.to_json(out/(policy+'.json'));r.to_csv(out/(policy+'.csv'))
        assert a['status']=='completed',a
        assert d['evidence']['header']['capture']['file_sha256']==source_hash==digest(source_path)
        assert a['received']==count and a['pending']==0 and c['cleanup']['active_processes']==0
        assert RuntimeResult.from_json(out/(policy+'.json')).to_dict()==d
        assert RuntimeResult.from_csv(out/(policy+'.csv')).to_dict()==d
        assert _result('P30',_read_journal(out/policy)).to_dict()['analysis']==a
        frames,conservation=lifecycle(d['evidence'],protocol.late_after_seconds)
        write_csv(out/(policy+'-frames.csv'),frames);write_csv(out/(policy+'-conservation.csv'),conservation)
        hashes={f['frame_id']:f['content_sha256'] for f in frames}
        assert hashes=={str(i):sha for i,sha in enumerate(inventory['frame_sha256'])}
        if previous_frames is not None:assert hashes==previous_frames
        processor=d['evidence']['header']['processor']
        if previous_processor is not None:assert processor==previous_processor
        previous_processor=processor
        previous_frames=hashes
        summary[policy]=a
        print(policy,json.dumps(a),flush=True)
    write_json(out/'summary.json',dict(source_sha256=source_hash,decoded_frames_each_policy=count,
        identical_decoded_frame_hashes=True,identical_model_and_preprocessing=True,policies=summary,live_camera='NOT_RUN_requires_separate_explicit_consent',
        cuda='NOT_RUN_no_new_authorization_used',B4='OPEN',CPU65536='OPEN'))


if __name__=='__main__':main()
