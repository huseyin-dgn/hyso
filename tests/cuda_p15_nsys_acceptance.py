"""Single small P15 workload for an external nsys CUDA/NVTX capture.

Run under nsys --trace=cuda-sw,nvtx --capture-range=cudaProfilerApi.
The existing comparison is unchanged; profiling timings are not speed evidence.
"""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sqlite',type=Path,help='analyze the saved capture only; no GPU execution')
    args=parser.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from hyso.performance import TransferProtocol,TransferResult,measure_transfers
    result=(TransferResult(json.loads((args.output/'transfer.json').read_text())) if args.sqlite else
        measure_transfers(protocol=TransferProtocol(shape=(4*1024**2,),
        pinned=True,non_blocking=True,probe_overlap=True,compare_execution=True,
        profile_capture=True,warmup_iterations=1,measure_iterations=2,repeats=1,
        timeout_seconds=90)))
    data=result.to_dict()
    assert data['actual']==dict(completed_copies=4,warmup_copies=2,validation_copies=4,pending_copies=0)
    assert all(r['equal'] and r['terminal_complete'] and r['host_pinned']
        and r['non_blocking_requested'] and r['transfer_bytes']==16777216
        for r in data['raw_evidence']['copies'])
    for pair in data['raw_evidence']['overlap_probe']['comparison']['pairs']:
        for mode in ('serial','concurrent'):
            assert all(pair[mode][k] for k in ('terminal_completed','round_trip_equal','compute_equal'))
    if args.sqlite:
        timeline=result.nsys_activity_timeline(args.sqlite)
        (args.output/'gpu-activity-timeline.json').write_text(json.dumps(timeline,indent=2)+'\n')
        for phase in timeline['phases']:
            assert phase['copy_active_ns']+phase['compute_active_ns']-phase['overlap_ns']==phase['busy_union_ns']
        print(json.dumps(dict(status='PASS',phases=[dict(pair=p['pair_index'],mode=p['mode'],
            overlap_ns=p['overlap_ns']) for p in timeline['phases']])))
    else:
        result.to_json(args.output/'transfer.json');result.to_csv(args.output/'transfer.csv')
    assert TransferResult(json.loads((args.output/'transfer.json').read_text())).to_dict()==data
    print(json.dumps(dict(status='PASS',result_id=data['result_id'],worker=data['raw_evidence']['worker'],
        trace_token=data['raw_evidence']['overlap_probe']['comparison']['trace_token'])))


if __name__=='__main__': main()
