"""Replay frozen vectors through real retrieval, heavy ranking and assembly.

Exports measured local exact-search evidence, never production ANN latency or
model qualification. No remote service or encoder download is supported. The
snapshot input is limited to8MiB and the child to180seconds; failures preserve previous output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from evaluation.stage_runner import MAX_SNAPSHOT_BYTES, FrozenSnapshot, execute_snapshot
from evaluation.recommendation_benchmark import digest
from scripts.evaluate_recommendations import atomic_json, read_json


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--artifact-dir',type=Path,default=Path(__file__).resolve().parents[1]/'inference')
    parser.add_argument('--expected-snapshot-sha',help=argparse.SUPPRESS)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    try:
        inputs=[args.snapshot,*(args.artifact_dir/name for name in ('heavy_ranker.pt','feature_scaler.json','model_manifest.json'))]
        if any(path.resolve()==args.output.resolve() for path in inputs):
            raise ValueError('output must not overwrite frozen inputs/artifacts')
        snapshot=FrozenSnapshot.model_validate(read_json(args.snapshot,max_input_bytes=MAX_SNAPSHOT_BYTES))
        if args.expected_snapshot_sha is not None and args.expected_snapshot_sha!=digest(snapshot):
            raise ValueError('frozen snapshot changed before worker execution')
        if not args.worker:
            result=subprocess.run([sys.executable,'-m','scripts.run_recommendation_stages','--worker',
                '--snapshot',str(args.snapshot.resolve()),'--output',str(args.output.resolve()),
                '--artifact-dir',str(args.artifact_dir.resolve()),'--expected-snapshot-sha',digest(snapshot)],timeout=180,check=False,capture_output=True,text=True)
            if result.returncode:
                print(json.dumps({'passed':False,'error':'WorkerFailed'}));return 2
            print(result.stdout.strip());return 0
        import socket
        from unittest.mock import patch
        import torch
        from retrieval.runtime import shutdown_retrieval_runtime
        torch.set_num_threads(1)
        def blocked(*args,**kwargs):
            raise RuntimeError('network access is forbidden in frozen replay')
        try:
            with patch.object(socket.socket,'connect',blocked),patch.object(socket,'create_connection',blocked):
                exported=execute_snapshot(snapshot,args.artifact_dir)
            atomic_json(args.output,exported.model_dump(mode='json'))
        finally:
            shutdown_retrieval_runtime()
        print(json.dumps({'passed':True,'format':exported.format,'profiles':len(exported.run.outputs),
                          'scope':exported.execution_scope,'qualified':False},sort_keys=True));return 0
    except (ValueError,TypeError,OSError,RuntimeError,RecursionError,subprocess.TimeoutExpired):
        print(json.dumps({'passed':False,'error':'InvalidOrFailedReplay'}));return 2


if __name__=='__main__':
    raise SystemExit(main())
