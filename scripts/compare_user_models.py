"""Compare single-vector and long/session models using frozen production stages.

All conditions share vectors, artifacts, reference time and seed. Exact-search
and synthetic-label diagnostics do not prove real relevance, backend privacy,
recent-window correctness, ANN latency, or production qualification.
"""
from __future__ import annotations
import argparse,json,subprocess,sys,tempfile
from pathlib import Path
from evaluation.model_history import FrozenModelHistory,compare_model_history,validate_model_comparison
from evaluation.stage_runner import MAX_SNAPSHOT_BYTES,require_offline_git,OfflineGitUnavailable
from evaluation.recommendation_benchmark import digest
from scripts.evaluate_recommendations import atomic_json,MAX_INPUT_BYTES
from evaluation.embedding_artifacts import read_regular,decode_json


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--artifact-dir',type=Path,default=Path(__file__).resolve().parents[1]/'inference')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--expected-input-sha',help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    try:
        inputs=[args.experiment,*(args.artifact_dir/name for name in ('heavy_ranker.pt','feature_scaler.json','model_manifest.json'))]
        if any(p.resolve()==args.output.resolve() for p in inputs):raise ValueError('output overwrites input')
        experiment=FrozenModelHistory.model_validate(decode_json(read_regular(args.experiment,MAX_SNAPSHOT_BYTES)))
        if args.expected_input_sha is not None and args.expected_input_sha!=digest(experiment):raise ValueError('input drift')
        if not args.worker:
            require_offline_git()
            # Preserve169's owned-worker output boundary: nothing reaches the
            # final path until exit, cleanup and parent validation all succeed.
            with tempfile.TemporaryDirectory(prefix='weave-variant-result-') as directory:
                private_output=Path(directory)/'comparison.json'
                result=subprocess.run([sys.executable,'-m','scripts.compare_user_models','--worker',
                    '--experiment',str(args.experiment.resolve()),'--output',str(private_output),
                    '--artifact-dir',str(args.artifact_dir.resolve()),'--expected-input-sha',digest(experiment)],
                    timeout=180,check=False,capture_output=True,text=True)
                if result.returncode:
                    print(json.dumps({'passed':False,'error':'ModelHistoryWorkerFailed'}));return 2
                comparison=validate_model_comparison(decode_json(read_regular(private_output,MAX_INPUT_BYTES)),experiment)
            atomic_json(args.output,comparison)
            print(result.stdout.strip());return 0
        import socket,torch
        from unittest.mock import patch
        from retrieval.runtime import shutdown_retrieval_runtime
        torch.set_num_threads(1)
        def blocked(*args,**kwargs):raise RuntimeError('offline experiment forbids socket connections')
        try:
            with patch.object(socket.socket,'connect',blocked),patch.object(socket,'create_connection',blocked):
                comparison=compare_model_history(experiment,args.artifact_dir)
            atomic_json(args.output,comparison)
        finally:shutdown_retrieval_runtime()
        print(json.dumps({'passed':True,'conditions':list(comparison['results']),
                          'label_kind':comparison['label_provenance']['kind'],'qualified':False}));return 0
    except OfflineGitUnavailable as error:
        print(json.dumps({'passed':False,'error':'UnsupportedGit','detail':str(error)}));return 2
    except (ValueError,TypeError,OSError,RuntimeError,RecursionError,subprocess.TimeoutExpired):
        print(json.dumps({'passed':False,'error':'InvalidOrFailedModelReplay'}));return 2


if __name__=='__main__':raise SystemExit(main())
