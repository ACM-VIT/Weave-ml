"""Offline fresh-process inference warmup costs; no API traffic or storage writes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

_MODULE_STARTED=time.perf_counter()
ROOT=Path(__file__).resolve().parents[1]


def validate_options(candidates,repeats,threads):
    if any(type(value) is not int for value in (candidates,repeats,threads)):
        raise ValueError('profile bounds must be integers')
    if not 1<=candidates<=2000 or not 2<=repeats<=100 or not 1<=threads<=8:
        raise ValueError('bounds: candidates1..2000, repeats2..100, CPU threads1..8')


def child_profile(mode,candidates,repeats,threads):
    validate_options(candidates,repeats,threads)
    if mode not in {'load_only_control','warmup'}: raise ValueError('invalid profile mode')
    # Scope is this disposable child. No production environment/config is edited.
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',
                      EMBEDDING_CPU_THREADS=str(threads),EMBEDDING_MAX_CONCURRENCY='1',CUDA_VISIBLE_DEVICES='')
    import inspect
    import resource
    import socket
    from unittest.mock import patch
    # Offline flags/local_files_only are the supported loader contract; also
    # reject any accidental socket connection rather than hiding a download.
    network_attempts=[]
    def no_network(*args,**kwargs):
        network_attempts.append(True)
        raise RuntimeError('network disabled in warmup profile')
    with patch.object(socket.socket,'connect',no_network),patch.object(socket.socket,'connect_ex',no_network),patch.object(socket,'create_connection',no_network):
        import numpy as np
        import torch
        from config import EMBEDDING_MODEL_REVISION,REPOSITORY_EMBEDDING_MODEL
        from embedding import runtime
        from inference.feature_spec import INPUT_DIM
        from inference.ranker_service import RankerService
        from scripts.benchmark_ranker_devices import ARTIFACTS,validate_artifacts,fixtures,summary

        torch.set_num_threads(threads)
        provenance=validate_artifacts(ARTIFACTS)
        setup_started=time.perf_counter()
        previous=getattr(runtime._warmup_loading,'local_files_only',False)
        runtime._warmup_loading.local_files_only=True
        try:
            model=runtime.shared_embedding_model()
            runtime.repository_embedding_pipeline()
            runtime.user_onboarding_pipeline()
        finally:
            runtime._warmup_loading.local_files_only=previous
        options=dict(model_path=str(ARTIFACTS/'heavy_ranker.pt'),
                     scaler_path=str(ARTIFACTS/'feature_scaler.json'),
                     manifest_path=str(ARTIFACTS/'model_manifest.json'),
                     expected_embedding_model=REPOSITORY_EMBEDDING_MODEL,
                     expected_embedding_model_revision=EMBEDDING_MODEL_REVISION,
                     require_production_manifest=True)
        if 'device' in inspect.signature(RankerService).parameters:
            ranker=RankerService(**options,device='cpu')
        else:
            # Compatibility with the pre-device-option constructor; observed
            # placement below must still be CPU. No GPU result is fabricated.
            with patch('torch.cuda.is_available',return_value=False): ranker=RankerService(**options)
        if not ranker.ready or ranker.device.type!='cpu': raise RuntimeError('validated CPU ranker unavailable')
        if str(model._model.device)!='cpu': raise RuntimeError('CPU encoder placement was not established')
        setup_ms=(time.perf_counter()-setup_started)*1000
        counts={'encode':0,'forward':0}
        original_encode=model._model.encode
        def counted(*args,**kwargs):
            counts['encode']+=1
            return original_encode(*args,**kwargs)
        def forward(*args): counts['forward']+=1
        model._model.encode=counted
        hook=ranker.model.register_forward_pre_hook(forward)
        try:
            started=time.perf_counter()
            if mode=='warmup':
                runtime.warm_embedding_runtime()
                ranker.warmup()
            warmup_ms=(time.perf_counter()-started)*1000
            before=dict(counts)
            for _ in range(3):
                runtime.embedding_runtime_status(); ranker.warmup_status()
                if mode=='warmup': model.warmup(); ranker.warmup()
            if counts!=before: raise AssertionError('readiness/repeated warmup repeated inference')
            expected=1 if mode=='warmup' else 0
            if counts!={'encode':expected,'forward':expected}: raise AssertionError('warmup did not execute exactly once')
            readiness={**runtime.embedding_runtime_status(),**ranker.warmup_status()}
        finally:
            model._model.encode=original_encode
            hook.remove()
        startup_work_ms=(time.perf_counter()-_MODULE_STARTED)*1000
        user,skills,repos=fixtures(candidates)
        def encode():
            result=np.asarray(model.encode(['Repository discovery for developer tools.'],
                convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False),dtype=np.float32)
            if result.shape!=(1,384) or not np.all(np.isfinite(result)): raise ValueError('invalid real encoder output')
            return result
        def score():
            result=ranker.score_batch_compact(user,skills,repos)
            if len(result)!=candidates or any(not np.isfinite(row['final_score']) for row in result):
                raise ValueError('invalid real ranker output')
            return result
        def timed(call):
            began=time.perf_counter(); result=call()
            return result,(time.perf_counter()-began)*1000
        first_embedding,first_encode_ms=timed(encode)
        first_scores,first_score_ms=timed(score)
        encode_samples=[];score_samples=[]
        for _ in range(repeats):
            _,duration=timed(encode);encode_samples.append(duration)
            _,duration=timed(score);score_samples.append(duration)
        if network_attempts: raise AssertionError('offline warmup attempted a socket connection')
        return {'mode':mode,'network_connection_attempts':len(network_attempts),'device':str(ranker.device),'threads':torch.get_num_threads(),
            'candidates':candidates,'encoder_device':str(model._model.device),'encoder_model':REPOSITORY_EMBEDDING_MODEL,
            'encoder_revision':EMBEDDING_MODEL_REVISION,'ranker_input_dim':INPUT_DIM,
            'artifact_provenance':provenance,'model_setup_ms':setup_ms,'warmup_ms':warmup_ms,
            'startup_work_ms':startup_work_ms,'first_encode_ms':first_encode_ms,
            'first_rank_ms':first_score_ms,'steady_encode':summary(encode_samples),
            'steady_rank':summary(score_samples),'warmup_execution_counts':counts,
            'warmup_readiness':readiness,
            'measurement_order':['first_encode','first_rank','alternating_steady_encode_rank'],
            'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'parity':{'embedding':first_embedding[0].tolist(),
                      'ids':[row['repo_id'] for row in first_scores],
                      'scores':[float(row['final_score']) for row in first_scores]}}


def source_hashes():
    paths=[Path(__file__).resolve(),ROOT/'embedding/runtime.py',ROOT/'inference/ranker_service.py',
           ROOT/'api/main.py',ROOT/'inference/feature_spec.py',ROOT/'inference/value_function.py',
           ROOT/'scripts/benchmark_ranker_devices.py']
    return {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates',type=int,default=150)
    parser.add_argument('--repeats',type=int,default=10)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--child',choices=['load_only_control','warmup'])
    args=parser.parse_args(argv)
    try: validate_options(args.candidates,args.repeats,args.threads)
    except ValueError as exc: parser.error(str(exc))
    if args.child:
        print(json.dumps(child_profile(args.child,args.candidates,args.repeats,args.threads)))
        return 0
    sources=source_hashes()
    rows=[]
    for mode in ['load_only_control','warmup']:
        began=time.perf_counter()
        result=subprocess.run([sys.executable,'-m','scripts.profile_inference_warmup','--child',mode,
            '--candidates',str(args.candidates),'--repeats',str(args.repeats),'--threads',str(args.threads)],
            cwd=ROOT,check=True,capture_output=True,text=True,timeout=180)
        row=json.loads(result.stdout)
        row['process_wall_ms']=(time.perf_counter()-began)*1000
        rows.append(row)
    import numpy as np
    left,right=[row.pop('parity') for row in rows]
    if left['ids']!=right['ids']: raise AssertionError('warmup changed ranked identities')
    errors={}
    for key in ['embedding','scores']:
        a,b=np.asarray(left[key]),np.asarray(right[key])
        errors[key]=float(np.max(np.abs(a-b)))
        if not np.allclose(a,b,rtol=1e-6,atol=1e-6): raise AssertionError('warmup changed inference values')
    if source_hashes()!=sources: raise RuntimeError('source changed during warmup profile; discard this comparison')
    print(json.dumps({'schema':'inference-warmup-profile-v1','source_sha256':sources,
        'scope':'Two fresh sequential CPU processes, cached pinned encoder and checked-in validated ranker; synthetic inputs. Load-only control is not a deployed production mode. Model/pipeline startup work excludes network services and full HTTP. No production speedup or quality measured.',
        'warm_vs_control_max_absolute_error':errors,'results':rows},indent=2,sort_keys=True))
    return 0


if __name__=='__main__': raise SystemExit(main())
