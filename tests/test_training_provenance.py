"""Exact newly trained data/code identities; legacy manifests remain readable."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pytest
from Ranking_Training_module import train_ranker as trainer
from tests.test_ranker_preprocessing import write_training_rows
from inference.feature_spec import FEATURE_COUNT


def test_data_identity_binds_values_dtype_shape_and_not_zip_or_layout():
    arrays={key:np.arange(6,dtype=np.float32).reshape(2,3) for key in trainer.DATA_KEYS}
    digest=trainer.training_array_sha256(arrays)
    assert trainer.training_array_sha256(dict(reversed(list(arrays.items()))))==digest
    assert trainer.training_array_sha256({k:np.asfortranarray(v) for k,v in arrays.items()})==digest
    for change in ('value','dtype','shape'):
        changed={k:v.copy() for k,v in arrays.items()}
        if change=='value':changed['y_ctr'][0,0]+=1
        elif change=='dtype':changed['y_ctr']=changed['y_ctr'].astype(np.float64)
        else:changed['y_ctr']=changed['y_ctr'].reshape(3,2)
        assert trainer.training_array_sha256(changed)!=digest
    arrays['y_ctr']=np.array([object()],dtype=object)
    with pytest.raises(ValueError,match='object arrays'):trainer.training_array_sha256(arrays)


def test_local_code_identity_records_exact_commit_dirty_and_content(monkeypatch,tmp_path):
    (tmp_path/'fixture.py').write_text('fixture\n');monkeypatch.setattr(trainer,'REPO_ROOT',tmp_path)
    calls=[]
    def git(command,**kwargs):
        calls.append((command,kwargs))
        if 'rev-parse' in command:return b'a'*40+b'\n'
        if 'ls-files' in command:return b'fixture.py\0'
        if 'status' in command:return b' M fixture.py\n'
        pytest.fail('unexpected Git command')
    monkeypatch.setattr(subprocess,'check_output',git)
    identity=trainer.training_source_identity()
    assert identity=={'code_commit':'a'*40,'source_tree_dirty':True,'source_sha256':{'fixture.py':hashlib.sha256(b'fixture\n').hexdigest()}}
    assert all(command[1]=='--no-lazy-fetch' and kwargs['timeout']==5 and kwargs['env']['GIT_NO_LAZY_FETCH']=='1' for command,kwargs in calls)


def test_unsupported_git_and_symbolic_revision_fail_closed(monkeypatch,tmp_path):
    monkeypatch.setattr(trainer,'REPO_ROOT',tmp_path)
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**kw:(_ for _ in ()).throw(subprocess.CalledProcessError(129,['git'])))
    with pytest.raises(ValueError,match='compatible local-only Git'):trainer.training_source_identity()
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**kw:b'main\n')
    with pytest.raises(ValueError,match='exact Git commit'):trainer.training_source_identity()


def test_training_source_drift_fails_before_any_artifact_publication(monkeypatch,tmp_path):
    identities=iter([{'code_commit':'a'*40},{'code_commit':'b'*40}])
    monkeypatch.setattr(trainer,'training_source_identity',lambda:next(identities))
    data=tmp_path/'training.npz';write_training_rows(data,np.zeros((5,FEATURE_COUNT),np.float32));output=tmp_path/'output'
    with pytest.raises(ValueError,match='source changed'):
        trainer.train(data_path=data,epochs=1,lr=.001,batch_size=4,output_dir=output)
    assert not output.exists()


def test_hashing_contiguous_training_arrays_does_not_copy_full_buffer():
    import tracemalloc
    arrays={key:np.zeros((1,),dtype=np.float32) for key in trainer.DATA_KEYS}
    arrays['user_embs']=np.zeros((8192,384),dtype=np.float32)
    tracemalloc.start()
    try:
        trainer.training_array_sha256(arrays)
        _, peak=tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1024*1024
