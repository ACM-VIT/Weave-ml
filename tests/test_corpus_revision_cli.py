"""Dry-run and explicit proof boundaries for corpus authority maintenance."""
import json
from types import SimpleNamespace
import pytest


@pytest.mark.parametrize('action',['recover','invalidate'])
def test_recovery_cli_dry_run_never_bootstraps_even_a_ready_authority(monkeypatch,capsys,action):
    from scripts import corpus_revision
    calls=[]
    authority=SimpleNamespace(status=lambda: {'state':'READY','writers':0},
                              bootstrap=lambda proof: calls.append(proof), invalidate=lambda: calls.append("invalidate"))
    monkeypatch.setattr(corpus_revision,'authority_from_environment',lambda **kwargs: authority)
    assert corpus_revision.main([action])==0
    report=json.loads(capsys.readouterr().out)
    assert report['dry_run'] is True and report['changed'] is False and calls==[]



def test_recovery_cli_execution_requires_all_proofs_before_connecting(monkeypatch):
    from scripts import corpus_revision
    monkeypatch.setattr(corpus_revision,'authority_from_environment',lambda **kwargs: pytest.fail('must validate proof first'))
    with pytest.raises(SystemExit) as error:
        corpus_revision.main(['recover','--execute','--maintenance-proof-id','timeout'])
    assert error.value.code==2



@pytest.mark.parametrize('action',['recover','invalidate'])
def test_disabled_authority_cannot_report_executed_maintenance(monkeypatch,action):
    from scripts import corpus_revision
    monkeypatch.setattr(corpus_revision,'authority_from_environment',lambda **kwargs: None)
    args=[action,'--execute']
    if action=='recover':
        args+=['--maintenance-proof-id','isolated-proof','--old-writers-retired',
               '--qdrant-operations-completed','--former-primaries-fenced','--admin-boundary-managed']
    with pytest.raises(SystemExit) as error:
        corpus_revision.main(args)
    assert error.value.code==2
