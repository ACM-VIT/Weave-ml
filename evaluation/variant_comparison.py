"""Same-input baseline/treatment diagnostics; no promotion or privacy proof."""
from __future__ import annotations
from collections import Counter
import json
from typing import Annotated, Literal
from pydantic import Field, model_validator
from evaluation.recommendation_benchmark import StrictModel,digest,evaluate
from evaluation.stage_runner import MAX_SNAPSHOT_BYTES,FrozenSnapshot,StageRunExport,execute_snapshot
from evaluation.variant_policy import FrozenTreatment,FrozenSocialEvidence,FrozenVariantContext


class VariantExperiment(StrictModel):
    schema_version: Literal[1]
    snapshot: FrozenSnapshot
    treatments: Annotated[list[FrozenTreatment],Field(min_length=2,max_length=4)]
    social_evidence: Annotated[list[FrozenSocialEvidence],Field(max_length=32)]
    k: Annotated[int,Field(ge=1,le=100)]

    @model_validator(mode='after')
    def complete(self):
        names=[row.name for row in self.treatments]
        if len(names)!=len(set(names)) or names[0]!='baseline' or self.k>self.snapshot.limit:
            raise ValueError('experiment requires unique treatments, baseline first, bounded common k')
        for treatment in self.treatments:
            FrozenVariantContext(treatment=treatment,social_evidence=self.social_evidence).validate_for(self.snapshot)
        return self


def _comparison_from_exports(experiment, exports):
    names=[row.name for row in experiment.treatments]
    if set(exports)!=set(names):
        raise ValueError('worker conditions differ from frozen treatments')
    identity=None
    results={}
    for treatment in experiment.treatments:
        exported=exports[treatment.name]
        if (exported.bundle_sha256!=digest(experiment.snapshot.bundle)
                or exported.snapshot_sha256!=digest(experiment.snapshot)
                or exported.artifacts!=experiment.snapshot.artifacts
                or exported.as_of!=experiment.snapshot.as_of or exported.seed!=experiment.snapshot.seed
                or exported.configuration.get('variant_context_sha256')!=digest(FrozenVariantContext(
                    treatment=treatment,social_evidence=experiment.social_evidence))):
            raise ValueError('worker condition source/input/time/seed binding mismatch')
        current=(exported.code_commit,exported.source_tree_dirty,exported.source_sha256,exported.artifacts)
        if identity is not None and current!=identity:
            raise ValueError('worker condition source identity mismatch')
        identity=current
        report=evaluate(experiment.snapshot.bundle,exported.run,k=experiment.k)
        evidence_share={}
        for stage in ('retrieval','ranking','assembly'):
            contributions=Counter();positions=0;all_contributions=Counter();all_positions=0
            for output in exported.run.outputs:
                for row in output.stages[stage]:
                    all_positions+=1
                    all_contributions.update(exported.source_scores[output.profile_id][row.repo_id].keys())
                for row in output.stages[stage][:experiment.k]:
                    positions+=1
                    contributions.update(exported.source_scores[output.profile_id][row.repo_id].keys())
            evidence_share[stage]={'positions':positions,
                'source_counts':dict(sorted(contributions.items())),
                'source_share':{name:count/positions for name,count in sorted(contributions.items())} if positions else {},
                'all_positions':all_positions,'all_source_counts':dict(sorted(all_contributions.items())),
                'all_source_share':{name:count/all_positions for name,count in sorted(all_contributions.items())} if all_positions else {}}
        results[treatment.name]={'execution':exported.model_dump(mode='json'),'report':report,
            'overlapping_source_presence':evidence_share}
    baseline=results['baseline']['report']
    deltas={}
    for name,result in results.items():
        deltas[name]={}
        for stage in ('retrieval','ranking','assembly'):
            actual=result['report']['stages'][stage]['metrics'];before=baseline['stages'][stage]['metrics']
            deltas[name][stage]={key:(actual[key]-before[key] if actual[key] is not None and before[key] is not None else None) for key in before}
    return {'format':'frozen-variant-diagnostic-v1','experiment_sha256':digest(experiment),
        'label_provenance':experiment.snapshot.bundle.provenance.model_dump(mode='json'),
        'scope':'offline local exact-search actual components; no real quality, backend privacy/window or production qualification proof',
        'variant_order':[row.name for row in experiment.treatments],'results':results,'delta_from_baseline':deltas,
        'source_presence_scope':'overlapping evidence channels; sums may exceed1; not causal attribution',
        'exposure_scope':'supplied frozen exclusions only; seven-day backend implementation and214 acceptance remain separate'}


def compare_experiment(experiment:VariantExperiment,artifact_dir):
    # Own every nested condition/source/snapshot value before invoking a stage.
    raw=json.dumps(experiment.model_dump(mode='json'),sort_keys=True,allow_nan=False,separators=(',',':')).encode()
    if len(raw)>MAX_SNAPSHOT_BYTES:
        raise ValueError('variant experiment exceeds offline byte budget')
    experiment=VariantExperiment.model_validate_json(raw)
    exports={treatment.name:execute_snapshot(experiment.snapshot,artifact_dir,
        treatment=treatment,social_evidence=experiment.social_evidence)
        for treatment in experiment.treatments}
    return _comparison_from_exports(experiment,exports)


def validate_comparison_result(value,experiment):
    """Rebuild all derived evidence from the bound measured stage exports."""
    if not isinstance(value,dict) or not isinstance(value.get('results'),dict):
        raise ValueError('worker returned malformed comparison')
    exports={}
    for name,item in value['results'].items():
        if not isinstance(item,dict) or set(item)!={'execution','report','overlapping_source_presence'}:
            raise ValueError('worker returned malformed condition')
        exports[name]=StageRunExport.model_validate(item['execution'])
    expected=_comparison_from_exports(experiment,exports)
    # Canonical JSON distinguishes bool/int and rejects nonfinite numbers too.
    def canonical(item):
        return json.dumps(item,sort_keys=True,allow_nan=False,separators=(',',':'))
    if canonical(value)!=canonical(expected):
        raise ValueError('worker comparison derived evidence differs from captured actual runs')
    return value
