"""Artificial contract examples; not model-quality or real calibration evidence."""
from copy import deepcopy
import json
import math
import subprocess
import sys

import pytest

from evaluation.ranking_targets import VERSION, evaluate_targets, target_dictionary
from inference.value_function import VALUE_WEIGHTS, compute_value_score


def dataset():
    return {"metadata": {"target_version": VERSION, "data_kind": "real_observed",
                         "dataset_id": "unit-fixture-only", "lineage": "artificial test data",
                         "consent_basis": "unit fixture; not users", "cohort": "fixture",
                         "model_version": "fixture-model", "observation_cutoff": "2026-10-23T00:00:00Z"},
            "exposures": [{"event_id": "impression-1", "user_id": "user-1", "session_id": "session-1",
                           "serve_id": "serve-1", "repo_id": "repo-1", "position": 0,
                           "occurred_at": "2026-10-20T00:00:00Z", "predicted_at": "2026-10-19T23:59:59Z",
                           "eligible": True, "initial_saved": False,
                           "coverage": {"ctr": True, "save": True, "gh_open": True, "dwell": True},
                           "predictions": {"p_ctr": 0.2, "p_save": 0.3, "p_gh": 0.4,
                                           "pred_dwell_fraction": 0.5, "p_follow": 0.1}}], "events": []}


def action(doc, kind, second=1, **extra):
    row = {k: doc["exposures"][0][k] for k in ("user_id", "session_id", "serve_id", "repo_id", "position")}
    row.update(event_id=f"event-{len(doc['events'])}", event_type=kind,
               occurred_at=f"2026-10-20T00:00:{second:02d}Z", **extra)
    doc["events"].append(row)
    return row


def test_reversals_duplicate_replays_and_value_contributions_match_serving():
    doc = dataset()
    action(doc, "readme_open")
    action(doc, "github_open")
    action(doc, "save", 2)
    action(doc, "unsave", 3)
    sample = action(doc, "dwell", 4, dwell_ms=300000)
    doc["events"].append(deepcopy(sample))
    doc["exposures"].append(deepcopy(doc["exposures"][0]))
    result = evaluate_targets(doc)
    assert result["exposures"] == 1
    heads = result["heads"]
    assert heads["ctr"]["calibration"]["brier"] == pytest.approx(0.64)
    assert heads["save"]["calibration"]["mean_observed"] == 0
    assert heads["gh_open"]["calibration"]["positives"] == 1
    assert heads["dwell"]["calibration"]["mae"] == 0
    assert heads["follow"]["calibration"] is None
    assert heads["follow"]["states"] == {"unsupported": 1}
    expected = compute_value_score(doc["exposures"][0]["predictions"])
    assert result["follow_zero_shadow"]["mean_score_before"] == pytest.approx(expected)
    assert result["follow_zero_shadow"]["mean_score_after"] == pytest.approx(expected - 2)
    assert heads["follow"]["aggregate_value_share"] == pytest.approx(2 / expected)
    assert heads["follow"]["aggregate_value_share"] != pytest.approx(20 / 28.1)
    assert sum(h["aggregate_value_share"] for h in heads.values()) == pytest.approx(1)


def test_empty_actions_only_mature_complete_binary_heads_get_zero():
    result = evaluate_targets(dataset())["heads"]
    for head in ("ctr", "save", "gh_open"):
        assert result[head]["calibration"]["mean_observed"] == 0
    assert result["dwell"]["states"] == {"missing": 1}
    assert result["dwell"]["calibration"] is None


@pytest.mark.parametrize("state,change", [
    ("censored", lambda d: d["metadata"].update(observation_cutoff="2026-10-20T12:00:00Z")),
    ("missing", lambda d: d["exposures"][0]["coverage"].update(ctr=False)),
    ("ineligible", lambda d: d["exposures"][0].update(eligible=False)),
    ("ambiguous", lambda d: d["exposures"].append(dict(d["exposures"][0], event_id="impression-2"))),
])
def test_unknown_outcomes_do_not_leak_early_positive_into_calibration(state, change):
    doc = dataset()
    action(doc, "readme_open")
    change(doc)
    report = evaluate_targets(doc)
    assert report["heads"]["ctr"]["calibration"] is None
    assert report["heads"]["ctr"]["states"] == {state: len(doc["exposures"])}


def test_boundaries_timezone_and_cross_identity_are_not_positive_labels():
    doc = dataset()
    # End boundary excluded despite offset spelling; same instant as UTC end.
    action(doc, "readme_open").update(occurred_at="2026-10-21T05:30:00+05:30")
    action(doc, "github_open", repo_id="other-repo")
    report = evaluate_targets(doc)
    assert report["heads"]["ctr"]["calibration"]["positives"] == 0
    assert report["heads"]["gh_open"]["calibration"]["positives"] == 0
    assert report["unmatched_events"] == 1
    action(doc, "share").update(occurred_at="2026-10-19T23:59:59Z")
    assert evaluate_targets(doc)["heads"]["ctr"]["states"] == {"ambiguous": 1}


@pytest.mark.parametrize("initial", [True, None])
def test_save_baseline_unknown_or_already_saved_excludes_labels(initial):
    doc = dataset()
    doc["exposures"][0]["initial_saved"] = initial
    action(doc, "save")
    assert evaluate_targets(doc)["heads"]["save"]["states"] == {"missing": 1}


def test_save_unsave_tie_is_ambiguous_and_later_resave_is_observed():
    doc = dataset()
    action(doc, "save", 1)
    action(doc, "unsave", 1)
    assert evaluate_targets(doc)["heads"]["save"]["states"] == {"ambiguous": 1}
    doc["events"][1]["occurred_at"] = "2026-10-20T00:00:02Z"
    action(doc, "save", 3)
    assert evaluate_targets(doc)["heads"]["save"]["calibration"]["positives"] == 1


@pytest.mark.parametrize("case", ["collision", "conflict", "follow", "future_prediction", "future_action",
                                 "nan", "boolean", "null_serve", "naive_time", "missing_head",
                                 "synthetic", "string_coverage", "bad_dwell"])
def test_malformed_or_untrustworthy_inputs_fail_closed(case):
    doc = dataset()
    row = action(doc, "readme_open")
    exposure = doc["exposures"][0]
    if case == "collision":
        row["event_id"] = exposure["event_id"]
    elif case == "conflict":
        doc["events"].append(dict(row, repo_id="different"))
    elif case == "follow":
        row["event_type"] = "follow"
    elif case == "future_prediction":
        exposure["predicted_at"] = "2026-10-20T00:00:01Z"
    elif case == "future_action":
        row["occurred_at"] = "2026-10-24T00:00:00Z"
    elif case == "nan":
        exposure["predictions"]["p_follow"] = math.nan
    elif case == "boolean":
        exposure["predictions"]["p_follow"] = True
    elif case == "null_serve":
        exposure["serve_id"] = None
    elif case == "naive_time":
        row["occurred_at"] = "2026-10-20T00:00:01"
    elif case == "missing_head":
        exposure["predictions"].pop("p_follow")
    elif case == "synthetic":
        doc["metadata"]["data_kind"] = "synthetic"
    elif case == "string_coverage":
        exposure["coverage"]["ctr"] = "true"
    elif case == "bad_dwell":
        row.update(event_type="dwell", dwell_ms=2999)
    with pytest.raises(ValueError):
        evaluate_targets(doc)


def test_zero_scores_empty_dataset_and_probability_endpoints_have_finite_reports():
    doc = dataset()
    doc["exposures"][0]["predictions"] = dict.fromkeys(doc["exposures"][0]["predictions"], 0)
    action(doc, "readme_open")
    report = evaluate_targets(doc)
    assert report["zero_score_rows"] == 1
    assert report["heads"]["follow"]["aggregate_value_share"] is None
    assert math.isfinite(report["heads"]["ctr"]["calibration"]["log_loss"])
    json.dumps(report, allow_nan=False)
    doc["exposures"][0]["predictions"]["p_ctr"] = 1
    endpoint = evaluate_targets(doc)["heads"]["ctr"]["calibration"]
    assert endpoint["brier"] == 0
    assert endpoint["bins"][0]["lower"] == 0.9
    doc["exposures"] = []
    report = evaluate_targets(doc)
    assert report["heads"]["ctr"]["calibration"] is None
    assert report["follow_zero_shadow"]["mean_score_before"] is None


def test_dictionary_and_cli_share_exact_version_weights_and_fail_safely(tmp_path):
    contract = target_dictionary()
    assert contract["version"] == VERSION
    assert contract["weights_snapshot"] == VALUE_WEIGHTS
    assert contract["heads"]["follow"]["event_types"] == []
    path = tmp_path / "observations.json"
    path.write_text(json.dumps(dataset()))
    run = subprocess.run([sys.executable, "-m", "scripts.evaluate_ranking_targets", str(path)],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["status"] == "proposal_diagnostic_only"
    path.write_text('{"secret-user-row": "bad"}')
    run = subprocess.run([sys.executable, "-m", "scripts.evaluate_ranking_targets", str(path)],
                         capture_output=True, text=True)
    assert run.returncode != 0
    assert "secret-user-row" not in run.stderr


@pytest.mark.parametrize("malformed", ["duplicate_coverage", "excessive_nesting"])
def test_cli_rejects_conflicting_or_deep_raw_json_without_output_or_traceback(tmp_path, malformed):
    path = tmp_path / "observations.json"
    if malformed == "duplicate_coverage":
        # Raw JSON is essential: a Python dict cannot represent repeated keys.
        raw = json.dumps(dataset()).replace('"coverage": {"ctr": true',
                                            '"coverage": {"ctr": false, "ctr": true')
    else:
        raw = '{"private-export-field":' + '[' * 20000 + '0' + ']' * 20000 + '}'
    path.write_text(raw)
    run = subprocess.run([sys.executable, "-m", "scripts.evaluate_ranking_targets", str(path)],
                         capture_output=True, text=True, timeout=10)
    assert run.returncode == 2
    assert run.stdout == ""
    assert "invalid target dataset" in run.stderr
    assert "Traceback" not in run.stderr
    assert "private-export-field" not in run.stderr
