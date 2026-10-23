"""Offline diagnostics for the *proposed* ranking-targets-v1 contract.

Never imported by serving; consumes an owner-reviewed, point-in-time export.
Coverage/provenance assertions are supplied by the exporter, not proven here.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
from statistics import mean

from inference.value_function import VALUE_WEIGHTS

ACTIONS = {"impression", "readme_open", "github_open", "save", "unsave", "dwell",
           "like", "unlike", "dislike", "undislike", "share"}


def target_dictionary():
    return json.loads((Path(__file__).resolve().parents[1] / "contracts" /
                       "ranking_targets_v1.json").read_text())


CONTRACT = target_dictionary()
VERSION = CONTRACT["version"]
KEY = tuple(CONTRACT["exposure"]["attribution_key"])
HEADS = {head: policy["output"] for head, policy in CONTRACT["heads"].items()}
WINDOW = timedelta(seconds=CONTRACT["exposure"]["window_seconds"])
DWELL_SCALE_MS = CONTRACT["heads"]["dwell"]["normalization_ms"]


def _time(value):
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("timestamps must be ISO8601 with timezone") from exc
    if result.tzinfo is None:
        raise ValueError("timestamps must include timezone")
    return result


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _number(value, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("numeric value required")
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError("numeric value outside finite bounds")
    return float(value)


def _key(row):
    for field in KEY[:-1]:
        _text(row.get(field), field)
    if type(row.get("position")) is not int or row["position"] < 0:
        raise ValueError("position must be a nonnegative integer")
    return tuple(row[field] for field in KEY)


def _unique(rows):
    """Replay exact duplicates once; reject conflicting identity globally."""
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("event collections must be arrays of objects")
    found = {}
    for row in rows:
        identity = _text(row.get("event_id"), "event_id")
        if identity in found and found[identity] != row:
            raise ValueError("conflicting event_id")
        found[identity] = row
    return list(found.values())


def _calibration(pairs, binary):
    if not pairs:
        return None
    bins = []
    for index in range(10):
        group = [(p, y) for p, y in pairs if min(int(p * 10), 9) == index]
        if group:
            bins.append({"lower": index / 10, "upper": (index + 1) / 10,
                         "count": len(group), "mean_prediction": mean(p for p, _ in group),
                         "mean_observed": mean(y for _, y in group)})
    result = {"count": len(pairs), "mean_prediction": mean(p for p, _ in pairs),
              "mean_observed": mean(y for _, y in pairs),
              "mse": mean((p - y) ** 2 for p, y in pairs), "bins": bins}
    if binary:
        result["brier"] = result["mse"]
        result["ece"] = sum(b["count"] * abs(b["mean_prediction"] - b["mean_observed"])
                            for b in bins) / len(pairs)
        result["log_loss"] = -mean(y * math.log(max(p, 1e-15)) +
                                  (1 - y) * math.log(max(1 - p, 1e-15)) for p, y in pairs)
        result["positives"] = sum(y for _, y in pairs)
    else:
        result["mae"] = mean(abs(p - y) for p, y in pairs)
    return result


def evaluate_targets(document):
    """Return aggregate diagnostics, excluding unknown/censored outcomes.

    This validates an offline interchange contract, not current wire payloads.
    No database access, model execution, training, or production mutation occurs.
    """
    if not isinstance(document, dict) or not isinstance(document.get("metadata"), dict):
        raise ValueError("dataset and metadata must be objects")
    metadata = document["metadata"]
    if metadata.get("target_version") != VERSION or metadata.get("data_kind") != "real_observed":
        raise ValueError("requires real_observed provenance and exact proposed target version")
    for field in ("dataset_id", "lineage", "consent_basis", "cohort", "model_version"):
        _text(metadata.get(field), field)
    cutoff = _time(metadata["observation_cutoff"])
    exposures = _unique(document["exposures"])
    events = _unique(document["events"])
    # Exposure IDs are impression event IDs: identity may not be reused by actions.
    if {e["event_id"] for e in exposures} & {e["event_id"] for e in events}:
        raise ValueError("event_id reused across exposure and action")
    grouped = defaultdict(list)
    for exposure in exposures:
        grouped[_key(exposure)].append(exposure)
        occurred = _time(exposure["occurred_at"])
        if occurred > cutoff or _time(exposure["predicted_at"]) > occurred:
            raise ValueError("future exposure or prediction after exposure")
        if type(exposure.get("eligible")) is not bool:
            raise ValueError("eligible must be boolean")
        if exposure.get("initial_saved") is not None and type(exposure["initial_saved"]) is not bool:
            raise ValueError("initial_saved must be boolean or null")
        coverage = exposure["coverage"]
        if not isinstance(coverage, dict) or not isinstance(exposure["predictions"], dict):
            raise ValueError("coverage and predictions must be objects")
        if set(coverage) != set(HEADS) - {"follow"} or any(type(v) is not bool for v in coverage.values()):
            raise ValueError("coverage must explicitly describe each observable head")
        if set(exposure["predictions"]) != set(HEADS.values()):
            raise ValueError("all five prediction outputs required for value attribution")
        for value in exposure["predictions"].values():
            _number(value, 0, 1)
    actions = defaultdict(list)
    unmatched = 0
    for event in events:
        key = _key(event)
        if event.get("event_type") not in ACTIONS - {"impression"}:
            raise ValueError("unsupported event type; follow is not observable")
        occurred = _time(event["occurred_at"])
        if occurred > cutoff:
            raise ValueError("action after observation cutoff")
        if event["event_type"] == "dwell":
            if type(event.get("dwell_ms")) is not int:
                raise ValueError("dwell_ms must be integer")
            _number(event["dwell_ms"], 3000, 300000)
        elif event.get("dwell_ms") is not None:
            raise ValueError("only dwell may carry dwell_ms")
        if key not in grouped:
            unmatched += 1
        actions[key].append(event)

    states = {head: Counter() for head in HEADS}
    pairs = {head: [] for head in HEADS}
    contributions = {head: [] for head in HEADS}
    fractions = {head: [] for head in HEADS}
    totals = []
    zero_scores = 0
    ambiguous = 0
    for key, candidates in grouped.items():
        for exposure in candidates:
            labels = {}
            occurred = _time(exposure["occurred_at"])
            end = occurred + WINDOW
            relevant = [e for e in actions[key] if occurred <= _time(e["occurred_at"]) < end]
            before = any(_time(e["occurred_at"]) < occurred for e in actions[key])
            # Multiple impressions with the same serve/session/item lack an explicit
            # action->impression link; do not guess last-touch attribution.
            if len(candidates) > 1 or before:
                ambiguous += 1
            for head in HEADS:
                if head == "follow":
                    state = "unsupported"
                elif not exposure["eligible"]:
                    state = "ineligible"
                elif len(candidates) > 1 or before:
                    state = "ambiguous"
                elif end > cutoff:
                    state = "censored"
                elif not exposure["coverage"][head]:
                    state = "missing"
                else:
                    state = "observed"
                    if head == "ctr":
                        labels[head] = float(any(e["event_type"] in CONTRACT["heads"]["ctr"]["event_types"] for e in relevant))
                    elif head == "gh_open":
                        labels[head] = float(any(e["event_type"] in CONTRACT["heads"]["gh_open"]["event_types"] for e in relevant))
                    elif head == "save":
                        saves = [e for e in relevant if e["event_type"] in {"save", "unsave"}]
                        # UTC time alone cannot order contradictory simultaneous state.
                        by_time = defaultdict(set)
                        for event in saves:
                            by_time[_time(event["occurred_at"])].add(event["event_type"])
                        if exposure.get("initial_saved") is not False:
                            state = "missing"
                        elif any(len(v) > 1 for v in by_time.values()):
                            state = "ambiguous"
                        else:
                            saves.sort(key=lambda e: _time(e["occurred_at"]))
                            labels[head] = float(bool(saves) and saves[-1]["event_type"] == "save")
                    elif head == "dwell":
                        dwell = [e["dwell_ms"] for e in relevant if e["event_type"] == "dwell"]
                        # V2 thresholds suppress short dwell; no sample is not zero.
                        if not dwell:
                            state = "missing"
                        else:
                            labels[head] = max(dwell) / DWELL_SCALE_MS
                states[head][state] += 1
                if state == "observed":
                    pairs[head].append((exposure["predictions"][HEADS[head]], labels[head]))
            if exposure["eligible"]:
                values = {h: VALUE_WEIGHTS[h] * exposure["predictions"][field]
                          for h, field in HEADS.items()}
                total = sum(values.values())
                totals.append(total)
                if total == 0:
                    zero_scores += 1
                for head, value in values.items():
                    contributions[head].append(value)
                    if total:
                        fractions[head].append(value / total)

    total_mass = sum(totals)
    return {"target_version": VERSION, "status": "proposal_diagnostic_only",
            "dataset": {f: metadata[f] for f in ("dataset_id", "cohort", "model_version")},
            "provenance_limit": "Exporter assertions require owner audit; this tool cannot verify consent or authenticity.",
            "exposures": len(exposures), "unmatched_events": unmatched,
            "ambiguous_exposures": ambiguous, "value_rows": len(totals), "zero_score_rows": zero_scores,
            "heads": {h: {"states": dict(states[h]),
                           "calibration": _calibration(pairs[h], h != "dwell"),
                           "mean_value_contribution": mean(contributions[h]) if totals else None,
                           "aggregate_value_share": sum(contributions[h]) / total_mass if total_mass else None,
                           "mean_row_value_share": mean(fractions[h]) if fractions[h] else None}
                      for h in HEADS},
            "weights": dict(VALUE_WEIGHTS),
            "follow_zero_shadow": {"mean_score_before": mean(totals) if totals else None,
                                   "mean_score_after": mean(t - f for t, f in zip(totals, contributions["follow"])) if totals else None,
                                   "promotion_evidence": False},
            "limitations": ["Follow has no labels: calibration is unavailable, never a negative label.",
                            "CTR is proposed README-open per eligible impression, not generic click probability.",
                            "Dwell is observed max reported episode / 600 seconds, thresholded and capped at 300 seconds in v2.",
                            "Conditional cohort metrics do not remove exposure/position bias or qualify a model.",
                            "No real dataset, target approval, ranking quality or rollback experiment is bundled."]}
