"""Network-free CPU/provider/memory profile for issue #88.

python3 -m scripts.profile_summary_reuse --baseline-ref upsteam/main
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import time
import tracemalloc
import types

from summarization.pipeline import CardSummaryPipeline
from summarization.reuse import SummaryReuseCache
from summarization.validation import analyze_comparison_texts, parse_summary_response, validate_generated_summary


def cpu_seconds(action, repeats):
    started = time.process_time()
    for _ in range(repeats):
        action()
    return time.process_time() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", help="Git ref containing the original validator")
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 1000:
        parser.error("--repeats must be between 1 and 1000")
    fixture = json.loads(Path("tests/fixtures/repository_summary_golden.json").read_text())["repositories"][0]
    content = parse_summary_response(json.dumps({"summary": fixture["new_summary"],
                                                "highlights": ["Shared review tools"]}))
    readme = "# Overview\n" + "Independent modules exchange messages through stable runtime boundaries. " * 850
    texts = (readme,)
    analysis = analyze_comparison_texts(texts)
    report = {
        "repeats": args.repeats, "source_chars": len(readme),
        "validation_cpu_seconds": cpu_seconds(
            lambda: validate_generated_summary(content, source_analysis=analysis), args.repeats),
        "analysis_cpu_seconds": cpu_seconds(lambda: analyze_comparison_texts(texts), args.repeats),
    }
    if args.baseline_ref:
        source = subprocess.run(
            ["git", "show", f"{args.baseline_ref}:summarization/validation.py"],
            check=True, capture_output=True, text=True,
        ).stdout
        baseline = types.ModuleType("summarization._profile_baseline")
        exec(compile(source, "baseline_validation.py", "exec"), baseline.__dict__)
        report["baseline_validation_cpu_seconds"] = cpu_seconds(
            lambda: baseline.validate_generated_summary(content, comparison_texts=texts), args.repeats)

    class Provider:
        def generate(self, *args, **kwargs):
            return content.model_dump_json()

    repo = {"repo_id": "profile-88", "full_name": "weave/profile", "readme": readme,
            "description": "A collaborative tool for reviewing project changes."}
    for scenario in ("unchanged", "changed"):
        gc.collect()
        tracemalloc.start()
        pipeline = CardSummaryPipeline(provider=Provider(), reuse_cache=SummaryReuseCache(max_entries=4))
        started = time.process_time()
        for index in range(args.repeats):
            current = repo if scenario == "unchanged" else {**repo, "readme": readme + f"\nEvidence {index}."}
            assert pipeline.summarize(current, current).source == "generated"
        elapsed = time.process_time() - started
        retained, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        snapshot = pipeline.reuse_cache.snapshot()
        report[scenario] = {**snapshot, "cpu_seconds_with_tracemalloc": elapsed,
                            "traced_retained_bytes": retained, "traced_peak_bytes": peak,
                            "artifact_hit_ratio": snapshot["artifact_hits"] / args.repeats}
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
