"""Analyze an explicitly supplied pseudonymous profile trace, without services."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from evaluation.profile_trace import analyze_trace, load_trace


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace",type=Path,required=True)
    parser.add_argument("--max-age-ms",type=int,nargs="*",default=[],help="explicit diagnostic ages only; no serving TTL/default is selected")
    args=parser.parse_args(argv)
    try:
        trace,digest=load_trace(args.trace)
        report=analyze_trace(trace,max_age_ms=args.max_age_ms)
        report["input_sha256"]=digest
        print(json.dumps(report,sort_keys=True,indent=2,allow_nan=False))
        return 0
    except (ValueError,OSError):
        print("PROFILE_TRACE_INVALID",file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
