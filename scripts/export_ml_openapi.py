"""Export the registered V2 product contract without starting services/models."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

PRODUCT_PATHS = (
    '/api/v2/recommendations/generate', '/api/v2/feedback/batch',
    '/api/v2/repositories/embed', '/api/v2/repositories/refresh',
    '/api/v2/users/onboard', '/api/v2/search/repositories',
)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / 'contracts/ml-v2.openapi.json'


def build_schema() -> dict:
    # Importing app registers routes; it does not enter its startup lifespan.
    from api.main import app
    schema = deepcopy(app.openapi())
    schema['paths'] = {path: schema['paths'][path] for path in PRODUCT_PATHS}
    schema['info']['version'] = '2.0.0'
    schema['info']['x-product-schema-version'] = 2
    schema['info']['x-database-schema-owner'] = 'ACM-VIT/Weave/database/migrations'
    schema['info']['x-online-product-database-access'] = False
    schema['components']['securitySchemes'] = {
        'InternalSecret': {'type': 'apiKey', 'in': 'header', 'name': 'x-internal-secret'},
    }
    for path in PRODUCT_PATHS:
        schema['paths'][path]['post']['security'] = [{'InternalSecret': []}]
    return schema


def serialized_schema() -> str:
    return json.dumps(build_schema(), sort_keys=True, indent=2, ensure_ascii=False) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    rendered = serialized_schema()
    if args.check:
        if not args.output.exists() or args.output.read_text() != rendered:
            raise SystemExit('ML OpenAPI contract is stale; regenerate with python -m scripts.export_ml_openapi.')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == '__main__':
    main()
