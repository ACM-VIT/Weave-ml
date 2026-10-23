"""Inspect a managed corpus authority; recovery is dry-run unless explicitly executed.

No timeout can prove retirement of old writers or completion of server-side
Qdrant work. Recovery assertions must come from controlled maintenance evidence.
"""
from __future__ import annotations

import argparse
import json

from config import QDRANT_COLLECTION_NAME, QDRANT_VECTOR_NAME
from embedding.corpus_revision import MaintenanceProof, authority_from_environment


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['status','recover','invalidate'])
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--maintenance-proof-id')
    parser.add_argument('--old-writers-retired',action='store_true')
    parser.add_argument('--qdrant-operations-completed',action='store_true')
    parser.add_argument('--former-primaries-fenced',action='store_true')
    parser.add_argument('--admin-boundary-managed',action='store_true')
    args=parser.parse_args(argv)
    if args.action=='status' and args.execute:
        parser.error('status has no execution mutation')
    maintenance=None
    if args.action=='recover' and args.execute:
        try:
            maintenance=MaintenanceProof(args.maintenance_proof_id or '',
                old_writers_retired=args.old_writers_retired,
                qdrant_operations_completed=args.qdrant_operations_completed,
                former_primaries_fenced=args.former_primaries_fenced,
                admin_boundary_managed=args.admin_boundary_managed)
        except ValueError as exc:
            parser.error(str(exc))
    authority=authority_from_environment(collection=QDRANT_COLLECTION_NAME,vector=QDRANT_VECTOR_NAME)
    if authority is None:
        if args.execute:
            parser.error('corpus authority is disabled; nothing was changed')
        print(json.dumps({'enabled':False,'changed':False,'serving':'uncached'}))
        return 0
    if maintenance is not None:
        authority.bootstrap(maintenance)
    if args.action=='invalidate' and args.execute:
        authority.invalidate()
    print(json.dumps({'enabled':True,'changed':maintenance is not None or (args.action=='invalidate' and args.execute),
        'action':args.action,'dry_run':not args.execute,'authority':authority.status(),
        'prerequisite':'external single endpoint/fencing and managed admin access; no inferred timeout recovery'},sort_keys=True))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
