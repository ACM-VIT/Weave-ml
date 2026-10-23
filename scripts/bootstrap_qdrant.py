"""Explicit production bootstrap for the V2 Qdrant collection contracts."""

from __future__ import annotations

import json
import os
import re

from qdrant_client import QdrantClient

from config import QDRANT_API_KEY, QDRANT_COLLECTION_NAME, QDRANT_URL, QDRANT_VECTOR_NAME
from embedding.corpus_revision import authority_from_environment
from embedding.qdrant_store import QdrantRepositoryStore
from embedding.user_profile_store import QdrantUserProfileStore


MINIMUM_QDRANT_VERSION = (1, 18, 0)


def _version(value: object) -> tuple[int, int, int]:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:[-+].*)?", str(value))
    if match is None:
        raise RuntimeError("Qdrant server returned an invalid version")
    return tuple(int(part) for part in match.groups())


def bootstrap() -> dict[str, object]:
    timeout = float(os.getenv("QDRANT_TIMEOUT_SECONDS", "10"))
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=timeout,
    )
    authority = None
    try:
        authority = authority_from_environment(
            collection=QDRANT_COLLECTION_NAME, vector=QDRANT_VECTOR_NAME, endpoint=QDRANT_URL,
        )
        version_info = client.info()
        server_version = str(getattr(version_info, "version", ""))
        if _version(server_version) < MINIMUM_QDRANT_VERSION:
            raise RuntimeError(
                "Qdrant server 1.18.0 or newer is required before collection bootstrap"
            )
        repository = QdrantRepositoryStore(client=client, corpus_authority=authority)
        users = QdrantUserProfileStore(client=client, timeout=max(1, int(timeout)))
        repository.ensure_collection()
        # Never auto-create the user collection in production: a misspelled
        # collection name would otherwise produce an empty but schema-valid
        # collection and silently disable personalization.
        users.validate_collection()
        # Re-read after synchronous index creation and validate exactly what
        # serving health will consume.
        info = client.get_collection(repository.collection_name)
        from retrieval.v2_retriever import QdrantV2Retriever

        QdrantV2Retriever._validate_repository_payload_indexes(info)
        return {
            "status": "ready",
            "qdrant_server_version": server_version,
            "repository_collection": repository.collection_name,
            "user_collection": users.contract.collection_name,
        }
    finally:
        try:
            if authority is not None:
                authority.close()
        finally:
            client.close()


def main() -> int:
    print(json.dumps(bootstrap(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
