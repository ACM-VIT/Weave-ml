"""Offline POSIX owned-server HNSW measurement; no service URL/credentials."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
import tempfile
import zipfile
import zlib

import numpy as np

from evaluation.embedding_registry import EmbeddingConfig, Registry, VectorBatch
from evaluation import embedding_registry as registry_module

ARCHIVE_LIMIT = 128 * 1024 * 1024
JSON_LIMIT = 32 * 1024 * 1024
METADATA_LIMIT = 1024 * 1024


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def encode_json(value) -> bytes:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def decode_json(raw: bytes):
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    except RecursionError as exc:
        raise ValueError("JSON nesting budget exceeded") from exc


def read_regular(path: Path, limit: int) -> bytes:
    """Read one bounded regular file without following its final symlink."""
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError("input is not a bounded regular file")
        with os.fdopen(fd, "rb", closefd=False) as source:
            raw = source.read(limit + 1)
        after = os.fstat(fd)
        if len(raw) > limit or len(raw) != before.st_size or (
            before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns
        ) != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError("input changed during bounded read")
        return raw
    finally:
        os.close(fd)


def read_json(path: Path):
    raw = read_regular(path, JSON_LIMIT)
    return decode_json(raw), digest(raw)


def _npy(raw: bytes, *, metadata: bool = False) -> np.ndarray:
    """Validate NPY header and exact size BEFORE constructing any array."""
    if len(raw) < 10 or raw[:6] != b"\x93NUMPY" or raw[6:8] not in {b"\x01\x00", b"\x02\x00", b"\x03\x00"}:
        raise ValueError("unsupported NPY header")
    start = 10 if raw[6] == 1 else 12
    if len(raw) < start:
        raise ValueError("truncated NPY length field")
    header_size = struct.unpack("<H" if start == 10 else "<I", raw[8:start])[0]
    if header_size > 16384 or start + header_size > len(raw):
        raise ValueError("oversized or truncated NPY header")
    try:
        header = ast.literal_eval(raw[start:start + header_size].decode("utf-8" if raw[6] == 3 else "latin1"))
    except (ValueError, SyntaxError) as exc:
        raise ValueError("invalid NPY header") from exc
    if not isinstance(header, dict) or set(header) != {"descr", "fortran_order", "shape"}:
        raise ValueError("invalid NPY header keys")
    if header["fortran_order"] is not False or not isinstance(header["shape"], tuple):
        raise ValueError("NPY requires C order and explicit shape")
    shape = header["shape"]
    dtype = np.dtype(header["descr"])
    if metadata:
        valid = shape == () and dtype.kind == "U" and 0 < dtype.itemsize <= METADATA_LIMIT * 4
    else:
        valid = (len(shape) == 2 and all(type(n) is int and n > 0 for n in shape)
                 and shape[0] <= 2000 and shape[1] <= 8192 and dtype.str == "<f4")
    if not valid or dtype.hasobject or dtype.fields is not None:
        raise ValueError("unsupported NPY dtype/shape")
    size = math.prod(shape) * dtype.itemsize
    offset = start + header_size
    if size > ARCHIVE_LIMIT or len(raw) - offset != size:
        raise ValueError("NPY declared shape does not match bounded data")
    return np.frombuffer(raw, dtype=dtype, offset=offset).reshape(shape)


@dataclass(frozen=True)
class Artifact:
    report: dict
    report_sha256: str
    artifact_sha256: str
    registry_sha256: str
    fixture_sha256: str
    configuration: EmbeddingConfig
    repository_ids: tuple[str, ...]
    documents: np.ndarray
    queries: np.ndarray
    report_path: str
    artifact_path: str


def load_artifact(report_path: Path, registry_path: Path, fixture_path: Path) -> Artifact:
    """Bind historical exports to their registry, fixture, roles and report."""
    report, report_hash = read_json(report_path)
    registry_raw, registry_hash = read_json(registry_path)
    fixture, fixture_hash = read_json(fixture_path)
    registry = Registry.model_validate(registry_raw)
    config = EmbeddingConfig.model_validate(report["configuration"])
    if registry.get(config.name) != config or report.get("schema_version") != 1:
        raise ValueError("encoder report configuration is not in supplied registry")
    if report.get("registry_sha256") != registry_hash or report.get("fixture_sha256") != fixture_hash:
        raise ValueError("encoder report input digests differ from supplied evidence")
    if report.get("embedding_version") != config.version:
        raise ValueError("encoder report embedding identity mismatch")
    from scripts.compare_embedding_models import validate_fixture
    repos, queries = validate_fixture(fixture)
    if report.get("fixture_version") != fixture["fixture_version"] or report.get("label_provenance") != fixture["provenance"]:
        raise ValueError("encoder report fixture/provenance mismatch")
    name = report["cost"]["shadow_artifact"]
    if not isinstance(name, str) or not name.endswith(".npz") or Path(name).name != name or name in {".", ".."}:
        raise ValueError("shadow artifact must be a sibling basename")
    artifact_path = report_path.parent / name
    raw = read_regular(artifact_path, ARCHIVE_LIMIT)
    if type(report["cost"].get("shadow_artifact_bytes")) is not int or report["cost"]["shadow_artifact_bytes"] != len(raw):
        raise ValueError("artifact size differs from encoder report")
    expected = {"documents.npy", "queries.npy", "metadata_json.npy"}
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = archive.infolist()
            if len(members) != 3 or {m.filename for m in members} != expected:
                raise ValueError("NPZ requires exactly three unique expected members")
            if sum(m.file_size for m in members) > ARCHIVE_LIMIT:
                raise ValueError("NPZ decompression budget exceeded")
            arrays = {}
            for member in members:
                if member.flag_bits & 1 or member.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                    raise ValueError("unsupported NPZ compression/encryption")
                if member.file_size > (METADATA_LIMIT * 4 + 16396 if member.filename == "metadata_json.npy" else ARCHIVE_LIMIT):
                    raise ValueError("NPZ member budget exceeded")
                with archive.open(member) as source:
                    expanded = source.read(member.file_size + 1)
                if len(expanded) != member.file_size:
                    raise ValueError("NPZ member expanded size mismatch")
                arrays[member.filename] = _npy(expanded, metadata=member.filename == "metadata_json.npy")
    except (zlib.error, struct.error, EOFError) as exc:
        raise ValueError("invalid compressed NPZ member") from exc
    metadata_raw = str(arrays["metadata_json.npy"].item()).encode()
    if len(metadata_raw) > METADATA_LIMIT:
        raise ValueError("metadata JSON budget exceeded")
    metadata = decode_json(metadata_raw)
    required_metadata = {"schema_version", "embedding_version", "configuration", "fixture_sha256", "repository_ids", "queries", "roles"}
    if set(metadata) != required_metadata or metadata["schema_version"] != 1:
        raise ValueError("unknown artifact metadata schema")
    if (metadata["embedding_version"] != config.version or metadata["configuration"] != config.model_dump()
            or metadata["fixture_sha256"] != fixture_hash or metadata["roles"] != ["query", "document"]):
        raise ValueError("artifact metadata/configuration binding mismatch")
    ids = tuple(repo["repo_id"] for repo in repos)
    if metadata["repository_ids"] != list(ids) or metadata["queries"] != [q["query"] for q in queries]:
        raise ValueError("artifact row order differs from canonical fixture IDs/queries")
    documents, query_rows = arrays["documents.npy"], arrays["queries.npy"]
    if documents.shape[0] != len(ids) or query_rows.shape[0] != len(queries) or query_rows.shape[0] > 1000:
        raise ValueError("artifact row count differs from fixture")
    counts = report.get("sample_counts", {})
    if counts.get("repositories") != len(ids) or counts.get("queries") != len(queries):
        raise ValueError("encoder report counts differ from artifact")
    if report["cost"].get("raw_repository_vector_bytes") != documents.nbytes:
        raise ValueError("encoder report vector byte count differs from artifact")
    VectorBatch.create(documents, config, role="document")
    VectorBatch.create(query_rows, config, role="query")
    # The original encoder schema did not record an NPZ SHA256. Report that
    # limitation rather than pretend today's digest was signed at encode time.
    expected_digest = report["cost"].get("shadow_artifact_sha256")
    if expected_digest is not None and expected_digest != digest(raw):
        raise ValueError("artifact digest differs from encoder report")
    return Artifact(report, report_hash, digest(raw), registry_hash, fixture_hash, config, ids,
                    documents, query_rows, str(report_path.resolve()), str(artifact_path.resolve()))


def publish_new_json(path: Path, value) -> None:
    """Atomic create-only publication; preserves an earlier complete report."""
    raw = encode_json(value) + b"\n"
    if len(raw) > JSON_LIMIT:
        raise ValueError("report exceeds byte budget")
    if path.exists() or path.is_symlink():
        raise ValueError("output already exists")
    fd, temporary = tempfile.mkstemp(prefix=".embedding-ann-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        # Same-filesystem link is atomic and refuses even a racing target.
        os.link(temporary, path, follow_symlinks=False)
    finally:
        os.unlink(temporary)
