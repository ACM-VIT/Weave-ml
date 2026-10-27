"""Bind a same-run Docker save archive to its verified source and image config.

The Actions artifact is the transport trust boundary. This manifest detects
missing, corrupt or mismatched content; it is not a signature for outside use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tarfile
from pathlib import Path

ARCHIVE = Path("gh-social-ml.tar.gz")
MANIFEST = Path("deployment.release.json")
INPUTS = ("Dockerfile", ".dockerignore", "uv.lock", "pyproject.toml")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def image_identity(sha: str) -> str:
    """Read the config digest and baked identity without trusting archive paths."""
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        manifests = [member for member in archive if member.name == "manifest.json"]
        if len(manifests) != 1 or not manifests[0].isfile() or manifests[0].size > 65536:
            raise ValueError("Image archive must contain one bounded manifest")
        manifest = json.load(archive.extractfile(manifests[0]))
        if len(manifest) != 1 or manifest[0]["RepoTags"] != [f"gh-social-ml:{sha}"]:
            raise ValueError("Image archive tag does not match the verified commit")
        config_name = manifest[0]["Config"]
        configs = [member for member in archive.getmembers() if member.name == config_name]
        if len(configs) != 1 or not configs[0].isfile() or configs[0].size > 1048576:
            raise ValueError("Image archive must contain one bounded config")
        config_bytes = archive.extractfile(configs[0]).read()
        image_id = "sha256:" + hashlib.sha256(config_bytes).hexdigest()
        config = json.loads(config_bytes)["config"]
        if config.get("Labels", {}).get("org.opencontainers.image.revision") != sha:
            raise ValueError("Image revision label does not match the verified commit")
        environment = config.get("Env", [])
        for key in ("ML_RELEASE_ID", "BAKED_ML_RELEASE_ID"):
            if [entry for entry in environment if entry.startswith(key + "=")] != [f"{key}={sha}"]:
                raise ValueError(f"Baked {key} does not match the verified commit")
        dockerfile = Path("Dockerfile").read_text()
        for key in ("EMBEDDING_MODEL", "EMBEDDING_MODEL_REVISION"):
            values = set(re.findall(rf"^ARG {key}=(.+)$", dockerfile, flags=re.MULTILINE))
            if len(values) != 1:
                raise ValueError(f"Expected one pinned {key} build input")
            value = values.pop()
            for variable in (key, "BAKED_" + key):
                if [entry for entry in environment if entry.startswith(variable + "=")] != [f"{variable}={value}"]:
                    raise ValueError(f"Image {variable} differs from the pinned build input")
        return image_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("create", "verify"))
    parser.add_argument("--sha", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.sha) or git("rev-parse", "HEAD") != args.sha:
        raise ValueError("Release must match the exact checked-out commit")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Release source has modified tracked files")
    binding = {
        "schema": 1,
        "commit": args.sha,
        "tree": git("rev-parse", "HEAD^{tree}"),
        "inputs": {name: digest(Path(name)) for name in INPUTS},
        "archive_sha256": digest(ARCHIVE),
        "image_id": image_identity(args.sha),
    }
    if args.mode == "create":
        MANIFEST.write_text(json.dumps(binding, sort_keys=True) + "\n")
        Path("deployment.image-id").write_text(binding["image_id"] + "\n")
    else:
        if json.loads(MANIFEST.read_text()) != binding:
            raise ValueError("Release artifact binding is missing, corrupt or mismatched")
        if Path("deployment.image-id").read_text() != binding["image_id"] + "\n":
            raise ValueError("Image digest metadata does not match the bound archive")
    print(f"Verified {args.sha} -> {binding['image_id']}")


if __name__ == "__main__":
    main()
