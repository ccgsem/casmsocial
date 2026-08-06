"""Acquire and verify public inputs for the Colorado Front Range dataset."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.request import Request, urlopen

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

CHUNK_SIZE = 1024 * 1024
_PACKAGE = "casmsocial.datasets.colorado_front_range"


class SourceModel(BaseModel):
    """Strict base for source-acquisition contracts."""

    model_config = ConfigDict(extra="forbid")


class SourceArtifact(SourceModel):
    """One local input and its acquisition and verification policy."""

    artifact_id: str
    source_id: str
    description: str
    destination: str
    access: Literal["download", "manual"]
    verification: Literal["pinned_sha256", "record_sha256"]
    url: str | None = None
    source_page: str | None = None
    expected_size: int | None = None
    sha256: str | None = None

    @field_validator("destination")
    @classmethod
    def validate_destination(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("destination must be a safe relative path")
        return value

    @field_validator("url", "source_page")
    @classmethod
    def validate_https_url(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith("https://"):
            raise ValueError("source URLs must use HTTPS")
        return value

    @model_validator(mode="after")
    def validate_acquisition_policy(self) -> SourceArtifact:
        if self.access == "download" and not self.url:
            raise ValueError("download artifacts require a URL")
        if self.access == "manual" and not self.source_page:
            raise ValueError("manual artifacts require a source page")
        if self.verification == "pinned_sha256":
            if self.expected_size is None or self.sha256 is None:
                raise ValueError("pinned artifacts require expected_size and sha256")
        if self.sha256 is not None:
            try:
                valid_digest = len(self.sha256) == 64 and int(self.sha256, 16) >= 0
            except ValueError:
                valid_digest = False
            if not valid_digest:
                raise ValueError("sha256 must be a 64-character hexadecimal digest")
        return self


class SourceInventory(SourceModel):
    """Versioned collection of required public-source artifacts."""

    schema_version: Literal[1]
    dataset_id: str
    artifacts: list[SourceArtifact]

    @model_validator(mode="after")
    def validate_unique_artifacts(self) -> SourceInventory:
        artifact_ids = [artifact.artifact_id for artifact in self.artifacts]
        destinations = [artifact.destination for artifact in self.artifacts]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("artifact_id values must be unique")
        if len(destinations) != len(set(destinations)):
            raise ValueError("artifact destinations must be unique")
        return self


def load_source_inventory() -> SourceInventory:
    """Load the bundled, versioned source-acquisition inventory."""
    resource = files(_PACKAGE).joinpath("assets", "source_inventory.yaml")
    content = yaml.safe_load(resource.read_text(encoding="utf-8"))
    return SourceInventory.model_validate(content)


def get_source_artifact(artifact_id: str) -> SourceArtifact:
    """Return one artifact contract by stable identifier."""
    inventory = load_source_inventory()
    for artifact in inventory.artifacts:
        if artifact.artifact_id == artifact_id:
            return artifact
    choices = ", ".join(sorted(item.artifact_id for item in inventory.artifacts))
    raise ValueError(f"Unknown Colorado source artifact {artifact_id!r}; choose from {choices}")


def sha256_file(path: Path) -> str:
    """Calculate a SHA-256 digest without loading the file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_path(data_dir: Path, artifact: SourceArtifact) -> Path:
    """Resolve an artifact path while preventing escape from the data directory."""
    root = data_dir.expanduser().resolve()
    destination = (root / artifact.destination).resolve()
    if not destination.is_relative_to(root):
        raise ValueError(f"Artifact destination escapes data directory: {artifact.destination}")
    return destination


def _provenance_path(destination: Path) -> Path:
    return destination.with_name(f"{destination.name}.provenance.json")


def _read_provenance(destination: Path) -> dict[str, object] | None:
    path = _provenance_path(destination)
    if not path.is_file():
        return None
    content = json.loads(path.read_text(encoding="utf-8"))
    return content if isinstance(content, dict) else None


def inspect_artifact(artifact: SourceArtifact, data_dir: Path) -> dict[str, object]:
    """Hash an artifact and report whether it satisfies its verification policy."""
    destination = artifact_path(data_dir, artifact)
    result: dict[str, object] = {
        "artifact_id": artifact.artifact_id,
        "destination": str(destination),
        "status": "missing",
    }
    if not destination.is_file():
        return result

    size = destination.stat().st_size
    digest = sha256_file(destination)
    result.update(size_bytes=size, sha256=digest)
    if artifact.verification == "pinned_sha256":
        result["status"] = "verified" if size == artifact.expected_size and digest == artifact.sha256 else "mismatch"
        return result

    provenance = _read_provenance(destination)
    if provenance and provenance.get("size_bytes") == size and provenance.get("sha256") == digest:
        result["status"] = "verified"
    else:
        result["status"] = "unrecorded"
    return result


def _write_provenance(
    artifact: SourceArtifact,
    destination: Path,
    *,
    acquisition: Literal["download", "manual"],
    resolved_url: str | None,
) -> dict[str, object]:
    provenance: dict[str, object] = {
        "schema_version": 1,
        "artifact_id": artifact.artifact_id,
        "source_id": artifact.source_id,
        "acquisition": acquisition,
        "requested_url": artifact.url,
        "source_page": artifact.source_page,
        "resolved_url": resolved_url,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "size_bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }
    path = _provenance_path(destination)
    temporary = path.with_suffix(f"{path.suffix}.part")
    temporary.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return provenance


def record_artifact(artifact: SourceArtifact, data_dir: Path) -> dict[str, object]:
    """Register a manually staged or otherwise unpinned local artifact."""
    destination = artifact_path(data_dir, artifact)
    if not destination.is_file():
        raise FileNotFoundError(f"Source artifact is not staged: {destination}")
    status = inspect_artifact(artifact, data_dir)
    if artifact.verification == "pinned_sha256" and status["status"] != "verified":
        raise ValueError(f"Pinned source artifact failed verification: {artifact.artifact_id}")
    return _write_provenance(artifact, destination, acquisition="manual", resolved_url=None)


def download_artifact(
    artifact: SourceArtifact,
    data_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Download one source atomically, resuming a retained partial response when supported."""
    if artifact.access != "download" or artifact.url is None:
        raise ValueError(f"{artifact.artifact_id} must be downloaded manually from {artifact.source_page}")

    destination = artifact_path(data_dir, artifact)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        status = inspect_artifact(artifact, data_dir)
        if status["status"] == "verified":
            return status
        raise FileExistsError(f"{destination} exists but is not verified; use --overwrite or record it")

    temporary = destination.with_suffix(f"{destination.suffix}.part")
    if overwrite:
        temporary.unlink(missing_ok=True)
    elif temporary.exists() and artifact.verification != "pinned_sha256":
        # A mutable source could have changed since the partial response was
        # created. Without a pinned digest, appending would risk combining two
        # source versions and then blessing the result with a new checksum.
        temporary.unlink()
    offset = temporary.stat().st_size if temporary.exists() else 0
    headers = {"User-Agent": "casmsocial-data/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(artifact.url, headers=headers)

    with urlopen(request, timeout=120) as response:  # noqa: S310 - URL comes from the bundled inventory.
        resumed = offset > 0 and getattr(response, "status", None) == 206
        mode = "ab" if resumed else "wb"
        with temporary.open(mode) as target:
            while chunk := response.read(CHUNK_SIZE):
                target.write(chunk)
        resolved_url = response.geturl() if hasattr(response, "geturl") else artifact.url

    size = temporary.stat().st_size
    digest = sha256_file(temporary)
    if artifact.expected_size is not None and size != artifact.expected_size:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Downloaded {artifact.artifact_id} has size {size}, expected {artifact.expected_size}")
    if artifact.sha256 is not None and digest != artifact.sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Downloaded {artifact.artifact_id} failed SHA-256 verification")

    temporary.replace(destination)
    _write_provenance(artifact, destination, acquisition="download", resolved_url=resolved_url)
    return inspect_artifact(artifact, data_dir)
