import json
from io import BytesIO
from pathlib import Path

import pytest

from casmsocial.datasets.colorado_front_range import list_profiles, load_profile
from casmsocial.datasets.colorado_front_range.sources import (
    SourceArtifact,
    artifact_path,
    download_artifact,
    get_source_artifact,
    inspect_artifact,
    load_source_inventory,
    record_artifact,
)


class Response(BytesIO):
    def __init__(self, content: bytes, *, status: int, url: str) -> None:
        super().__init__(content)
        self.status = status
        self._url = url

    def geturl(self) -> str:
        return self._url


def _artifact(content: bytes, *, access: str = "download") -> SourceArtifact:
    import hashlib

    return SourceArtifact.model_validate({
        "artifact_id": "test-source",
        "source_id": "test-source-v1",
        "description": "test",
        "destination": "raw/test/source.bin",
        "access": access,
        "url": "https://example.test/source.bin" if access == "download" else None,
        "source_page": "https://example.test/" if access == "manual" else None,
        "verification": "pinned_sha256",
        "expected_size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    })


def test_inventory_covers_all_profile_sources_and_manual_atus_files():
    inventory = load_source_inventory()
    assert len(inventory.artifacts) == 7
    assert {artifact.source_id for artifact in inventory.artifacts} == {
        "osf-fpnc2-colorado-2024-09",
        "osf-ts9mg-colorado-education-sites-2024-10",
        "census-tiger-line-counties-2023",
        "bls-atus-2024",
        "geofabrik-colorado-osm-pbf-pinned",
    }
    assert {artifact.artifact_id for artifact in inventory.artifacts if artifact.access == "manual"} == {
        "bls-atus-2024-respondents",
        "bls-atus-2024-activities",
        "bls-atus-2024-roster",
    }
    profile_source_ids = {
        source_id
        for profile_name in list_profiles()
        for source_id in load_profile(profile_name).sources.model_dump().values()
    }
    assert profile_source_ids == {artifact.source_id for artifact in inventory.artifacts}


def test_pinned_artifact_is_verified_and_mismatch_is_reported(tmp_path: Path):
    content = b"verified source"
    artifact = _artifact(content)
    destination = artifact_path(tmp_path, artifact)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(content)
    assert inspect_artifact(artifact, tmp_path)["status"] == "verified"

    destination.write_bytes(b"wrong")
    assert inspect_artifact(artifact, tmp_path)["status"] == "mismatch"
    with pytest.raises(ValueError, match="failed verification"):
        record_artifact(artifact, tmp_path)


def test_download_is_atomic_and_records_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    content = b"complete download"
    artifact = _artifact(content)

    def open_source(request, timeout):
        assert request.full_url == artifact.url
        assert timeout == 120
        return Response(content, status=200, url="https://cdn.example.test/source.bin")

    monkeypatch.setattr("casmsocial.datasets.colorado_front_range.sources.urlopen", open_source)
    status = download_artifact(artifact, tmp_path)
    destination = artifact_path(tmp_path, artifact)
    provenance = json.loads(destination.with_name("source.bin.provenance.json").read_text())

    assert status["status"] == "verified"
    assert destination.read_bytes() == content
    assert not destination.with_suffix(".bin.part").exists()
    assert provenance["resolved_url"] == "https://cdn.example.test/source.bin"
    assert provenance["sha256"] == artifact.sha256


def test_download_resumes_partial_content_when_server_supports_range(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    content = b"0123456789"
    artifact = _artifact(content)
    destination = artifact_path(tmp_path, artifact)
    destination.parent.mkdir(parents=True)
    partial = destination.with_suffix(".bin.part")
    partial.write_bytes(content[:4])

    def open_source(request, timeout):
        assert request.get_header("Range") == "bytes=4-"
        return Response(content[4:], status=206, url=artifact.url or "")

    monkeypatch.setattr("casmsocial.datasets.colorado_front_range.sources.urlopen", open_source)
    assert download_artifact(artifact, tmp_path)["status"] == "verified"
    assert destination.read_bytes() == content


def test_unpinned_download_discards_stale_partial_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    artifact = SourceArtifact.model_validate({
        "artifact_id": "mutable-source",
        "source_id": "mutable-source-v1",
        "description": "mutable",
        "destination": "raw/test/mutable.bin",
        "access": "download",
        "url": "https://example.test/latest.bin",
        "verification": "record_sha256",
    })
    destination = artifact_path(tmp_path, artifact)
    destination.parent.mkdir(parents=True)
    destination.with_suffix(".bin.part").write_bytes(b"old version")

    def open_source(request, timeout):
        assert request.get_header("Range") is None
        return Response(b"new version", status=200, url=artifact.url or "")

    monkeypatch.setattr("casmsocial.datasets.colorado_front_range.sources.urlopen", open_source)
    assert download_artifact(artifact, tmp_path)["status"] == "verified"
    assert destination.read_bytes() == b"new version"


def test_manual_artifact_must_be_staged_and_recorded(tmp_path: Path):
    artifact = get_source_artifact("bls-atus-2024-roster")
    with pytest.raises(ValueError, match="downloaded manually"):
        download_artifact(artifact, tmp_path)
    with pytest.raises(FileNotFoundError, match="not staged"):
        record_artifact(artifact, tmp_path)

    destination = artifact_path(tmp_path, artifact)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"local ATUS extract")
    assert inspect_artifact(artifact, tmp_path)["status"] == "unrecorded"
    record_artifact(artifact, tmp_path)
    assert inspect_artifact(artifact, tmp_path)["status"] == "verified"


def test_source_destination_cannot_escape_data_directory():
    with pytest.raises(ValueError, match="safe relative path"):
        SourceArtifact.model_validate({
            "artifact_id": "unsafe",
            "source_id": "unsafe",
            "description": "unsafe",
            "destination": "../outside",
            "access": "manual",
            "source_page": "https://example.test/",
            "verification": "record_sha256",
        })
