import tarfile
import zipfile
from pathlib import Path

from scripts.verify_release_artifacts import scan_path, verify_release_artifacts


def test_release_artifact_scan_accepts_code_metadata_and_public_examples(tmp_path: Path):
    artifact = tmp_path / "casmsocial.whl"
    with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr("casmsocial/model.py", "")
        archive.writestr("casmsocial/datasets/assets/source_inventory.yaml", "schema_version: 1\n")
        archive.writestr("examples/mvp/road_builder_places.csv", "place_id,x,y\n")

    assert scan_path(artifact) == []


def test_release_artifact_scan_rejects_identifier_bearing_archive_members(tmp_path: Path):
    source = tmp_path / "persons.parquet"
    source.write_bytes(b"fixture")
    artifact = tmp_path / "casmsocial.tar.gz"
    with tarfile.open(artifact, "w:gz") as archive:
        archive.add(source, arcname="casmsocial/testdata/tables/persons.parquet")

    violations = scan_path(artifact)

    assert len(violations) == 1
    assert "testdata/tables/persons.parquet" in violations[0]


def test_release_artifact_scan_rejects_local_data_and_ducklake_directories(tmp_path: Path):
    raw = tmp_path / "data" / "raw" / "atus" / "atusrost_2024.dat"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"fixture")
    metadata = tmp_path / "examples" / "mvp" / "mvp.ducklake" / "metadata.sqlite"
    metadata.parent.mkdir(parents=True)
    metadata.write_bytes(b"fixture")

    violations = verify_release_artifacts([tmp_path])

    assert len(violations) == 2
    assert any("data/raw" in violation for violation in violations)
    assert any("mvp.ducklake" in violation for violation in violations)


def test_release_artifact_scan_reports_missing_artifact(tmp_path: Path):
    assert scan_path(tmp_path / "missing.whl") == [f"{tmp_path / 'missing.whl'}: artifact does not exist"]
