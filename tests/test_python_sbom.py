import json
from pathlib import Path

import pytest
import yaml

from scripts.verify_python_sbom import verify_python_sbom


def _write_policy(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump({
            "reviewed_spdx_identifiers": ["BSD-3-Clause", "MIT"],
            "license_name_aliases": {"BSD": "BSD-3-Clause"},
            "metadata_overrides": {"legacy": {"version": "1.0", "license": "BSD-3-Clause", "evidence": "LICENSE"}},
        })
    )
    return path


def _write_sbom(path: Path, components: list[dict]) -> Path:
    path.write_text(
        json.dumps({
            "bomFormat": "CycloneDX",
            "specVersion": "1.6",
            "metadata": {
                "component": {
                    "name": "casmsocial",
                    "version": "1.0",
                    "licenses": [{"license": {"id": "MIT"}}],
                }
            },
            "components": components,
        })
    )
    return path


def test_python_sbom_accepts_reviewed_license_and_versioned_override(tmp_path: Path):
    policy = _write_policy(tmp_path / "policy.yaml")
    sbom = _write_sbom(
        tmp_path / "sbom.json",
        [
            {"name": "direct", "version": "2.0", "licenses": [{"license": {"id": "MIT"}}]},
            {"name": "legacy", "version": "1.0"},
        ],
    )

    assert verify_python_sbom(sbom, policy) == {"components": 2, "metadata_overrides": 1, "violations": 0}


def test_python_sbom_rejects_missing_license_without_override(tmp_path: Path):
    policy = _write_policy(tmp_path / "policy.yaml")
    sbom = _write_sbom(tmp_path / "sbom.json", [{"name": "unknown", "version": "1.0"}])

    with pytest.raises(ValueError, match="missing license declaration"):
        verify_python_sbom(sbom, policy)


def test_python_sbom_rejects_unreviewed_license(tmp_path: Path):
    policy = _write_policy(tmp_path / "policy.yaml")
    sbom = _write_sbom(
        tmp_path / "sbom.json",
        [{"name": "new-license", "version": "1.0", "licenses": [{"expression": "MIT OR GPL-3.0-only"}]}],
    )

    with pytest.raises(ValueError, match="GPL-3.0-only"):
        verify_python_sbom(sbom, policy)


def test_python_sbom_rejects_override_after_dependency_version_changes(tmp_path: Path):
    policy = _write_policy(tmp_path / "policy.yaml")
    sbom = _write_sbom(tmp_path / "sbom.json", [{"name": "legacy", "version": "2.0"}])

    with pytest.raises(ValueError, match="reviewed override"):
        verify_python_sbom(sbom, policy)


def test_python_sbom_requires_root_license_metadata(tmp_path: Path):
    policy = _write_policy(tmp_path / "policy.yaml")
    sbom = _write_sbom(tmp_path / "sbom.json", [])
    document = json.loads(sbom.read_text())
    document["metadata"]["component"].pop("licenses")
    sbom.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="CASMSocial license"):
        verify_python_sbom(sbom, policy)
