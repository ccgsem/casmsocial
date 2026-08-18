import json
from pathlib import Path

import pytest
import yaml

from scripts.verify_python_sbom import verify_python_sbom

POLICY = (
    Path(__file__).parents[1]
    / "casmsocial"
    / "datasets"
    / "colorado_front_range"
    / "assets"
    / "dependency_license_policy.yaml"
)


def _write_policy(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "reviewed_spdx_identifiers": ["BSD-3-Clause", "MIT"],
                "license_name_aliases": {"BSD": "BSD-3-Clause"},
                "ambiguous_license_names": ["License :: OSI Approved :: BSD License"],
                "metadata_overrides": {"legacy": {"version": "1.0", "license": "BSD-3-Clause", "evidence": "LICENSE"}},
            }
        )
    )
    return path


def _write_sbom(path: Path, components: list[dict]) -> Path:
    path.write_text(
        json.dumps(
            {
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
            }
        )
    )
    return path


def test_python_sbom_accepts_reviewed_license_and_versioned_override(tmp_path: Path):
    policy = _write_policy(tmp_path / "policy.yaml")
    sbom = _write_sbom(
        tmp_path / "sbom.json",
        [
            {"name": "direct", "version": "2.0", "licenses": [{"license": {"id": "MIT"}}]},
            {
                "name": "legacy",
                "version": "1.0",
                "licenses": [{"license": {"name": "License :: OSI Approved :: BSD License"}}],
            },
        ],
    )

    assert verify_python_sbom(sbom, policy) == {"components": 2, "metadata_overrides": 1, "violations": 0}


def test_python_sbom_rejects_missing_license_without_override(tmp_path: Path):
    policy = _write_policy(tmp_path / "policy.yaml")
    sbom = _write_sbom(tmp_path / "sbom.json", [{"name": "unknown", "version": "1.0"}])

    with pytest.raises(ValueError, match="missing license declaration"):
        verify_python_sbom(sbom, policy)


def test_python_sbom_rejects_ambiguous_license_name_without_override(tmp_path: Path):
    policy = _write_policy(tmp_path / "policy.yaml")
    sbom = _write_sbom(
        tmp_path / "sbom.json",
        [
            {
                "name": "unknown",
                "version": "1.0",
                "licenses": [{"license": {"name": "License :: OSI Approved :: BSD License"}}],
            }
        ],
    )

    with pytest.raises(ValueError, match="reviewed override"):
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


def test_python_sbom_rejects_override_without_evidence(tmp_path: Path):
    policy = _write_policy(tmp_path / "policy.yaml")
    policy_document = yaml.safe_load(policy.read_text())
    policy_document["metadata_overrides"]["legacy"].pop("evidence")
    policy.write_text(yaml.safe_dump(policy_document))
    sbom = _write_sbom(tmp_path / "sbom.json", [{"name": "legacy", "version": "1.0"}])

    with pytest.raises(ValueError, match="missing license evidence"):
        verify_python_sbom(sbom, policy)


def test_python_sbom_rejects_alias_for_ambiguous_license_name(tmp_path: Path):
    policy = _write_policy(tmp_path / "policy.yaml")
    policy_document = yaml.safe_load(policy.read_text())
    policy_document["license_name_aliases"]["License :: OSI Approved :: BSD License"] = "BSD-3-Clause"
    policy.write_text(yaml.safe_dump(policy_document))
    sbom = _write_sbom(tmp_path / "sbom.json", [])

    with pytest.raises(ValueError, match="cannot be normalized by alias"):
        verify_python_sbom(sbom, policy)


def test_python_sbom_requires_root_license_metadata(tmp_path: Path):
    policy = _write_policy(tmp_path / "policy.yaml")
    sbom = _write_sbom(tmp_path / "sbom.json", [])
    document = json.loads(sbom.read_text())
    document["metadata"]["component"].pop("licenses")
    sbom.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="CASMSocial license"):
        verify_python_sbom(sbom, policy)


def test_production_policy_does_not_guess_ambiguous_bsd_variant():
    policy = yaml.safe_load(POLICY.read_text())
    ambiguous = "License :: OSI Approved :: BSD License"

    assert ambiguous in policy["ambiguous_license_names"]
    assert ambiguous not in policy["license_name_aliases"]
    assert policy["metadata_overrides"]["numba"]["license"] == "BSD-2-Clause"
    for name in ("affine", "colorama", "Jinja2", "mpmath", "nodeenv", "pandas", "scipy", "shapely", "sympy"):
        override = policy["metadata_overrides"][name]
        assert override["license"] == "BSD-3-Clause"
        assert override["evidence"]

    assert "LGPL-2.1" in policy["metadata_overrides"]["shapely"]["evidence"]
