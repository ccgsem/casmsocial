import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts.verify_container_sbom import verify_container_sbom

PRODUCTION_POLICY = (
    Path(__file__).parents[1]
    / "casmsocial"
    / "datasets"
    / "colorado_front_range"
    / "assets"
    / "container_dependency_license_policy.yaml"
)


def _write_evidence(root: Path, path: str, content: bytes = b"license") -> str:
    evidence = root / path.lstrip("/")
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _write_policy(path: Path, evidence_root: Path, *, status: str = "review_required") -> Path:
    digest = _write_evidence(evidence_root, "/licenses/legacy")
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "reviewed_override_licenses": ["MIT"],
                "forbidden_package_types": ["cargo"],
                "metadata_overrides": {
                    "legacy": {
                        "package_type": "pypi",
                        "version": "1.0",
                        "license": "MIT",
                        "source_url": "https://example.test/legacy/LICENSE",
                        "evidence": [{"path": "/licenses/legacy", "sha256": digest}],
                    }
                },
                "scanner_subject": {
                    "package_type": "oci",
                    "spdx_id_prefix": "SPDXRef-DocumentRoot-",
                    "required_count": 1,
                },
                "native_review": {
                    "package_type": "deb",
                    "require_license_metadata": True,
                    "status": status,
                    "review_trigger_patterns": ["GPL-", "LicenseRef-"],
                },
            }
        )
    )
    return path


def _package(name: str, version: str, package_type: str, license_declared: str | None = None) -> dict:
    package = {
        "SPDXID": f"SPDXRef-Package-{name}",
        "name": name,
        "versionInfo": version,
        "externalRefs": [
            {
                "referenceCategory": "PACKAGE-MANAGER",
                "referenceType": "purl",
                "referenceLocator": f"pkg:{package_type}/{name}@{version}",
            }
        ],
    }
    if license_declared is not None:
        package["licenseDeclared"] = license_declared
    return package


def _write_sbom(path: Path, packages: list[dict] | None = None) -> Path:
    document_root = _package("image", "sha256:123", "oci")
    document_root["SPDXID"] = "SPDXRef-DocumentRoot-Image"
    path.write_text(
        json.dumps(
            {
                "spdxVersion": "SPDX-2.3",
                "packages": (
                    packages
                    if packages is not None
                    else [
                        _package("legacy", "1.0", "pypi"),
                        _package("libc6", "1.0", "deb", "LGPL-2.1-only AND LicenseRef-notice"),
                        document_root,
                    ]
                ),
            }
        )
    )
    return path


def test_container_sbom_accepts_versioned_override_native_review_and_scanner_subject(tmp_path: Path):
    policy = _write_policy(tmp_path / "policy.yaml", tmp_path)
    sbom = _write_sbom(tmp_path / "sbom.json")

    assert verify_container_sbom(sbom, policy, evidence_root=tmp_path) == {
        "packages": 3,
        "metadata_overrides": 1,
        "scanner_subjects": 1,
        "native_packages": 1,
        "native_review_packages": 1,
        "violations": 0,
        "status": "review_required",
    }


def test_container_sbom_rejects_unclassified_missing_license(tmp_path: Path):
    policy = _write_policy(tmp_path / "policy.yaml", tmp_path)
    sbom = _write_sbom(tmp_path / "sbom.json", [_package("unknown", "1.0", "pypi")])

    with pytest.raises(ValueError, match="missing license declaration"):
        verify_container_sbom(sbom, policy)


def test_container_sbom_rejects_override_version_drift(tmp_path: Path):
    policy = _write_policy(tmp_path / "policy.yaml", tmp_path)
    packages = [_package("legacy", "2.0", "pypi"), _write_document_root()]
    sbom = _write_sbom(tmp_path / "sbom.json", packages)

    with pytest.raises(ValueError, match="pinned to 1.0"):
        verify_container_sbom(sbom, policy)


def _write_document_root() -> dict:
    package = _package("image", "sha256:123", "oci")
    package["SPDXID"] = "SPDXRef-DocumentRoot-Image"
    return package


def test_container_sbom_rejects_changed_evidence(tmp_path: Path):
    policy = _write_policy(tmp_path / "policy.yaml", tmp_path)
    sbom = _write_sbom(tmp_path / "sbom.json")
    (tmp_path / "licenses/legacy").write_text("changed")

    with pytest.raises(ValueError, match="evidence SHA-256 changed"):
        verify_container_sbom(sbom, policy, evidence_root=tmp_path)


def test_container_sbom_rejects_forbidden_cargo_package(tmp_path: Path):
    policy = _write_policy(tmp_path / "policy.yaml", tmp_path)
    packages = [
        _package("legacy", "1.0", "pypi"),
        _package("rust-crate", "1.0", "cargo", "MIT"),
        _write_document_root(),
    ]
    sbom = _write_sbom(tmp_path / "sbom.json", packages)

    with pytest.raises(ValueError, match="forbidden package type cargo"):
        verify_container_sbom(sbom, policy)


def test_container_sbom_rejects_non_root_oci_exception(tmp_path: Path):
    policy = _write_policy(tmp_path / "policy.yaml", tmp_path)
    packages = [_package("legacy", "1.0", "pypi"), _package("nested", "1.0", "oci")]
    sbom = _write_sbom(tmp_path / "sbom.json", packages)

    with pytest.raises(ValueError, match="reviewed classification"):
        verify_container_sbom(sbom, policy)


def test_container_sbom_rejects_native_package_without_declared_license(tmp_path: Path):
    policy = _write_policy(tmp_path / "policy.yaml", tmp_path)
    packages = [_package("legacy", "1.0", "pypi"), _package("libc6", "1.0", "deb"), _write_document_root()]
    sbom = _write_sbom(tmp_path / "sbom.json", packages)

    with pytest.raises(ValueError, match="native package has no declared or concluded license"):
        verify_container_sbom(sbom, policy)


def test_container_sbom_keeps_organizational_review_open(tmp_path: Path):
    policy = _write_policy(tmp_path / "policy.yaml", tmp_path, status="approved")
    sbom = _write_sbom(tmp_path / "sbom.json")

    with pytest.raises(ValueError, match="expected native review status"):
        verify_container_sbom(sbom, policy, expect_status="review_required")


def test_production_container_policy_classifies_known_gaps_without_approving_distribution():
    policy = yaml.safe_load(PRODUCTION_POLICY.read_text())

    assert policy["forbidden_package_types"] == ["cargo"]
    assert policy["native_review"]["status"] == "review_required"
    assert policy["scanner_subject"]["spdx_id_prefix"] == "SPDXRef-DocumentRoot-"
    expected = {
        "annotated-types": ("0.7.0", "MIT"),
        "duckdb": ("1.5.4", "MIT"),
        "jinja2": ("3.1.6", "BSD-3-Clause"),
        "loguru": ("0.7.3", "MIT"),
        "markdown-it-py": ("4.2.0", "MIT"),
        "python": ("3.12.13", "PSF-2.0"),
    }
    assert {
        name: (override["version"], override["license"]) for name, override in policy["metadata_overrides"].items()
    } == expected

    loguru_notice = Path(__file__).parents[1] / policy["metadata_overrides"]["loguru"]["evidence"][1][
        "path"
    ].removeprefix("/app/")
    assert (
        hashlib.sha256(loguru_notice.read_bytes()).hexdigest()
        == policy["metadata_overrides"]["loguru"]["evidence"][1]["sha256"]
    )
