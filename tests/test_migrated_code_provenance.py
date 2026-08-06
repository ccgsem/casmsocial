from pathlib import Path

import pytest
import yaml

from scripts.verify_migrated_code_provenance import verify_migrated_code_provenance

ROOT = Path(__file__).parents[1]
ASSETS = ROOT / "casmsocial" / "datasets" / "colorado_front_range" / "assets"
PROVENANCE = ASSETS / "migrated_code_provenance.yaml"
LICENSES = ASSETS / "source_licenses.yaml"


def test_migrated_code_provenance_covers_every_recorded_source_destination_pair():
    result = verify_migrated_code_provenance(PROVENANCE, LICENSES, ROOT)

    assert result == {
        "sources": 12,
        "destinations": 7,
        "source_repository_verified": False,
        "authority_status": "organization_review_required",
    }


def test_migrated_code_provenance_rejects_license_manifest_drift(tmp_path: Path):
    provenance = yaml.safe_load(PROVENANCE.read_text())
    provenance["migrations"][0]["destinations"] = ["casmsocial/datasets/colorado_front_range/profile_runtime.py"]
    changed = tmp_path / "migrated_code_provenance.yaml"
    changed.write_text(yaml.safe_dump(provenance, sort_keys=False))

    with pytest.raises(ValueError, match="manifests disagree"):
        verify_migrated_code_provenance(changed, LICENSES, ROOT)


def test_migrated_code_provenance_rejects_authority_claim_as_automatically_complete(tmp_path: Path):
    provenance = yaml.safe_load(PROVENANCE.read_text())
    provenance["authority"]["status"] = "complete"
    changed = tmp_path / "migrated_code_provenance.yaml"
    changed.write_text(yaml.safe_dump(provenance, sort_keys=False))

    with pytest.raises(ValueError, match="organizational review"):
        verify_migrated_code_provenance(changed, LICENSES, ROOT)
