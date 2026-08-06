from pathlib import Path

import yaml

LICENSE_MANIFEST = (
    Path(__file__).parents[1] / "casmsocial" / "datasets" / "colorado_front_range" / "assets" / "source_licenses.yaml"
)


def test_source_license_manifest_preserves_distribution_gates():
    manifest = yaml.safe_load(LICENSE_MANIFEST.read_text())
    sources = manifest["data_sources"]

    assert manifest["schema_version"] == 1
    assert sources["osf_synthetic_population"]["license"] == "CC0-1.0"
    assert sources["osf_education_sites"]["license"] == "CC0-1.0"
    assert sources["bls_atus"]["rights"] == "US-federal-public-domain"
    assert sources["census_tiger_line"]["rights"] == "US-federal-public-domain"
    assert sources["openstreetmap"]["license"] == "ODbL-1.0"
    assert sources["openstreetmap"]["selected_distribution_policy"] == "local_build_only"
    assert "Local build by default" in sources["openstreetmap"]["distribution_policy"]

    gates = {gate["id"]: gate["status"] for gate in manifest["release_gates"]}
    assert gates == {
        "migrated_code_provenance": "documented_review_required",
        "dependency_license_inventory": "automated_review_required",
        "osm_distribution_policy": "implemented",
        "release_artifact_scan": "automated",
        "final_distribution_review": "organization_review_required",
    }
    migrated = manifest["software"]["migrated_mydatalakehouse_code"]["migrated_modules"]
    assert migrated == [
        {
            "destination": "casmsocial/datasets/colorado_front_range/profile_runtime.py",
            "sources": [
                "mydatalakehouse/colorado_front_range_routing.py",
                "mydatalakehouse/colorado_front_range_feasibility_routing.py",
                "mydatalakehouse/colorado_front_range_casmsocial_export.py",
            ],
            "source_commits": {
                "colorado_front_range_routing.py": "f9b0d2eb9fa830f7004eec1c95c3bdc6bef4abb6",
                "colorado_front_range_feasibility_routing.py": "f9b0d2eb9fa830f7004eec1c95c3bdc6bef4abb6",
                "colorado_front_range_casmsocial_export.py": "ec2336af55fb7221922ad972ec1b5f750746f114",
            },
            "notice": "THIRD_PARTY_NOTICES.md",
        },
        {
            "destination": "casmsocial/datasets/colorado_front_range/destination_supply.py",
            "sources": [
                "mydatalakehouse/colorado_front_range_osm_pois.py",
                "mydatalakehouse/colorado_front_range_routing.py",
            ],
            "source_commits": {
                "colorado_front_range_osm_pois.py": "ecb3b0c729f546c2db1dd4d2eeeab98add6d9c93",
                "colorado_front_range_routing.py": "f9b0d2eb9fa830f7004eec1c95c3bdc6bef4abb6",
            },
            "notice": "THIRD_PARTY_NOTICES.md",
        },
        {
            "destination": "casmsocial/datasets/colorado_front_range/atus.py",
            "sources": [
                "mydatalakehouse/atus_donor_diaries.py",
                "mydatalakehouse/atus_diary_normalization.py",
                "mydatalakehouse/dc_metro_donor_matching.py",
            ],
            "source_commits": {
                "atus_donor_diaries.py": "653250f6789f66315bf442a7252ab8bac85f3277",
                "atus_diary_normalization.py": "653250f6789f66315bf442a7252ab8bac85f3277",
                "dc_metro_donor_matching.py": "844cb7eba28ff4697b7606899a2eef2eb7bf4193",
            },
            "notice": "THIRD_PARTY_NOTICES.md",
        },
        {
            "destination": "casmsocial/datasets/colorado_front_range/profile_schedules.py",
            "source": "mydatalakehouse/colorado_front_range_schedules.py",
            "source_commit": "00380e58c1a33449d07bd346ce6c0df3eb6ceaf1",
            "notice": "THIRD_PARTY_NOTICES.md",
        },
        {
            "destination": "casmsocial/datasets/colorado_front_range/profile_population.py",
            "sources": [
                "mydatalakehouse/colorado_front_range_quality.py",
                "mydatalakehouse/colorado_front_range_pilot.py",
                "mydatalakehouse/colorado_front_range_casmsocial_fixture.py",
            ],
            "source_commit": "ec2336af55fb7221922ad972ec1b5f750746f114",
            "notice": "THIRD_PARTY_NOTICES.md",
        },
        {
            "destination": "casmsocial/datasets/colorado_front_range/osf_ducklake.py",
            "source": "mydatalakehouse/osf_synthetic_ducklake.py",
            "source_commit": "4a9687de19ad192b97139f085d3e348dfe187cbd",
            "notice": "THIRD_PARTY_NOTICES.md",
        },
        {
            "destination": "casmsocial/datasets/colorado_front_range/osf_tables.py",
            "source": "mydatalakehouse/osf_synthetic_ducklake.py",
            "source_commit": "4a9687de19ad192b97139f085d3e348dfe187cbd",
            "notice": "THIRD_PARTY_NOTICES.md",
        },
    ]
    assert (LICENSE_MANIFEST.parents[4] / "THIRD_PARTY_NOTICES.md").is_file()
