import json
from pathlib import Path

from casmsocial.datasets.colorado_front_range import load_profile, workflow


def _write_manifest(path: Path, status: str = "passed") -> dict[str, object]:
    manifest: dict[str, object] = {"schema_version": 1, "status": status}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_build_profile_orchestrates_all_stages_and_writes_resumable_receipt(tmp_path: Path, monkeypatch):
    profile = load_profile("example-10k")
    artifacts = {
        artifact_id: workflow.get_source_artifact(artifact_id) for artifact_id in workflow.REQUIRED_ARTIFACT_IDS
    }
    calls: list[str] = []
    monkeypatch.setattr(workflow, "_verified_artifacts", lambda data_dir: artifacts)

    def state_tables(output_dir, *args, **kwargs):
        calls.append("osf_tables")
        return _write_manifest(output_dir / "source_state=CO" / "manifest.json")

    def ducklake(state_dir, catalog, data_path, **kwargs):
        calls.append("osf_ducklake")
        return _write_manifest(catalog.with_suffix(".ducklake.manifest.json"))

    def population(*args, **kwargs):
        calls.append("profile_population")
        return _write_manifest(args[4] / "manifest.json")

    def schedules(*args, **kwargs):
        calls.append("schedules")
        return _write_manifest(args[5] / "manifest.json")

    def destinations(*args, **kwargs):
        calls.append("destinations")
        return _write_manifest(args[5] / "manifest.json")

    def runtime(*args, **kwargs):
        calls.append("runtime")
        return _write_manifest(args[3] / "manifest.json")

    def verification(*args, **kwargs):
        calls.append("runtime_verification")
        return _write_manifest(args[2] / "manifest.json")

    monkeypatch.setattr(workflow, "_state_tables", state_tables)
    monkeypatch.setattr(workflow, "build_ducklake", ducklake)
    monkeypatch.setattr(workflow, "build_profile_population", population)
    monkeypatch.setattr(workflow, "build_profile_schedules", schedules)
    monkeypatch.setattr(workflow, "build_profile_destinations", destinations)
    monkeypatch.setattr(workflow, "build_profile_runtime", runtime)
    monkeypatch.setattr(workflow, "verify_profile_runtime", verification)

    first = workflow.build_colorado_profile(tmp_path, "example-10k", profile)
    resumed = workflow.build_colorado_profile(tmp_path, "example-10k", profile)

    expected = [
        "osf_tables",
        "osf_ducklake",
        "profile_population",
        "schedules",
        "destinations",
        "runtime",
        "runtime_verification",
    ]
    assert calls == expected * 2
    assert first["status"] == "passed"
    assert first["resumed"] is False
    assert resumed["resumed"] is True
    assert [stage["name"] for stage in first["stages"]] == expected
    assert "identifier-bearing products remain local" in first["governance"]

    deferred = workflow.build_colorado_profile(
        tmp_path,
        "example-10k",
        profile,
        run_runtime_verification=False,
    )
    assert deferred["status"] == "built_unverified"
    assert deferred["stages"][-1] == {
        "name": "runtime_verification",
        "status": "skipped",
        "reason": "disabled",
    }


def test_build_plan_reports_all_source_statuses_without_writes(tmp_path: Path):
    plan = workflow.profile_build_plan(tmp_path, "example-1k", load_profile("example-1k"))

    assert plan["ready"] is False
    assert len(plan["sources"]) == len(workflow.REQUIRED_ARTIFACT_IDS)
    assert all(source["status"] == "missing" for source in plan["sources"])
    assert plan["stages"][-1]["required"] is True
    assert not (tmp_path / "local").exists()


def test_state_table_resume_requires_matching_source_hashes(tmp_path: Path, monkeypatch):
    population = tmp_path / "population.zip"
    education = tmp_path / "education.zip"
    population.write_bytes(b"population-v1")
    education.write_bytes(b"education-v1")
    state_dir = tmp_path / "tables" / "source_state=CO"
    manifest_path = state_dir / "manifest.json"
    manifest = {
        "schema_version": 1,
        "state": "CO",
        "inputs": {
            "population_archive": {"sha256": workflow.sha256_file(population)},
            "education_archive": {"sha256": workflow.sha256_file(education)},
        },
    }
    state_dir.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        workflow,
        "validate_state_partitions",
        lambda output_dir: [{"state": "CO", "manifest_path": str(manifest_path)}],
    )

    cached = workflow._cached_state_tables(tmp_path / "tables", population, education)
    population.write_bytes(b"population-v2")
    stale = workflow._cached_state_tables(tmp_path / "tables", population, education)

    assert cached is not None
    assert cached["resumed"] is True
    assert stale is None
