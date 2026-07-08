from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from scripts.list_mvp_artifacts import main as list_mvp_artifacts_main
from scripts.verify_mvp_manifest import MvpManifestVerificationError, verify_mvp_manifest
from scripts.write_mvp_manifest import MvpRunSpec, build_mvp_manifest, mvp_artifact_paths, write_mvp_manifest
from tests.test_mvp_output_validation import _write_agent_log, _write_behavior_log


def _write_summary(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# MVP Summary\n", encoding="utf-8")


def _write_validation_report(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "validation": {
                    "agents": 2,
                    "legs": 4,
                    "routes": {
                        "100->300": {
                            "origin_place_id": 100,
                            "destination_place_id": 300,
                            "origin_node_id": 1,
                            "destination_node_id": 3,
                            "distance_m": 5000.0,
                            "travel_time_min": 12,
                            "count": 1,
                        }
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_extra_artifact(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("generated artifact\n", encoding="utf-8")


def _run_specs(tmp_path: Path) -> tuple[MvpRunSpec, ...]:
    return (
        MvpRunSpec(
            name="single_rank",
            summary_path=tmp_path / "output" / "mvp_summary.md",
            agent_log_path=tmp_path / "output" / "mvp_agent_log.parquet",
            behavior_log_path=tmp_path / "output" / "mvp_behavior_log.parquet",
            expected_ranks=1,
        ),
        MvpRunSpec(
            name="two_rank",
            summary_path=tmp_path / "output" / "mvp_2rank_summary.md",
            agent_log_path=tmp_path / "output" / "mvp_2rank_agent_log.parquet",
            behavior_log_path=tmp_path / "output" / "mvp_2rank_behavior_log.parquet",
            expected_ranks=2,
        ),
        MvpRunSpec(
            name="routed",
            summary_path=tmp_path / "output" / "mvp_routed_summary.md",
            agent_log_path=tmp_path / "output" / "mvp_routed_agent_log.parquet",
            behavior_log_path=tmp_path / "output" / "mvp_routed_behavior_log.parquet",
            expected_ranks=1,
            validation_report_path=tmp_path / "output" / "mvp_routed_plan_validation.json",
        ),
        MvpRunSpec(
            name="built_roads",
            summary_path=tmp_path / "output" / "mvp_built_roads_summary.md",
            agent_log_path=tmp_path / "output" / "mvp_built_roads_agent_log.parquet",
            behavior_log_path=tmp_path / "output" / "mvp_built_roads_behavior_log.parquet",
            expected_ranks=1,
            validation_report_path=tmp_path / "output" / "mvp_built_roads_plan_validation.json",
            extra_artifact_paths=(
                tmp_path / "output" / "mvp_built_road_nodes.parquet",
                tmp_path / "output" / "mvp_built_road_edges.parquet",
                tmp_path / "output" / "mvp_built_place_road_snap.parquet",
                tmp_path / "output" / "mvp_built_road_artifacts.json",
            ),
        ),
        MvpRunSpec(
            name="delta_state",
            summary_path=tmp_path / "output" / "mvp_delta_state_summary.md",
            agent_log_path=tmp_path / "output" / "mvp_delta_state_agent_log.parquet",
            behavior_log_path=tmp_path / "output" / "mvp_delta_state_behavior_log.parquet",
            expected_ranks=1,
            validation_report_path=tmp_path / "output" / "mvp_delta_state_validation.json",
            extra_artifact_paths=(
                tmp_path / "output" / "mvp_agent_state_delta.parquet",
                tmp_path / "output" / "mvp_agent_state_delta_audit.parquet",
                tmp_path / "output" / "mvp_agent_state_reconstructed.parquet",
                tmp_path / "output" / "mvp_agent_state_delta_ducklake_report.md",
            ),
        ),
    )


def _write_outputs(run_specs: tuple[MvpRunSpec, ...]) -> None:
    for run_spec in run_specs:
        _write_summary(run_spec.summary_path)
        rank_by_agent = {1: 0, 2: 1} if run_spec.expected_ranks == 2 else None
        _write_agent_log(run_spec.agent_log_path, rank_by_agent=rank_by_agent)
        _write_behavior_log(run_spec.behavior_log_path, rank_by_agent=rank_by_agent)
        if run_spec.validation_report_path is not None:
            _write_validation_report(run_spec.validation_report_path)
        for artifact_path in run_spec.extra_artifact_paths:
            _write_extra_artifact(artifact_path)


def test_build_mvp_manifest_validates_runs_and_records_artifacts(tmp_path):
    run_specs = _run_specs(tmp_path)
    _write_outputs(run_specs)

    manifest = build_mvp_manifest(run_specs, generated_at="2026-05-25T00:00:00+00:00")

    assert manifest["version"] == 1
    assert manifest["generated_at"] == "2026-05-25T00:00:00+00:00"
    assert manifest["runs"]["single_rank"]["validation"]["agent"] == {
        "rows": 48,
        "runs": 1,
        "agents": 2,
        "ticks": 24,
        "ranks": 1,
    }
    assert manifest["runs"]["two_rank"]["validation"]["behavior"] == {
        "rows": 48,
        "runs": 1,
        "agents": 2,
        "ticks": 24,
        "ranks": 2,
    }
    assert manifest["runs"]["routed"]["validation"]["agent"] == {
        "rows": 48,
        "runs": 1,
        "agents": 2,
        "ticks": 24,
        "ranks": 1,
    }
    assert manifest["runs"]["routed"]["validation_report_path"] == str(run_specs[2].validation_report_path)
    assert manifest["runs"]["built_roads"]["validation_report_path"] == str(run_specs[3].validation_report_path)
    assert manifest["runs"]["built_roads"]["extra_artifact_paths"] == [
        str(path) for path in run_specs[3].extra_artifact_paths
    ]
    assert manifest["runs"]["delta_state"]["validation_report_path"] == str(run_specs[4].validation_report_path)
    assert manifest["runs"]["delta_state"]["extra_artifact_paths"] == [
        str(path) for path in run_specs[4].extra_artifact_paths
    ]
    assert len(manifest["artifacts"]) == 26
    summary_metadata = manifest["artifacts"][str(run_specs[0].summary_path)]
    assert summary_metadata["kind"] == "file"
    assert summary_metadata["size_bytes"] > 0
    assert len(summary_metadata["sha256"]) == 64
    agent_log_metadata = manifest["artifacts"][str(run_specs[1].agent_log_path)]
    assert agent_log_metadata["kind"] == "directory"
    assert agent_log_metadata["parquet_file_count"] > 0
    assert len(agent_log_metadata["sha256"]) == 64


def test_mvp_artifact_paths_cover_upload_manifest_and_run_outputs():
    paths = mvp_artifact_paths()

    assert paths[0] == Path("data/output/mvp_manifest.json")
    assert len(paths) == 27
    assert len(set(paths)) == len(paths)
    assert Path("data/output/mvp_built_road_artifacts.json") in paths
    assert Path("data/output/mvp_agent_state_reconstructed.parquet") in paths
    assert Path("data/output/mvp_agent_state_delta_ducklake_report.md") in paths
    assert len(mvp_artifact_paths(include_manifest=False)) == 26
    assert Path("data/output/mvp_manifest.json") not in mvp_artifact_paths(include_manifest=False)


def test_list_mvp_artifacts_prints_upload_paths(capsys):
    exit_code = list_mvp_artifacts_main([])

    assert exit_code == 0
    assert capsys.readouterr().out.splitlines() == [str(path) for path in mvp_artifact_paths()]


def test_write_mvp_manifest_writes_json_file(tmp_path):
    run_specs = _run_specs(tmp_path)
    _write_outputs(run_specs)
    manifest_path = tmp_path / "output" / "mvp_manifest.json"

    manifest = write_mvp_manifest(manifest_path, run_specs)

    written_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert written_manifest == manifest
    assert written_manifest["runs"]["two_rank"]["expected"]["ranks"] == 2
    assert written_manifest["runs"]["two_rank"]["expected"]["runs"] == 1
    assert written_manifest["runs"]["routed"]["expected"]["ranks"] == 1
    assert written_manifest["runs"]["built_roads"]["expected"]["ranks"] == 1


def test_verify_mvp_manifest_accepts_written_manifest(tmp_path):
    run_specs = _run_specs(tmp_path)
    _write_outputs(run_specs)
    manifest_path = tmp_path / "output" / "mvp_manifest.json"
    write_mvp_manifest(manifest_path, run_specs)

    summary = verify_mvp_manifest(manifest_path)

    assert summary["runs"] == 5
    assert summary["artifacts"] == 26


def test_verify_mvp_manifest_detects_changed_artifact(tmp_path):
    run_specs = _run_specs(tmp_path)
    _write_outputs(run_specs)
    manifest_path = tmp_path / "output" / "mvp_manifest.json"
    write_mvp_manifest(manifest_path, run_specs)
    run_specs[0].summary_path.write_text("# Changed Summary\n", encoding="utf-8")

    with pytest.raises(MvpManifestVerificationError, match="Manifest artifact mismatch"):
        verify_mvp_manifest(manifest_path)


def test_verify_mvp_manifest_accepts_flattened_artifact_download(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.chdir(repo_root)
    run_specs = (
        MvpRunSpec(
            name="single_rank",
            summary_path=Path("data/output/mvp_summary.md"),
            agent_log_path=Path("data/output/mvp_agent_log.parquet"),
            behavior_log_path=Path("data/output/mvp_behavior_log.parquet"),
            expected_ranks=1,
        ),
        MvpRunSpec(
            name="two_rank",
            summary_path=Path("data/output/mvp_2rank_summary.md"),
            agent_log_path=Path("data/output/mvp_2rank_agent_log.parquet"),
            behavior_log_path=Path("data/output/mvp_2rank_behavior_log.parquet"),
            expected_ranks=2,
        ),
        MvpRunSpec(
            name="routed",
            summary_path=Path("data/output/mvp_routed_summary.md"),
            agent_log_path=Path("data/output/mvp_routed_agent_log.parquet"),
            behavior_log_path=Path("data/output/mvp_routed_behavior_log.parquet"),
            expected_ranks=1,
            validation_report_path=Path("data/output/mvp_routed_plan_validation.json"),
        ),
        MvpRunSpec(
            name="built_roads",
            summary_path=Path("data/output/mvp_built_roads_summary.md"),
            agent_log_path=Path("data/output/mvp_built_roads_agent_log.parquet"),
            behavior_log_path=Path("data/output/mvp_built_roads_behavior_log.parquet"),
            expected_ranks=1,
            validation_report_path=Path("data/output/mvp_built_roads_plan_validation.json"),
            extra_artifact_paths=(
                Path("data/output/mvp_built_road_nodes.parquet"),
                Path("data/output/mvp_built_road_edges.parquet"),
                Path("data/output/mvp_built_place_road_snap.parquet"),
                Path("data/output/mvp_built_road_artifacts.json"),
            ),
        ),
        MvpRunSpec(
            name="delta_state",
            summary_path=Path("data/output/mvp_delta_state_summary.md"),
            agent_log_path=Path("data/output/mvp_delta_state_agent_log.parquet"),
            behavior_log_path=Path("data/output/mvp_delta_state_behavior_log.parquet"),
            expected_ranks=1,
            validation_report_path=Path("data/output/mvp_delta_state_validation.json"),
            extra_artifact_paths=(
                Path("data/output/mvp_agent_state_delta.parquet"),
                Path("data/output/mvp_agent_state_delta_audit.parquet"),
                Path("data/output/mvp_agent_state_reconstructed.parquet"),
                Path("data/output/mvp_agent_state_delta_ducklake_report.md"),
            ),
        ),
    )
    _write_outputs(run_specs)
    manifest_path = Path("data/output/mvp_manifest.json")
    write_mvp_manifest(manifest_path, run_specs)

    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    artifact_paths = [manifest_path]
    for run_spec in run_specs:
        artifact_paths.extend((run_spec.summary_path, run_spec.agent_log_path, run_spec.behavior_log_path))
        if run_spec.validation_report_path is not None:
            artifact_paths.append(run_spec.validation_report_path)
        artifact_paths.extend(run_spec.extra_artifact_paths)

    for artifact_path in artifact_paths:
        target_path = artifact_root / artifact_path.name
        if artifact_path.is_dir():
            shutil.copytree(artifact_path, target_path)
        else:
            shutil.copy2(artifact_path, target_path)

    Path("data/output/mvp_routed_plan_validation.json").write_text('{"stale": true}\n', encoding="utf-8")
    summary = verify_mvp_manifest(artifact_root / "mvp_manifest.json")

    assert summary["runs"] == 5
    assert summary["artifacts"] == 26
    assert all(Path(path).parent == artifact_root for path in summary["resolved_artifacts"].values())


def test_build_mvp_manifest_rejects_missing_summary(tmp_path):
    run_specs = _run_specs(tmp_path)
    _write_outputs(run_specs)
    run_specs[0].summary_path.unlink()

    with pytest.raises(FileNotFoundError):
        build_mvp_manifest(run_specs)
