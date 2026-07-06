from __future__ import annotations

from pathlib import Path

from scripts.clean_mvp_artifacts import clean_mvp_artifacts
from scripts.write_mvp_manifest import mvp_artifact_paths


def _write_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("generated", encoding="utf-8")


def test_clean_mvp_artifacts_removes_only_configured_paths(tmp_path):
    summary_path = tmp_path / "output" / "mvp_summary.md"
    agent_log_path = tmp_path / "output" / "mvp_agent_log.parquet"
    behavior_log_path = tmp_path / "output" / "mvp_behavior_log.parquet"
    manifest_path = tmp_path / "output" / "mvp_manifest.json"
    two_rank_summary_path = tmp_path / "output" / "mvp_2rank_summary.md"
    two_rank_agent_log_path = tmp_path / "output" / "mvp_2rank_agent_log.parquet"
    two_rank_behavior_log_path = tmp_path / "output" / "mvp_2rank_behavior_log.parquet"
    routed_summary_path = tmp_path / "output" / "mvp_routed_summary.md"
    routed_agent_log_path = tmp_path / "output" / "mvp_routed_agent_log.parquet"
    routed_behavior_log_path = tmp_path / "output" / "mvp_routed_behavior_log.parquet"
    routed_plan_validation_path = tmp_path / "output" / "mvp_routed_plan_validation.json"
    built_road_nodes_path = tmp_path / "output" / "mvp_built_road_nodes.parquet"
    built_road_edges_path = tmp_path / "output" / "mvp_built_road_edges.parquet"
    built_place_snap_path = tmp_path / "output" / "mvp_built_place_road_snap.parquet"
    built_road_report_path = tmp_path / "output" / "mvp_built_road_artifacts.json"
    built_roads_summary_path = tmp_path / "output" / "mvp_built_roads_summary.md"
    built_roads_agent_log_path = tmp_path / "output" / "mvp_built_roads_agent_log.parquet"
    built_roads_behavior_log_path = tmp_path / "output" / "mvp_built_roads_behavior_log.parquet"
    built_roads_plan_validation_path = tmp_path / "output" / "mvp_built_roads_plan_validation.json"
    delta_state_summary_path = tmp_path / "output" / "mvp_delta_state_summary.md"
    delta_state_agent_log_path = tmp_path / "output" / "mvp_delta_state_agent_log.parquet"
    delta_state_behavior_log_path = tmp_path / "output" / "mvp_delta_state_behavior_log.parquet"
    delta_state_log_path = tmp_path / "output" / "mvp_agent_state_delta.parquet"
    delta_state_audit_log_path = tmp_path / "output" / "mvp_agent_state_delta_audit.parquet"
    delta_state_reconstructed_log_path = tmp_path / "output" / "mvp_agent_state_reconstructed.parquet"
    delta_state_validation_path = tmp_path / "output" / "mvp_delta_state_validation.json"
    delta_state_ducklake_report_path = tmp_path / "output" / "mvp_agent_state_delta_ducklake_report.md"
    ducklake_path = tmp_path / "examples" / "mvp" / "mvp.ducklake"
    unrelated_output_path = tmp_path / "output" / "agent_log.parquet" / "part-0.parquet"
    readme_path = tmp_path / "examples" / "mvp" / "README.md"

    _write_file(summary_path)
    _write_file(agent_log_path / "tick=60" / "rank=0" / "part-0.parquet")
    _write_file(behavior_log_path / "tick=60" / "rank=0" / "part-0.parquet")
    _write_file(manifest_path)
    _write_file(two_rank_summary_path)
    _write_file(two_rank_agent_log_path / "tick=60" / "rank=0" / "part-0.parquet")
    _write_file(two_rank_behavior_log_path / "tick=60" / "rank=0" / "part-0.parquet")
    _write_file(routed_summary_path)
    _write_file(routed_agent_log_path / "tick=60" / "rank=0" / "part-0.parquet")
    _write_file(routed_behavior_log_path / "tick=60" / "rank=0" / "part-0.parquet")
    _write_file(routed_plan_validation_path)
    _write_file(built_road_nodes_path)
    _write_file(built_road_edges_path)
    _write_file(built_place_snap_path)
    _write_file(built_road_report_path)
    _write_file(built_roads_summary_path)
    _write_file(built_roads_agent_log_path / "tick=60" / "rank=0" / "part-0.parquet")
    _write_file(built_roads_behavior_log_path / "tick=60" / "rank=0" / "part-0.parquet")
    _write_file(built_roads_plan_validation_path)
    _write_file(delta_state_summary_path)
    _write_file(delta_state_agent_log_path / "tick=60" / "rank=0" / "part-0.parquet")
    _write_file(delta_state_behavior_log_path / "tick=60" / "rank=0" / "part-0.parquet")
    _write_file(delta_state_log_path / "tick=60" / "rank=0" / "part-0.parquet")
    _write_file(delta_state_audit_log_path / "tick=60" / "rank=0" / "part-0.parquet")
    _write_file(delta_state_reconstructed_log_path / "tick=60" / "rank=0" / "part-0.parquet")
    _write_file(delta_state_validation_path)
    _write_file(delta_state_ducklake_report_path)
    _write_file(ducklake_path / "metadata.sqlite")
    _write_file(unrelated_output_path)
    _write_file(readme_path)

    removed_paths = clean_mvp_artifacts(
        (
            summary_path,
            agent_log_path,
            behavior_log_path,
            manifest_path,
            two_rank_summary_path,
            two_rank_agent_log_path,
            two_rank_behavior_log_path,
            routed_summary_path,
            routed_agent_log_path,
            routed_behavior_log_path,
            routed_plan_validation_path,
            built_road_nodes_path,
            built_road_edges_path,
            built_place_snap_path,
            built_road_report_path,
            built_roads_summary_path,
            built_roads_agent_log_path,
            built_roads_behavior_log_path,
            built_roads_plan_validation_path,
            delta_state_summary_path,
            delta_state_agent_log_path,
            delta_state_behavior_log_path,
            delta_state_log_path,
            delta_state_audit_log_path,
            delta_state_reconstructed_log_path,
            delta_state_validation_path,
            delta_state_ducklake_report_path,
            ducklake_path,
        )
    )

    assert removed_paths == [
        summary_path,
        agent_log_path,
        behavior_log_path,
        manifest_path,
        two_rank_summary_path,
        two_rank_agent_log_path,
        two_rank_behavior_log_path,
        routed_summary_path,
        routed_agent_log_path,
        routed_behavior_log_path,
        routed_plan_validation_path,
        built_road_nodes_path,
        built_road_edges_path,
        built_place_snap_path,
        built_road_report_path,
        built_roads_summary_path,
        built_roads_agent_log_path,
        built_roads_behavior_log_path,
        built_roads_plan_validation_path,
        delta_state_summary_path,
        delta_state_agent_log_path,
        delta_state_behavior_log_path,
        delta_state_log_path,
        delta_state_audit_log_path,
        delta_state_reconstructed_log_path,
        delta_state_validation_path,
        delta_state_ducklake_report_path,
        ducklake_path,
    ]
    assert not summary_path.exists()
    assert not agent_log_path.exists()
    assert not behavior_log_path.exists()
    assert not manifest_path.exists()
    assert not two_rank_summary_path.exists()
    assert not two_rank_agent_log_path.exists()
    assert not two_rank_behavior_log_path.exists()
    assert not routed_summary_path.exists()
    assert not routed_agent_log_path.exists()
    assert not routed_behavior_log_path.exists()
    assert not routed_plan_validation_path.exists()
    assert not built_road_nodes_path.exists()
    assert not built_road_edges_path.exists()
    assert not built_place_snap_path.exists()
    assert not built_road_report_path.exists()
    assert not built_roads_summary_path.exists()
    assert not built_roads_agent_log_path.exists()
    assert not built_roads_behavior_log_path.exists()
    assert not built_roads_plan_validation_path.exists()
    assert not delta_state_summary_path.exists()
    assert not delta_state_agent_log_path.exists()
    assert not delta_state_behavior_log_path.exists()
    assert not delta_state_log_path.exists()
    assert not delta_state_audit_log_path.exists()
    assert not delta_state_reconstructed_log_path.exists()
    assert not delta_state_validation_path.exists()
    assert not delta_state_ducklake_report_path.exists()
    assert not ducklake_path.exists()
    assert unrelated_output_path.exists()
    assert readme_path.exists()


def test_clean_mvp_artifacts_ignores_missing_paths(tmp_path):
    missing_paths = (tmp_path / "output" / "mvp_summary.md", tmp_path / "examples" / "mvp" / "mvp.ducklake")

    removed_paths = clean_mvp_artifacts(missing_paths)

    assert removed_paths == []


def test_ci_uploads_and_verifies_standard_two_rank_and_routed_mvp_artifacts():
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    makefile = Path("Makefile").read_text(encoding="utf-8")

    assert ".PHONY: mvp-all" in makefile
    assert makefile.index("mvp-all:") < makefile.index("mvp-clean:")
    assert makefile.index("$(MAKE) --no-print-directory mvp\n") < makefile.index(
        "$(MAKE) --no-print-directory mvp-2rank"
    )
    assert makefile.index("$(MAKE) --no-print-directory mvp-2rank") < makefile.index(
        "$(MAKE) --no-print-directory mvp-routed"
    )
    assert makefile.index("$(MAKE) --no-print-directory mvp-routed") < makefile.index(
        "$(MAKE) --no-print-directory mvp-built-roads"
    )
    assert makefile.index("$(MAKE) --no-print-directory mvp-built-roads") < makefile.index(
        "$(MAKE) --no-print-directory mvp-delta-state"
    )
    assert makefile.index("$(MAKE) --no-print-directory mvp-delta-state") < makefile.index(
        "$(MAKE) --no-print-directory mvp-manifest"
    )
    assert makefile.index("scripts/validate_agent_state_delta.py") < makefile.index(
        "scripts/load_agent_state_delta_ducklake.py"
    )
    assert makefile.index("scripts/load_agent_state_delta_ducklake.py") < makefile.index(
        "scripts/report_agent_state_delta_ducklake.py"
    )
    assert makefile.index("$(MAKE) --no-print-directory mvp-manifest") < makefile.index(
        "$(MAKE) --no-print-directory mvp-verify-manifest"
    )
    assert workflow.index("Run tests") < workflow.index("Run MVP proof suite")
    assert workflow.index("Run MVP proof suite") < workflow.index("List MVP output artifact paths")
    assert workflow.index("List MVP output artifact paths") < workflow.index("Upload MVP output")
    assert workflow.index("Upload MVP output") < workflow.index("Download MVP output artifact")
    assert workflow.index("Download MVP output artifact") < workflow.index("Verify downloaded MVP output artifact")
    assert "make mvp-all" in workflow
    assert "id: mvp-artifacts" in workflow
    assert "make mvp-artifacts" in workflow
    assert "path: ${{ steps.mvp-artifacts.outputs.paths }}" in workflow
    assert "--manifest downloaded-mvp-output/mvp_manifest.json" in workflow
    assert "--artifact-root downloaded-mvp-output" in workflow
    assert len(mvp_artifact_paths()) == 27
