import json
from pathlib import Path

import polars as pl

from casmsocial.datasets.colorado_front_range import (
    aggregate_runtime_output,
    load_profile,
    runtime_verification,
    validate_runtime_runs,
    verify_profile_runtime,
)


def _write_outputs(root: Path, ranks: int = 1) -> None:
    occupancy = root / "schedule_occupancy.parquet"
    interactions = root / "social_interactions.parquet"
    occupancy.mkdir(parents=True)
    interactions.mkdir()
    rows = []
    for tick in range(25):
        for rank in range(ranks):
            rows.append({
                "tick": tick,
                "rank": rank,
                "active_person_count": 10,
                "active_place_count": 4,
                "co_located_person_count": 8,
                "max_place_occupancy": 3,
                "places_with_1_person": 1,
                "places_with_2_to_4_people": 3,
                "places_with_5_to_9_people": 0,
                "places_with_10_or_more_people": 0,
                "in_person_interaction_count": 2,
                "remote_message_count": 1,
            })
    pl.DataFrame(rows).write_parquet(occupancy / "part.parquet")
    pl.DataFrame({
        "channel": ["in_person", "remote"],
        "network_kind": ["social", "social"],
        "event_count": [25 * 2 * ranks, 25 * ranks],
    }).write_parquet(interactions / "part.parquet")


def test_aggregate_runtime_output_is_privacy_safe_and_rank_aware(tmp_path: Path):
    _write_outputs(tmp_path, ranks=2)
    summary = aggregate_runtime_output(tmp_path)

    assert summary["ticks"] == 25
    assert summary["all_ticks_have_active_people"] is True
    assert summary["peak_place_occupancy"] == 3
    assert summary["in_person_interaction_total"] == 100
    assert summary["remote_message_total"] == 50
    assert "person_id" not in summary


def test_profile_runtime_equivalence_accepts_matching_global_metrics():
    profile = load_profile("example-10k")
    single = {
        "ticks": 25,
        "all_ticks_have_active_people": True,
        "peak_place_occupancy": 4,
        "in_person_interaction_total": 100,
        "remote_message_total": 50,
    }
    parallel = {**single}

    result = validate_runtime_runs({1: single, 2: parallel}, profile)

    assert result["status"] == "passed"
    assert all(result["checks"].values())


def test_verify_profile_runtime_runs_required_ranks_and_resumes(tmp_path: Path, monkeypatch):
    profile = load_profile("example-10k")
    runtime_dir = tmp_path / "runtime"
    output_dir = tmp_path / "verification"
    (runtime_dir / "ducklake").mkdir(parents=True)
    (runtime_dir / "casmsocial.yaml").write_text("simulation: {}\n", encoding="utf-8")
    (runtime_dir / "manifest.json").write_text(
        json.dumps({
            "status": "passed",
            "profile_id": profile.profile_id,
            "profile_version": profile.profile_version,
        }),
        encoding="utf-8",
    )
    calls: list[int] = []
    summary = {
        "ticks": 25,
        "all_ticks_have_active_people": True,
        "peak_place_occupancy": 4,
        "in_person_interaction_total": 100,
        "remote_message_total": 50,
    }

    def fake_run(runtime, output, ranks, selected_profile):
        assert runtime == runtime_dir
        assert output == output_dir / f"rank-{ranks}"
        assert selected_profile is profile
        calls.append(ranks)
        return summary

    monkeypatch.setattr(runtime_verification, "_run", fake_run)

    first = verify_profile_runtime(runtime_dir, profile, output_dir)
    resumed = verify_profile_runtime(runtime_dir, profile, output_dir)

    assert calls == [1, 2]
    assert first["status"] == "passed"
    assert first["resumed"] is False
    assert resumed["resumed"] is True
