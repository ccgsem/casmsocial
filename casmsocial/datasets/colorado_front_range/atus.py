"""Stage, normalize, and match public ATUS donor diaries.

Derived from the mydatalakehouse ATUS staging, normalization, and donor
matching modules. See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from casmsocial.datasets.colorado_front_range.sources import sha256_file

DAY_TYPES = ("weekday", "weekend")
DIARY_DAY_START_MINUTE = 4 * 60
RESPONDENT_CORE_COLUMNS = {"TUCASEID", "TUDIARYDAY", "TUFINLWGT", "TELFS"}
RESPONDENT_DEMOGRAPHIC_COLUMNS = {"TEAGE", "TESEX"}
ROSTER_COLUMNS = {"TUCASEID", "TULINENO", "TEAGE", "TESEX"}
ACTIVITY_COLUMNS = {
    "TUCASEID",
    "TUACTIVITY_N",
    "TUSTARTTIM",
    "TUSTOPTIME",
    "TUTIER1CODE",
    "TEWHERE",
}
NORMALIZED_COLUMNS = {
    "donor_id",
    "day_type",
    "activity_sequence",
    "activity_kind",
    "start_minute",
    "end_minute",
    "atus_activity_code",
    "atus_location_code",
    "diary_weight",
    "age",
    "sex_code",
    "labor_force_status",
}
ACTIVITY_KIND_BY_TIER_1 = {
    1: "personal_care",
    4: "work",
    5: "school",
    6: "shopping",
    7: "healthcare",
    10: "meal",
    11: "social",
    12: "recreation",
    16: "travel",
}


def _require_columns(columns: set[str], required: set[str], label: str) -> None:
    if missing := required - columns:
        raise ValueError(f"{label} is missing columns: {', '.join(sorted(missing))}")


def _csv_columns(path: Path) -> set[str]:
    return set(pl.read_csv(path, n_rows=0).columns)


def _atus_time_to_diary_minute(column: str) -> pl.Expr:
    clock = pl.col(column).cast(pl.String)
    hour = clock.str.extract(r"^(\d{1,2}):", 1).cast(pl.Int64)
    minute = clock.str.extract(r"^\d{1,2}:(\d{2})", 1).cast(pl.Int64)
    return ((hour * 60 + minute - DIARY_DAY_START_MINUTE) % 1440).cast(pl.Int64)


def stage_atus_donor_diaries(
    respondents_path: Path,
    activities_path: Path,
    roster_path: Path,
    output_dir: Path,
    *,
    source_year: int = 2024,
) -> dict[str, object]:
    """Convert official ATUS extracts to typed, 04:00-based donor intervals."""
    paths = [path.expanduser().resolve() for path in (respondents_path, activities_path, roster_path)]
    respondents_path, activities_path, roster_path = paths
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if source_year < 2003:
        raise ValueError("ATUS source_year must be 2003 or later")

    respondent_columns = _csv_columns(respondents_path)
    _require_columns(respondent_columns, RESPONDENT_CORE_COLUMNS, "ATUS respondent extract")
    has_demographics = RESPONDENT_DEMOGRAPHIC_COLUMNS <= respondent_columns
    if not has_demographics:
        _require_columns(respondent_columns, {"TULINENO"}, "ATUS respondent extract")
    respondent_input = RESPONDENT_CORE_COLUMNS | (RESPONDENT_DEMOGRAPHIC_COLUMNS if has_demographics else {"TULINENO"})
    respondents = pl.read_csv(respondents_path, columns=sorted(respondent_input))

    if not has_demographics:
        roster_columns = _csv_columns(roster_path)
        _require_columns(roster_columns, ROSTER_COLUMNS, "ATUS roster extract")
        roster = pl.read_csv(roster_path, columns=sorted(ROSTER_COLUMNS)).select(
            pl.col("TUCASEID").cast(pl.String).alias("roster_case_id"),
            pl.col("TULINENO").cast(pl.Int64).alias("roster_line_number"),
            pl.col("TEAGE").cast(pl.Int64),
            pl.col("TESEX").cast(pl.Int64),
        )
        respondents = respondents.join(
            roster,
            left_on=[pl.col("TUCASEID").cast(pl.String), pl.col("TULINENO").cast(pl.Int64)],
            right_on=["roster_case_id", "roster_line_number"],
            how="left",
        )
        if respondents.select(pl.any_horizontal(pl.col(["TEAGE", "TESEX"]).is_null()).any()).item():
            raise ValueError("ATUS roster does not resolve age and sex for every respondent")

    respondent_frame = respondents.select(
        pl.col("TUCASEID").cast(pl.String).alias("case_id"),
        pl.col("TUDIARYDAY").cast(pl.Int64).alias("diary_day_code"),
        pl.col("TUFINLWGT").cast(pl.Float64).alias("diary_weight"),
        pl.col("TEAGE").cast(pl.Int64).alias("age"),
        pl.col("TESEX").cast(pl.Int64).alias("sex_code"),
        pl.col("TELFS").cast(pl.Int64).alias("labor_force_status"),
    ).with_columns(
        pl.concat_str([pl.lit(str(source_year)), pl.col("case_id")], separator=":").alias("donor_id"),
        pl
        .when(pl.col("diary_day_code").is_in([1, 7]))
        .then(pl.lit("weekend"))
        .otherwise(pl.lit("weekday"))
        .alias("day_type"),
    )

    activity_columns = _csv_columns(activities_path)
    _require_columns(activity_columns, ACTIVITY_COLUMNS, "ATUS activity extract")
    activities = pl.read_csv(activities_path, columns=sorted(ACTIVITY_COLUMNS))
    activity_frame = (
        activities
        .select(
            pl.col("TUCASEID").cast(pl.String).alias("case_id"),
            pl.col("TUACTIVITY_N").cast(pl.Int64).alias("activity_sequence"),
            pl.col("TUTIER1CODE").cast(pl.Int64).alias("atus_activity_code"),
            pl.col("TEWHERE").cast(pl.Int64).alias("atus_location_code"),
            _atus_time_to_diary_minute("TUSTARTTIM").alias("start_minute"),
            _atus_time_to_diary_minute("TUSTOPTIME").alias("raw_end_minute"),
        )
        .with_columns(
            pl
            .when(pl.col("raw_end_minute") <= pl.col("start_minute"))
            .then(pl.col("raw_end_minute") + 1440)
            .otherwise(pl.col("raw_end_minute"))
            .clip(upper_bound=1440)
            .alias("end_minute"),
            pl
            .col("atus_activity_code")
            .replace_strict(ACTIVITY_KIND_BY_TIER_1, default="other")
            .alias("activity_kind"),
        )
        .drop("raw_end_minute")
    )

    donors = (
        activity_frame
        .join(respondent_frame, on="case_id", how="inner")
        .select(
            "donor_id",
            "day_type",
            "activity_sequence",
            "activity_kind",
            "start_minute",
            "end_minute",
            "atus_activity_code",
            "atus_location_code",
            "diary_weight",
            "age",
            "sex_code",
            "labor_force_status",
        )
        .sort("donor_id", "activity_sequence")
    )
    if donors.is_empty():
        raise ValueError("ATUS donor staging produced no matched activities")
    if donors.select(pl.struct("donor_id", "activity_sequence").is_duplicated().any()).item():
        raise ValueError("ATUS donor activities must have unique donor and sequence pairs")
    if donors.select((pl.col("start_minute") >= pl.col("end_minute")).any()).item():
        raise ValueError("ATUS donor intervals must have positive duration")
    if donors.select((pl.col("diary_weight") <= 0).any()).item():
        raise ValueError("ATUS donor diary_weight must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "atus_donor_activities.parquet"
    donors.write_parquet(output_path, compression="zstd")
    return {
        "source_year": source_year,
        "diary_day_start_local_time": "04:00",
        "sources": {
            "respondents": {"path": str(respondents_path), "sha256": sha256_file(respondents_path)},
            "activities": {"path": str(activities_path), "sha256": sha256_file(activities_path)},
            "roster": {"path": str(roster_path), "sha256": sha256_file(roster_path)},
        },
        "output": {
            "path": "donors/atus_donor_activities.parquet",
            "rows": donors.height,
            "donors": donors["donor_id"].n_unique(),
            "sha256": sha256_file(output_path),
        },
    }


def _home_gap(template: dict[str, object], start: int, end: int) -> dict[str, object]:
    return {
        **template,
        "activity_kind": "home",
        "start_minute": start,
        "end_minute": end,
        "atus_activity_code": 1,
        "atus_location_code": None,
    }


def _compact_short_intervals(
    rows: list[dict[str, object]], minimum_minutes: int
) -> tuple[list[dict[str, object]], int]:
    compacted = 0
    index = 0
    while index < len(rows):
        row = rows[index]
        if int(row["end_minute"]) - int(row["start_minute"]) >= minimum_minutes or len(rows) == 1:
            index += 1
            continue
        if index:
            rows[index - 1]["end_minute"] = row["end_minute"]
        else:
            rows[index + 1]["start_minute"] = row["start_minute"]
        rows.pop(index)
        compacted += 1
    return rows, compacted


def normalize_atus_donor_diaries(
    input_path: Path,
    output_dir: Path,
    *,
    minimum_routable_minutes: int = 10,
) -> dict[str, object]:
    """Make every donor diary contiguous and complete over its 1,440 minutes."""
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if minimum_routable_minutes < 1:
        raise ValueError("minimum_routable_minutes must be positive")
    donors = pl.read_parquet(input_path)
    _require_columns(set(donors.columns), NORMALIZED_COLUMNS, "ATUS donor activities")
    rows: list[dict[str, object]] = []
    inserted_gaps = clipped_intervals = coalesced_intervals = compacted_intervals = 0
    for _, plan in donors.sort("donor_id", "activity_sequence").group_by("donor_id", maintain_order=True):
        cursor = 0
        normalized: list[dict[str, object]] = []
        for source in plan.to_dicts():
            start = max(0, int(source["start_minute"]))
            end = min(1440, int(source["end_minute"]))
            if start > cursor:
                normalized.append(_home_gap(source, cursor, start))
                inserted_gaps += 1
            if start < cursor:
                start = cursor
                clipped_intervals += 1
            if end <= start:
                continue
            source["start_minute"], source["end_minute"] = start, end
            if (
                normalized
                and normalized[-1]["activity_kind"] == source["activity_kind"]
                and normalized[-1]["end_minute"] == source["start_minute"]
            ):
                normalized[-1]["end_minute"] = end
                coalesced_intervals += 1
            else:
                normalized.append(source)
            cursor = max(cursor, end)
        if cursor < 1440:
            normalized.append(_home_gap(normalized[-1], cursor, 1440))
            inserted_gaps += 1
        normalized, compacted = _compact_short_intervals(normalized, minimum_routable_minutes)
        compacted_intervals += compacted
        for sequence, row in enumerate(normalized, start=1):
            row["activity_sequence"] = sequence
            rows.append(row)

    normalized = pl.DataFrame(
        rows,
        schema_overrides={
            "activity_sequence": pl.Int64,
            "start_minute": pl.Int64,
            "end_minute": pl.Int64,
            "atus_activity_code": pl.Int64,
            "atus_location_code": pl.Int64,
            "age": pl.Int64,
            "sex_code": pl.Int64,
            "labor_force_status": pl.Int64,
        },
    ).sort("donor_id", "activity_sequence")
    for donor_id, plan in normalized.group_by("donor_id", maintain_order=True):
        intervals = plan.sort("activity_sequence").select("start_minute", "end_minute").rows()
        if not intervals or intervals[0][0] != 0 or intervals[-1][1] != 1440:
            raise ValueError(f"Donor {donor_id} does not cover the diary day")
        if any(right[0] != left[1] or right[1] <= right[0] for left, right in zip(intervals, intervals[1:])):
            raise ValueError(f"Donor {donor_id} is not contiguous")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "atus_donor_activities_normalized.parquet"
    normalized.write_parquet(output_path, compression="zstd")
    return {
        "input_sha256": sha256_file(input_path),
        "output": {
            "path": "donors/atus_donor_activities_normalized.parquet",
            "rows": normalized.height,
            "donors": normalized["donor_id"].n_unique(),
            "sha256": sha256_file(output_path),
        },
        "normalization": {
            "minimum_routable_minutes": minimum_routable_minutes,
            "inserted_home_gaps": inserted_gaps,
            "clipped_overlaps": clipped_intervals,
            "coalesced_intervals": coalesced_intervals,
            "compacted_short_intervals": compacted_intervals,
        },
    }


def _age_band(age: pl.Expr) -> pl.Expr:
    return (
        pl
        .when(age < 25)
        .then(pl.lit("15_24"))
        .when(age < 45)
        .then(pl.lit("25_44"))
        .when(age < 65)
        .then(pl.lit("45_64"))
        .otherwise(pl.lit("65_plus"))
    )


def _uniform(person_ids: np.ndarray, seed: int, day_index: int) -> np.ndarray:
    values = person_ids.astype(np.uint64, copy=True)
    salt = np.uint64((seed + (day_index + 1) * 0x9E3779B97F4A7C15) & ((1 << 64) - 1))
    values ^= salt
    values ^= values >> np.uint64(30)
    values *= np.uint64(0xBF58476D1CE4E5B9)
    values ^= values >> np.uint64(27)
    values *= np.uint64(0x94D049BB133111EB)
    values ^= values >> np.uint64(31)
    return values.astype(np.float64) / np.float64(np.iinfo(np.uint64).max)


def _candidate_set(
    donors: pl.DataFrame,
    day_type: str,
    age_band: str,
    sex_code: int,
    labor_force_status: int,
) -> tuple[pl.DataFrame, str]:
    options = [
        (
            (pl.col("day_type") == day_type)
            & (pl.col("age_band") == age_band)
            & (pl.col("sex_code") == sex_code)
            & (pl.col("labor_force_status") == labor_force_status),
            "exact",
        ),
        (
            (pl.col("day_type") == day_type) & (pl.col("age_band") == age_band) & (pl.col("sex_code") == sex_code),
            "age_sex",
        ),
        ((pl.col("day_type") == day_type) & (pl.col("age_band") == age_band), "age_band"),
        (pl.col("day_type") == day_type, "day_type"),
    ]
    for predicate, level in options:
        candidates = donors.filter(predicate).sort("donor_id")
        if not candidates.is_empty():
            return candidates, level
    raise ValueError(f"No ATUS donors are available for day_type={day_type}")


def assign_atus_donors(
    persons_path: Path,
    donor_activities_path: Path,
    output_dir: Path,
    *,
    random_seed: int,
) -> dict[str, object]:
    """Assign one weighted weekday and weekend ATUS diary to each adult."""
    for path in (persons_path, donor_activities_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    persons = pl.read_parquet(persons_path)
    donors = pl.read_parquet(donor_activities_path)
    _require_columns(set(persons.columns), {"person_id", "age", "sex_code", "labor_force_status"}, "Synthetic persons")
    _require_columns(
        set(donors.columns),
        {"donor_id", "day_type", "diary_weight", "age", "sex_code", "labor_force_status"},
        "ATUS donor activities",
    )
    people = persons.select(
        pl.col("person_id").cast(pl.Int64),
        pl.col("age").cast(pl.Int64),
        pl.col("sex_code").cast(pl.Int64),
        pl.col("labor_force_status").cast(pl.Int64),
    ).filter(pl.col("age") >= 15)
    if people.is_empty():
        raise ValueError("Synthetic persons contains no adults age 15 or older")
    if people["person_id"].is_duplicated().any():
        raise ValueError("Synthetic person_id values must be unique")
    donor_people = (
        donors
        .select("donor_id", "day_type", "diary_weight", "age", "sex_code", "labor_force_status")
        .unique()
        .with_columns(_age_band(pl.col("age")).alias("age_band"))
    )
    if not set(DAY_TYPES) <= set(donor_people["day_type"]):
        raise ValueError("ATUS donor activities must include weekday and weekend donors")
    targets = people.with_columns(_age_band(pl.col("age")).alias("age_band"))

    assignments: list[pl.DataFrame] = []
    for day_index, day_type in enumerate(DAY_TYPES):
        for stratum in targets.partition_by(["age_band", "sex_code", "labor_force_status"], as_dict=False):
            age_band, sex_code, labor_status = stratum.select("age_band", "sex_code", "labor_force_status").row(0)
            candidates, level = _candidate_set(donor_people, day_type, age_band, sex_code, labor_status)
            weights = candidates["diary_weight"].to_numpy().astype(np.float64)
            if np.any(weights <= 0) or not np.isfinite(weights).all():
                raise ValueError("ATUS donor diary_weight must be finite and positive")
            cumulative = np.cumsum(weights / weights.sum())
            positions = np.searchsorted(
                cumulative,
                _uniform(stratum["person_id"].to_numpy(), random_seed, day_index),
                side="right",
            )
            positions = np.minimum(positions, len(cumulative) - 1)
            assignments.append(
                stratum.select("person_id").with_columns(
                    pl.lit(day_type).alias("day_type"),
                    pl.Series("donor_id", candidates["donor_id"].to_numpy()[positions]).cast(pl.String),
                    pl.lit(level).alias("match_level"),
                    pl.lit(age_band).alias("match_age_band"),
                    pl.lit(sex_code).cast(pl.Int64).alias("match_sex_code"),
                    pl.lit(labor_status).cast(pl.Int64).alias("match_labor_force_status"),
                    pl.Series("selected_donor_weight", weights[positions]).cast(pl.Float64),
                    pl.lit(random_seed).cast(pl.Int64).alias("random_seed"),
                )
            )

    frame = pl.concat(assignments).sort("person_id", "day_type")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "atus_donor_assignments.parquet"
    frame.write_parquet(output_path, compression="zstd")
    return {
        "random_seed": random_seed,
        "output": {
            "path": "atus_donor_assignments.parquet",
            "rows": frame.height,
            "adult_persons": people.height,
            "match_level_counts": frame.group_by("match_level").len().sort("match_level").to_dicts(),
            "sha256": sha256_file(output_path),
        },
    }
