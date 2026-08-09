from __future__ import annotations

from pathlib import Path


def test_public_sources_exclude_restricted_dataset_identifiers() -> None:
    restricted_identifier = "rti_" + "synth_pop"
    roots = [Path("casmsocial"), Path("config"), Path("docs"), Path("examples"), Path("scripts")]
    paths = [Path("Makefile")]
    for root in roots:
        paths.extend(path for path in root.rglob("*") if path.is_file())

    matches: list[str] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if restricted_identifier in text.lower():
            matches.append(str(path))

    assert matches == []
