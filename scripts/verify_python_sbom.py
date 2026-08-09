"""Validate a CycloneDX Python SBOM against the reviewed license policy."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml  # type: ignore[import-untyped]

SPDX_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.-]*")
SPDX_OPERATORS = {"AND", "OR", "WITH"}


def _license_tokens(value: str) -> set[str]:
    return {token for token in SPDX_TOKEN.findall(value) if token not in SPDX_OPERATORS}


def _declared_licenses(component: dict, aliases: dict[str, str], ambiguous_names: set[str]) -> set[str]:
    declared: set[str] = set()
    for item in component.get("licenses", []):
        if expression := item.get("expression"):
            declared.update(_license_tokens(expression))
            continue
        license_data = item.get("license", {})
        value = license_data.get("id") or license_data.get("name")
        if value and value not in ambiguous_names:
            declared.update(_license_tokens(aliases.get(value, value)))
    return declared


def verify_python_sbom(sbom_path: Path, policy_path: Path) -> dict[str, int]:
    """Validate component coverage and return summary counts."""
    sbom = json.loads(sbom_path.read_text())
    policy = yaml.safe_load(policy_path.read_text())
    if sbom.get("bomFormat") != "CycloneDX" or not sbom.get("specVersion"):
        raise ValueError("SBOM must be a versioned CycloneDX document")
    root = sbom.get("metadata", {}).get("component", {})
    if root.get("name") != "casmsocial":
        raise ValueError("SBOM root component must be casmsocial")

    reviewed = set(policy["reviewed_spdx_identifiers"])
    aliases = policy.get("license_name_aliases", {})
    ambiguous_names = set(policy.get("ambiguous_license_names", []))
    if overlap := ambiguous_names & set(aliases):
        raise ValueError(f"Ambiguous license names cannot be normalized by alias: {sorted(overlap)}")
    root_licenses = _declared_licenses(root, aliases, ambiguous_names)
    if not root_licenses:
        raise ValueError("SBOM root component must declare the CASMSocial license")
    if unreviewed_root := root_licenses - reviewed:
        raise ValueError(f"CASMSocial has unreviewed license identifiers {sorted(unreviewed_root)}")
    overrides = policy.get("metadata_overrides", {})
    override_count = 0
    violations: list[str] = []
    components = sbom.get("components", [])
    for component in components:
        name = component.get("name", "<unnamed>")
        version = component.get("version")
        declared = _declared_licenses(component, aliases, ambiguous_names)
        if not declared:
            override = overrides.get(name)
            if not override or override.get("version") != version:
                violations.append(f"{name}=={version}: missing license declaration and reviewed override")
                continue
            evidence = override.get("evidence")
            license_expression = override.get("license")
            if not isinstance(evidence, str) or not evidence.strip():
                violations.append(f"{name}=={version}: reviewed override is missing license evidence")
                continue
            if not isinstance(license_expression, str) or not (declared := _license_tokens(license_expression)):
                violations.append(f"{name}=={version}: reviewed override is missing a license expression")
                continue
            override_count += 1
        unreviewed = declared - reviewed
        if unreviewed:
            violations.append(f"{name}=={version}: unreviewed license identifiers {sorted(unreviewed)}")

    if violations:
        raise ValueError("Python SBOM license policy failed:\n- " + "\n- ".join(violations))
    return {"components": len(components), "metadata_overrides": override_count, "violations": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sbom", type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    args = parser.parse_args()
    try:
        summary = verify_python_sbom(args.sbom, args.policy)
    except ValueError as error:
        print(error)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
