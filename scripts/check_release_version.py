#!/usr/bin/env python3
"""Fail CI when NietzscheDB public release metadata drifts out of sync."""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = (ROOT / "VERSION").read_text(encoding="utf-8").strip()

SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")
if not SEMVER.fullmatch(VERSION):
    raise SystemExit(f"VERSION is not valid semver: {VERSION!r}")


def toml_version(relative: str) -> str:
    with (ROOT / relative).open("rb") as fh:
        data = tomllib.load(fh)
    return str(data["package" if "package" in data else "project"]["version"])


checks = {
    "crates/nietzsche-server/Cargo.toml": toml_version("crates/nietzsche-server/Cargo.toml"),
    "crates/nietzsche-api/Cargo.toml": toml_version("crates/nietzsche-api/Cargo.toml"),
    "sdks/python/pyproject.toml": toml_version("sdks/python/pyproject.toml"),
}

init_text = (ROOT / "sdks/python/nietzschedb/__init__.py").read_text(encoding="utf-8")
m = re.search(r'__version__\s*=\s*"([^"]+)"', init_text)
checks["sdks/python/nietzschedb/__init__.py"] = m.group(1) if m else "<missing>"

errors: list[str] = []
for path, found in checks.items():
    if found != VERSION:
        errors.append(f"{path}: expected {VERSION}, found {found}")

changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
if not re.search(rf"^## \[{re.escape(VERSION)}\]\s+-\s+", changelog, re.MULTILINE):
    errors.append(f"CHANGELOG.md: missing release heading for {VERSION}")

dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
if f'org.opencontainers.image.version="{VERSION}"' not in dockerfile:
    errors.append(f"Dockerfile: OCI image version is not {VERSION}")

if errors:
    print("NietzscheDB release metadata drift detected:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    raise SystemExit(1)

print(f"release metadata OK: NietzscheDB {VERSION}")
