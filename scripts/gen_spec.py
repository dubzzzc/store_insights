#!/usr/bin/env python3
"""Legacy helper that now verifies the PyInstaller spec file."""
from pathlib import Path

spec_path = Path(__file__).with_name("vfp_dbf_to_rdsv2.spec")
if spec_path.exists():
    print(f"Reusing checked-in spec file: {spec_path.name}")
else:
    raise SystemExit("vfp_dbf_to_rdsv2.spec is missing; update the repository copy before building.")
