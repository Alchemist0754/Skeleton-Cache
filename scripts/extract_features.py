from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.pipeline import extract_smie_bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a pre-encoded bundle for evaluation")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    out = extract_smie_bundle(args.config)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
