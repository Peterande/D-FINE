#!/usr/bin/env python3
"""Compare reports only when their evaluation protocols are identical."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    left = json.loads(args.left.read_text())
    right = json.loads(args.right.read_text())
    if left["protocol_fingerprint"] != right["protocol_fingerprint"]:
        raise SystemExit("REFUSED: protocol fingerprints differ")
    print(json.dumps({"protocol_fingerprint": left["protocol_fingerprint"], "left": left["metrics"], "right": right["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
