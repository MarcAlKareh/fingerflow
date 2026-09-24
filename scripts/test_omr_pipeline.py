#!/usr/bin/env python3
"""Smoke-test the full image preprocessing -> Audiveris pipeline.

This is an integration test, so Audiveris must be installed separately.

Example:
    python scripts/test_omr_pipeline.py samples/score.jpg
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.omr import AudiverisError, recognize_with_recovery, validate_musicxml  # noqa: E402
from scripts.preprocess_score import preprocess_score  # noqa: E402


def run_pipeline(
    input_image: Path,
    output_root: Path,
    audiveris_command: str | None,
) -> Path:
    """Preprocess one image, recognize it, and return valid MusicXML."""
    if not input_image.is_file():
        raise FileNotFoundError(f"Input image does not exist: {input_image}")

    # A unique folder prevents output from an older run passing this test.
    run_dir = output_root / f"{input_image.stem}-{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)

    preprocessed_path = run_dir / "preprocessed.png"
    audiveris_output = run_dir / "audiveris"

    print(f"[1/3] Preprocessing {input_image}")
    preprocess_score(
        input_image,
        preprocessed_path,
        target_width=2200,
        crop_padding=48,
        threshold_block_size=31,
        threshold_c=12,
        debug_dir=None,
    )

    print("[2/3] Running Audiveris (this may take a few minutes)")
    result = recognize_with_recovery(
        preprocessed_path,
        audiveris_output,
        audiveris_command=audiveris_command,
    )

    print("[3/3] Validating MusicXML")
    validate_musicxml(result.musicxml_path)
    print(f"PASS: valid MusicXML produced at {result.musicxml_path}")

    if result.additional_musicxml_paths:
        print(
            "Audiveris also exported "
            f"{len(result.additional_musicxml_paths)} additional movement(s)."
        )

    return result.musicxml_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test preprocessing followed by Audiveris recognition."
    )
    parser.add_argument("image", type=Path, help="Input sheet music image")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "tmp" / "omr-tests",
        help="Folder for test artifacts (default: tmp/omr-tests)",
    )
    parser.add_argument(
        "--audiveris",
        default=None,
        help="Path to Audiveris launcher; otherwise uses AUDIVERIS_CMD or PATH",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        run_pipeline(
            args.image.resolve(),
            args.output_dir.resolve(),
            args.audiveris,
        )
    except (AudiverisError, FileNotFoundError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
