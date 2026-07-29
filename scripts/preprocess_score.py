#!/usr/bin/env python3
"""
Preprocess sheet music images for OMR tools such as Audiveris.

Pipeline:
1. Load image
2. Convert to grayscale
3. Resize for a consistent working resolution
4. Boost local contrast
5. Denoise lightly
6. Deskew based on foreground pixels
7. Crop to content
8. Adaptive threshold to black/white
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def decode_image(data: bytes) -> np.ndarray:
    """Turn raw image bytes (PNG/JPG) into an OpenCV color image."""
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image bytes")
    return image


def encode_image(image: np.ndarray, suffix: str = ".png") -> bytes:
    """Turn an OpenCV image back into file bytes (default: PNG)."""
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError("Could not encode image")
    return encoded.tobytes()


def read_image(path: Path) -> np.ndarray:
    """Load an image from disk into an OpenCV array."""
    return decode_image(path.read_bytes())


def write_image(path: Path, image: np.ndarray) -> None:
    """Save an OpenCV image to disk, creating folders if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    path.write_bytes(encode_image(image, suffix))


def resize_to_target_width(image: np.ndarray, target_width: int) -> np.ndarray:
    """Scale the image so its width matches target_width (keeps aspect ratio)."""
    height, width = image.shape[:2]
    if width == target_width:
        return image

    scale = target_width / float(width)
    # Cubic when enlarging, area when shrinking — both reduce artifacts
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(
        image,
        (target_width, max(1, int(round(height * scale)))),
        interpolation=interpolation,
    )


def deskew_image(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Straighten a slightly tilted page using the ink pixels' orientation."""
    # Invert so ink becomes white (easier for angle detection)
    inverted = cv2.bitwise_not(image)
    points = np.column_stack(np.where(inverted > 0))

    # Not enough ink pixels to estimate a reliable angle
    if len(points) < 200:
        return image, 0.0

    angle = cv2.minAreaRect(points)[-1]
    if angle < -45:
        angle = 90 + angle

    # Ignore tiny tilts — rotating them can hurt more than it helps
    if abs(angle) < 0.15:
        return image, 0.0

    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, angle


def crop_to_content(image: np.ndarray, padding: int) -> np.ndarray:
    """Trim empty margins and keep a small padding around the score."""
    # Pixels darker than near-white are treated as content
    mask = image < 245
    coords = np.argwhere(mask)
    if coords.size == 0:
        return image

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    y_min = max(0, y_min - padding)
    x_min = max(0, x_min - padding)
    y_max = min(image.shape[0], y_max + padding + 1)
    x_max = min(image.shape[1], x_max + padding + 1)

    return image[y_min:y_max, x_min:x_max]


def adaptive_binarize(image: np.ndarray, block_size: int, c_value: int) -> np.ndarray:
    """Convert grayscale to black/white using local (adaptive) thresholds."""
    # OpenCV requires an odd neighborhood size
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(3, block_size)

    return cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c_value,
    )


def preprocess_array(
    original: np.ndarray,
    *,
    target_width: int = 2200,
    crop_padding: int = 24,
    threshold_block_size: int = 31,
    threshold_c: int = 12,
) -> tuple[np.ndarray, float]:
    """Run the full cleanup pipeline on an in-memory OpenCV image."""
    # 1. Grayscale — OMR cares about ink vs paper, not color
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    # 2. Normalize size so later steps behave consistently
    resized = resize_to_target_width(gray, target_width)

    # 3. Boost local contrast so faint staff lines / notes stand out
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    contrast = clahe.apply(resized)

    # 4. Light denoise without wiping out thin notation
    denoised = cv2.fastNlMeansDenoising(
        contrast, None, h=9, templateWindowSize=7, searchWindowSize=21
    )
    # 5. Rough binary for deskew, then straighten, crop, and finalize
    provisional_binary = adaptive_binarize(denoised, threshold_block_size, threshold_c)
    deskewed, angle = deskew_image(provisional_binary)
    cropped = crop_to_content(deskewed, crop_padding)
    final = adaptive_binarize(cropped, threshold_block_size, threshold_c)
    return final, angle


def preprocess_bytes(
    data: bytes,
    *,
    target_width: int = 2200,
    crop_padding: int = 24,
    threshold_block_size: int = 31,
    threshold_c: int = 12,
) -> tuple[bytes, float]:
    """API helper: bytes in → cleaned PNG bytes out (plus deskew angle)."""
    original = decode_image(data)
    final, angle = preprocess_array(
        original,
        target_width=target_width,
        crop_padding=crop_padding,
        threshold_block_size=threshold_block_size,
        threshold_c=threshold_c,
    )
    return encode_image(final, ".png"), angle


def preprocess_score(
    input_path: Path,
    output_path: Path,
    *,
    target_width: int,
    crop_padding: int,
    threshold_block_size: int,
    threshold_c: int,
    debug_dir: Path | None,
) -> None:
    """CLI helper: read a file, preprocess it, write the result (optional debug dumps)."""
    original = read_image(input_path)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    resized = resize_to_target_width(gray, target_width)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    contrast = clahe.apply(resized)

    denoised = cv2.fastNlMeansDenoising(
        contrast, None, h=9, templateWindowSize=7, searchWindowSize=21
    )
    provisional_binary = adaptive_binarize(denoised, threshold_block_size, threshold_c)
    deskewed, angle = deskew_image(provisional_binary)
    cropped = crop_to_content(deskewed, crop_padding)
    final = adaptive_binarize(cropped, threshold_block_size, threshold_c)

    write_image(output_path, final)

    # Optional: save each intermediate step so you can tune the pipeline
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        write_image(debug_dir / "01-gray.png", gray)
        write_image(debug_dir / "02-resized.png", resized)
        write_image(debug_dir / "03-contrast.png", contrast)
        write_image(debug_dir / "04-denoised.png", denoised)
        write_image(debug_dir / "05-binary-before-deskew.png", provisional_binary)
        write_image(debug_dir / "06-deskewed.png", deskewed)
        write_image(debug_dir / "07-cropped.png", cropped)
        write_image(debug_dir / "08-final.png", final)
        (debug_dir / "meta.txt").write_text(
            "\n".join(
                [
                    f"input={input_path}",
                    f"output={output_path}",
                    f"target_width={target_width}",
                    f"crop_padding={crop_padding}",
                    f"threshold_block_size={threshold_block_size}",
                    f"threshold_c={threshold_c}",
                    f"deskew_angle_deg={angle:.4f}",
                ]
            ),
            encoding="utf-8",
        )


def build_parser() -> argparse.ArgumentParser:
    """Define the command-line flags for the CLI script."""
    parser = argparse.ArgumentParser(
        description="Preprocess a sheet music image for OMR."
    )
    parser.add_argument("input", type=Path, help="Input image path (.png, .jpg, .jpeg)")
    parser.add_argument("output", type=Path, help="Output image path")
    parser.add_argument(
        "--target-width",
        type=int,
        default=2200,
        help="Resize image to this width before processing (default: 2200)",
    )
    parser.add_argument(
        "--crop-padding",
        type=int,
        default=24,
        help="Extra padding in pixels around detected content (default: 24)",
    )
    parser.add_argument(
        "--threshold-block-size",
        type=int,
        default=31,
        help="Adaptive threshold neighborhood size, must be odd (default: 31)",
    )
    parser.add_argument(
        "--threshold-c",
        type=int,
        default=12,
        help="Adaptive threshold constant subtracted from local mean (default: 12)",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Optional directory to save intermediate steps",
    )
    return parser


def main() -> None:
    """Parse CLI args and run file-based preprocessing."""
    parser = build_parser()
    args = parser.parse_args()

    preprocess_score(
        args.input,
        args.output,
        target_width=args.target_width,
        crop_padding=args.crop_padding,
        threshold_block_size=args.threshold_block_size,
        threshold_c=args.threshold_c,
        debug_dir=args.debug_dir,
    )


if __name__ == "__main__":
    main()
