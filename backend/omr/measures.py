"""Split a preprocessed piano system into per-measure images for OMR retry."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MIN_STAFF_HEIGHT = 28


def _load_gray(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read score image: {path}")
    return image


def _dark_mask(gray: np.ndarray) -> np.ndarray:
    return ((gray < 128).astype(np.uint8)) * 255


def _staff_bands(dark: np.ndarray) -> list[tuple[int, int]]:
    row = dark.sum(axis=1)
    threshold = max(dark.shape[1] * 0.22, float(np.percentile(row, 80)) * 0.45)
    rows = np.where(row > threshold)[0]
    if rows.size == 0:
        return []

    bands: list[tuple[int, int]] = []
    start = prev = int(rows[0])
    for raw in rows[1:]:
        y = int(raw)
        if y - prev > 6:
            bands.append((start, prev))
            start = y
        prev = y
    bands.append((start, prev))
    staves = [(a, b) for a, b in bands if (b - a) >= MIN_STAFF_HEIGHT]
    return staves or bands


def _barline_xs(dark: np.ndarray, y0: int, y1: int) -> list[int]:
    band = dark[y0:y1, :]
    height = max(1, y1 - y0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, max(12, height // 2)))
    verts = cv2.morphologyEx(band, cv2.MORPH_OPEN, kernel)
    col = (verts > 0).sum(axis=0)
    hits = np.where(col > height * 0.42)[0]
    groups: list[list[int]] = []
    for raw in hits:
        x = int(raw)
        if not groups or x - groups[-1][-1] > 10:
            groups.append([x])
        else:
            groups[-1].append(x)
    xs = [int(round(sum(group) / len(group))) for group in groups]
    width = dark.shape[1]
    return [x for x in xs if 8 < x < width - 8]


def detect_measure_spans(gray: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive-exclusive (left, right) pixel spans for each measure."""
    dark = _dark_mask(gray)
    bands = _staff_bands(dark)
    if len(bands) >= 2:
        y0, y1 = bands[0][0], bands[-1][1]
    else:
        y0, y1 = 0, gray.shape[0]

    xs = _barline_xs(dark, y0, y1)
    if len(xs) < 2:
        return [(0, gray.shape[1])]

    # Leftmost hit is usually the system barline before the clefs.
    spans = [(xs[i], xs[i + 1]) for i in range(len(xs) - 1)]
    min_width = max(80, int(gray.shape[1] * 0.08))
    return [span for span in spans if span[1] - span[0] >= min_width]


def erase_inter_staff_marks(gray: np.ndarray) -> np.ndarray:
    """Clear hairpins/dynamics between grand-staff staves so MusicXML export cannot crash on them."""
    dark = _dark_mask(gray)
    bands = _staff_bands(dark)
    if len(bands) < 2:
        return gray

    top_bottom = bands[0][1]
    bottom_top = bands[1][0]
    gap = bottom_top - top_bottom
    if gap < 18:
        return gray

    cleaned = gray.copy()
    pad = max(4, gap // 8)
    y0 = top_bottom + pad
    y1 = bottom_top - pad
    x0 = int(gray.shape[1] * 0.16)
    if y1 <= y0:
        return gray
    cleaned[y0:y1, x0:] = 255
    return cleaned


def _staff_line_mask(dark: np.ndarray) -> np.ndarray:
    width = max(12, dark.shape[1] // 8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width, 1))
    return cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)


def isolate_measure(
    gray: np.ndarray,
    span: tuple[int, int],
    header_right: int,
    *,
    padding: int = 36,
) -> np.ndarray:
    """Keep clefs/key/time and one measure; clear notes in the other measures."""
    left, right = span
    width = gray.shape[1]
    left = max(0, left)
    right = min(width, right)
    isolated = gray.copy()
    dark = _dark_mask(gray)
    staff_lines = _staff_line_mask(dark) > 0
    clear = (dark > 0) & ~staff_lines

    # Keep the left header (clefs, key, time) and this measure. The previous
    # crop started at the header cut and dropped the clefs, so Audiveris
    # guessed pitches and the recovered bar was unreadable.
    for x in range(width):
        if x < header_right or left <= x < right:
            continue
        isolated[clear[:, x], x] = 255

    crop = isolated[:, 0 : min(width, right + padding)]
    return cv2.copyMakeBorder(
        crop,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=255,
    )


def write_measure_crops(image_path: Path, dest_dir: Path) -> list[Path]:
    """Write one PNG per detected measure. Empty if the system cannot be split."""
    gray = _load_gray(image_path)
    spans = detect_measure_spans(gray)
    if len(spans) < 2:
        return []

    header_right = spans[0][0] + int((spans[0][1] - spans[0][0]) * 0.42)
    header_right = max(spans[0][0] + 40, min(header_right, spans[0][1] - 20))

    dest_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, span in enumerate(spans, start=1):
        crop = isolate_measure(gray, span, header_right)
        path = dest_dir / f"measure-{index:02d}.png"
        ok, encoded = cv2.imencode(".png", crop)
        if not ok:
            continue
        path.write_bytes(encoded.tobytes())
        paths.append(path)
    return paths


def write_selected_measure_crops(
    image_path: Path,
    dest_dir: Path,
    measure_numbers: list[int],
) -> dict[int, Path]:
    """Write crops only for the requested 1-based measure numbers."""
    gray = _load_gray(image_path)
    spans = detect_measure_spans(gray)
    if not spans:
        return {}

    header_right = spans[0][0] + int((spans[0][1] - spans[0][0]) * 0.42)
    header_right = max(spans[0][0] + 40, min(header_right, spans[0][1] - 20))
    dest_dir.mkdir(parents=True, exist_ok=True)
    written: dict[int, Path] = {}
    for number in measure_numbers:
        if number < 1 or number > len(spans):
            continue
        crop = isolate_measure(gray, spans[number - 1], header_right)
        path = dest_dir / f"measure-{number:02d}.png"
        ok, encoded = cv2.imencode(".png", crop)
        if not ok:
            continue
        path.write_bytes(encoded.tobytes())
        written[number] = path
    return written
