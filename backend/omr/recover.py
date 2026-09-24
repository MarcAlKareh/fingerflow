"""Retry Audiveris when a full-system export drops measures."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import music21 as m21

from .annotate import compact_piano_layout
from .audiveris import AudiverisRecognitionError, AudiverisResult, recognize_score
from .measures import erase_inter_staff_marks, write_selected_measure_crops

_RAW_MEASURES = re.compile(r"(\d+)\s+raw measures")
_FAILED_MEASURE = re.compile(r"Error visiting Measure\{#(\d+)\}")


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def first_part_measure_count(path: Path) -> int:
    return len(exported_measure_numbers(path))


def exported_measure_numbers(path: Path) -> list[int]:
    root = ET.parse(path).getroot()
    parts = [child for child in root if _local(child.tag) == "part"]
    if not parts:
        return []
    numbers: list[int] = []
    for child in parts[0]:
        if _local(child.tag) != "measure":
            continue
        raw = child.attrib.get("number", "")
        try:
            numbers.append(int(raw))
        except ValueError:
            numbers.append(len(numbers) + 1)
    return numbers


def raw_measure_count(log_text: str) -> int | None:
    matches = _RAW_MEASURES.findall(log_text or "")
    if not matches:
        return None
    return int(matches[-1])


def missing_measure_numbers(result: AudiverisResult) -> list[int]:
    failed = [int(num) for num in _FAILED_MEASURE.findall(result.log_text or "")]
    if failed:
        return sorted(set(failed))
    raw = raw_measure_count(result.log_text)
    exported = set(exported_measure_numbers(result.musicxml_path))
    if raw is None:
        return []
    return [number for number in range(1, raw + 1) if number not in exported]


def needs_recovery(result: AudiverisResult) -> bool:
    return bool(missing_measure_numbers(result))


def _as_score(path: Path) -> m21.stream.Score:
    parsed = m21.converter.parse(str(path))
    if isinstance(parsed, m21.stream.Score):
        return parsed
    wrapped = m21.stream.Score()
    wrapped.insert(0, parsed)
    return wrapped


def _measure_note_count(measure: m21.stream.Measure) -> int:
    return sum(len(element.pitches) for element in measure.flatten().notes)


def _noted_measures(part: m21.stream.Part) -> list[m21.stream.Measure]:
    return [
        measure
        for measure in part.getElementsByClass(m21.stream.Measure)
        if _measure_note_count(measure) > 0
    ]


def _best_measure(part: m21.stream.Part) -> m21.stream.Measure | None:
    scored = _noted_measures(part)
    if not scored:
        return None
    return max(scored, key=_measure_note_count)


def _write_score(score: m21.stream.Score, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = Path(str(score.write("musicxml", fp=str(dest))))
    if written.resolve() != dest.resolve() and written.is_file():
        dest.write_bytes(written.read_bytes())
    return dest


def _measure_for_part(rec_score: m21.stream.Score, part_index: int) -> m21.stream.Measure | None:
    if part_index < len(rec_score.parts):
        return _best_measure(rec_score.parts[part_index])
    if len(rec_score.parts) == 1:
        noted = _noted_measures(rec_score.parts[0])
        if part_index < len(noted):
            return noted[part_index]
    return None


def insert_recovered_measures(
    base_xml: Path,
    recovered: dict[int, Path],
    dest: Path,
) -> Path:
    """Keep the original export and splice in bars recovered from isolated crops."""
    base = _as_score(base_xml)
    recovered_scores = {number: _as_score(path) for number, path in recovered.items()}
    combined = m21.stream.Score()

    for part_index, part in enumerate(base.parts):
        by_number: dict[int, m21.stream.Measure] = {}
        for measure in part.getElementsByClass(m21.stream.Measure):
            try:
                number = int(measure.number)
            except (TypeError, ValueError):
                number = len(by_number) + 1
            by_number[number] = deepcopy(measure)

        for number, rec_score in recovered_scores.items():
            rec_measure = _measure_for_part(rec_score, part_index)
            if rec_measure is None:
                continue
            copied = deepcopy(rec_measure)
            copied.number = number
            by_number[number] = copied

        new_part = m21.stream.Part()
        for number in sorted(by_number):
            new_part.append(by_number[number])
        if new_part.getElementsByClass(m21.stream.Measure):
            combined.append(new_part)

    compact_piano_layout(combined)
    return _write_score(combined, dest)


def _save_gray_png(array, path: Path) -> Path:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", array)
    if not ok:
        raise ValueError(f"Could not encode {path}")
    path.write_bytes(encoded.tobytes())
    return path


def _recognize_erased_wedges(
    image_path: Path,
    output_dir: Path,
    *,
    audiveris_command,
    timeout_seconds: int,
) -> AudiverisResult | None:
    import cv2
    import numpy as np

    data = np.fromfile(str(image_path), dtype=np.uint8)
    gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    erased = erase_inter_staff_marks(gray)
    if erased is gray:
        return None
    erased_path = _save_gray_png(erased, output_dir / "no_wedges.png")
    try:
        return recognize_score(
            erased_path,
            output_dir / "no_wedges",
            audiveris_command=audiveris_command,
            timeout_seconds=timeout_seconds,
        )
    except AudiverisRecognitionError:
        return None


def recover_dropped_measures(
    image_path: Path,
    base_result: AudiverisResult,
    output_dir: Path,
    *,
    audiveris_command=None,
    timeout_seconds: int = 300,
) -> AudiverisResult | None:
    missing = missing_measure_numbers(base_result)
    if not missing:
        return None

    crops = write_selected_measure_crops(
        image_path, output_dir / "measure_crops", missing
    )
    if not crops:
        return None

    recovered: dict[int, Path] = {}
    logs: list[str] = []
    for number, crop in crops.items():
        try:
            result = recognize_score(
                crop,
                output_dir / "measure_omr" / f"m{number:02d}",
                audiveris_command=audiveris_command,
                timeout_seconds=timeout_seconds,
            )
        except AudiverisRecognitionError as exc:
            logs.append(str(exc))
            continue
        recovered[number] = result.musicxml_path
        logs.append(result.log_text)

    if not recovered:
        return None

    dest = output_dir / "recovered.musicxml"
    insert_recovered_measures(base_result.musicxml_path, recovered, dest)
    if first_part_measure_count(dest) <= first_part_measure_count(base_result.musicxml_path):
        return None

    return AudiverisResult(
        input_image=image_path,
        musicxml_path=dest,
        additional_musicxml_paths=tuple(recovered.values()),
        stdout="",
        stderr="",
        log_text="\n".join(logs),
    )


def recognize_with_recovery(
    image_path: str | Path,
    output_dir: str | Path,
    *,
    audiveris_command=None,
    timeout_seconds: int = 300,
) -> AudiverisResult:
    """Run Audiveris, then recover any measure the full-page export skipped."""
    image = Path(image_path)
    destination = Path(output_dir)
    result = recognize_score(
        image,
        destination,
        audiveris_command=audiveris_command,
        timeout_seconds=timeout_seconds,
    )
    if not needs_recovery(result):
        return result

    retry = _recognize_erased_wedges(
        image,
        destination,
        audiveris_command=audiveris_command,
        timeout_seconds=timeout_seconds,
    )
    if retry is not None and not needs_recovery(retry):
        if first_part_measure_count(retry.musicxml_path) > first_part_measure_count(
            result.musicxml_path
        ):
            return retry

    recovered = recover_dropped_measures(
        image,
        result,
        destination,
        audiveris_command=audiveris_command,
        timeout_seconds=timeout_seconds,
    )
    return recovered or result
