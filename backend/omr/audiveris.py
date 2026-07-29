"""Run Audiveris in batch mode and validate its MusicXML output.

This module deliberately knows nothing about FastAPI or FingerFlow's eventual
note-event JSON. It accepts a prepared image and returns validated MusicXML,
which keeps the OMR boundary easy to test and replace later.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

MUSICXML_ROOTS = {"score-partwise", "score-timewise", "opus"}
PLAIN_XML_CONSTANT = "org.audiveris.omr.sheet.BookManager.useCompression=false"


class AudiverisError(RuntimeError):
    """Base error for failures in the Audiveris integration."""


class AudiverisNotFoundError(AudiverisError):
    """Raised when no usable Audiveris launcher can be located."""


class AudiverisRecognitionError(AudiverisError):
    """Raised when Audiveris cannot recognize or export a score."""


class InvalidMusicXMLError(AudiverisError):
    """Raised when Audiveris output is missing or is not valid MusicXML."""


@dataclass(frozen=True)
class AudiverisResult:
    """The output and useful diagnostics from one recognition run."""

    input_image: Path
    musicxml_path: Path
    additional_musicxml_paths: tuple[Path, ...]
    stdout: str
    stderr: str


def resolve_audiveris_command(command: str | Path | None = None) -> Path:
    """Find the Audiveris launcher from an argument, environment, or PATH."""
    configured = str(command or os.environ.get("AUDIVERIS_CMD", "")).strip()
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return path.resolve()

        discovered = shutil.which(configured)
        if discovered:
            return Path(discovered).resolve()

        raise AudiverisNotFoundError(
            f"Configured Audiveris launcher does not exist: {configured}"
        )

    for name in ("Audiveris", "Audiveris.bat", "audiveris"):
        discovered = shutil.which(name)
        if discovered:
            return Path(discovered).resolve()

    if os.name == "nt":
        # The official Windows MSI installs here by default but does not
        # necessarily add Audiveris to PATH.
        for path in (
            Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
            / "Audiveris"
            / "Audiveris.exe",
            Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
            / "Audiveris"
            / "bin"
            / "Audiveris.bat",
        ):
            if path.is_file():
                return path.resolve()

    raise AudiverisNotFoundError(
        "Audiveris was not found. Set AUDIVERIS_CMD to the Audiveris launcher "
        "(for example, C:\\Program Files\\Audiveris\\Audiveris.exe)."
    )


def _subprocess_command(executable: Path, arguments: list[str]) -> list[str]:
    """Build a subprocess command that also handles Windows batch launchers."""
    command = [str(executable), *arguments]
    if os.name == "nt" and executable.suffix.lower() in {".bat", ".cmd"}:
        # Batch files must run through cmd.exe. list2cmdline preserves spaces.
        return ["cmd.exe", "/d", "/s", "/c", subprocess.list2cmdline(command)]
    return command


def _xml_root_name(data: bytes) -> str:
    """Return an XML document's root tag without its optional namespace."""
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise InvalidMusicXMLError(f"MusicXML is not well-formed: {exc}") from exc
    return root.tag.rsplit("}", 1)[-1]


def _read_mxl_score(path: Path) -> bytes:
    """Read the main MusicXML document from a compressed .mxl archive."""
    try:
        with zipfile.ZipFile(path) as archive:
            candidates: list[str] = []

            # The container file is the authoritative pointer when available.
            if "META-INF/container.xml" in archive.namelist():
                container = ET.fromstring(archive.read("META-INF/container.xml"))
                for element in container.iter():
                    if element.tag.rsplit("}", 1)[-1] == "rootfile":
                        full_path = element.attrib.get("full-path")
                        if full_path:
                            candidates.append(full_path)

            candidates.extend(
                name
                for name in archive.namelist()
                if name.lower().endswith((".xml", ".musicxml"))
                and not name.upper().startswith("META-INF/")
            )

            for name in dict.fromkeys(candidates):
                if name in archive.namelist():
                    return archive.read(name)
    except (zipfile.BadZipFile, ET.ParseError, KeyError) as exc:
        raise InvalidMusicXMLError(f"Invalid compressed MusicXML: {exc}") from exc

    raise InvalidMusicXMLError(f"No score XML document found inside {path}")


def validate_musicxml(path: Path) -> None:
    """Verify that a plain XML or compressed MXL file contains a score."""
    if not path.is_file() or path.stat().st_size == 0:
        raise InvalidMusicXMLError(f"MusicXML output is missing or empty: {path}")

    data = _read_mxl_score(path) if path.suffix.lower() == ".mxl" else path.read_bytes()
    root_name = _xml_root_name(data)
    if root_name not in MUSICXML_ROOTS:
        raise InvalidMusicXMLError(
            f"Unexpected MusicXML root element <{root_name}> in {path}"
        )


def _find_exports(output_dir: Path) -> list[Path]:
    """Find MusicXML exports, including Audiveris movement subfolders."""
    candidates = [
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".xml", ".musicxml", ".mxl"}
    ]

    valid: list[Path] = []
    for path in sorted(candidates):
        try:
            validate_musicxml(path)
        except InvalidMusicXMLError:
            continue
        valid.append(path)
    return valid


def recognize_score(
    image_path: str | Path,
    output_dir: str | Path,
    *,
    audiveris_command: str | Path | None = None,
    timeout_seconds: int = 300,
) -> AudiverisResult:
    """Recognize a preprocessed score image and export validated MusicXML.

    Audiveris can create multiple MusicXML files when it detects multiple
    movements. The first export is exposed as ``musicxml_path`` and any others
    are retained in ``additional_musicxml_paths`` for later pipeline stages.
    """
    image = Path(image_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()

    if not image.is_file():
        raise AudiverisRecognitionError(f"Input image does not exist: {image}")
    if image.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        raise AudiverisRecognitionError(
            f"Unsupported image type '{image.suffix}'. Use PNG, JPG, or TIFF."
        )
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")

    executable = resolve_audiveris_command(audiveris_command)
    destination.mkdir(parents=True, exist_ok=True)

    arguments = [
        "-batch",
        "-export",
        "-constant",
        PLAIN_XML_CONSTANT,
        "-output",
        str(destination),
        "--",
        str(image),
    ]

    try:
        completed = subprocess.run(
            _subprocess_command(executable, arguments),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudiverisRecognitionError(
            f"Audiveris timed out after {timeout_seconds} seconds."
        ) from exc
    except OSError as exc:
        raise AudiverisRecognitionError(
            f"Could not start Audiveris: {exc}"
        ) from exc

    exports = _find_exports(destination)
    if completed.returncode != 0 or not exports:
        details = (completed.stderr or completed.stdout).strip()
        if len(details) > 2000:
            details = details[-2000:]
        message = (
            f"Audiveris recognition failed (exit code {completed.returncode})."
        )
        if details:
            message += f"\n{details}"
        elif not exports:
            message += "\nNo valid MusicXML file was produced."
        raise AudiverisRecognitionError(message)

    return AudiverisResult(
        input_image=image,
        musicxml_path=exports[0],
        additional_musicxml_paths=tuple(exports[1:]),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
