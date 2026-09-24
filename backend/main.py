from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from engine import Weights, assign_fingering
from omr.annotate import write_fingered_musicxml
from omr.heads import find_book_omr, parse_omr_heads
from omr.parser import parse_musicxml
from omr.recover import recognize_with_recovery

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
UPLOADS = Path(__file__).resolve().parent / "uploads"

sys.path.insert(0, str(SCRIPTS))

from preprocess_score import preprocess_bytes  # noqa: E402

UPLOADS.mkdir(parents=True, exist_ok=True)

# Optional learned weights (see engine/training/train.py). When the file
# is absent the hand-tuned defaults are used.
WEIGHTS_PATH = Path(os.environ.get("FINGERFLOW_WEIGHTS", str(Path(__file__).resolve().parent / "weights_learned.json")))
LEARNED_WEIGHTS: Optional[Weights] = Weights.load(WEIGHTS_PATH) if WEIGHTS_PATH.is_file() else None

DEFAULT_HAND_SPAN_CM = 20.0
GOALS = {"expression", "speed"}

app = FastAPI(title="FingerFlow API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/api/uploads", StaticFiles(directory=UPLOADS), name="uploads")

ACCEPTED_TYPES = {"image/png", "image/jpeg", "image/jpg"}


def _parse_span(value: str, fallback: float) -> float:
    """Hand span in centimetres: maximum spread from little finger to thumb."""
    try:
        span = float(value)
    except (TypeError, ValueError):
        return fallback
    if not (10.0 <= span <= 32.0):
        return fallback
    return span


def _parse_tempo(value: str) -> Optional[float]:
    try:
        bpm = float(value)
    except (TypeError, ValueError):
        return None
    return bpm if 10.0 <= bpm <= 400.0 else None


def _meta_path(job_id: str) -> Path:
    return UPLOADS / job_id / "meta.json"


def read_meta(job_id: str) -> dict[str, Any]:
    path = _meta_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Job not found.")
    return json.loads(path.read_text(encoding="utf-8"))


def write_meta(job_id: str, meta: dict[str, Any]) -> None:
    path = _meta_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp.replace(path)


def update_meta(job_id: str, **fields: Any) -> dict[str, Any]:
    meta = read_meta(job_id)
    meta.update(fields)
    write_meta(job_id, meta)
    return meta


def finger_hand(music_data: dict, hand: str, span_cm: float, goal: Optional[str]) -> None:
    key = f"{hand}_hand"
    notes = music_data.get(key) or []
    if not notes:
        return
    try:
        result = assign_fingering(
            notes, hand=hand, hand_span_cm=span_cm, goal=goal, weights=LEARNED_WEIGHTS, explain=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"{hand} hand optimization failed: {exc}")
        for note in notes:
            note["finger"] = None
        return
    for note in notes:
        nid = int(note["note_id"])
        note["finger"] = result.fingers.get(nid)
        note["reasons"] = result.explanations.get(nid, [])
    music_data.setdefault("engine", {})[hand] = {
        "total_cost": round(result.total_cost, 3),
        "events": len(result.events),
        "unfingered_note_ids": result.unfingered_note_ids,
    }


def run_recognition_pipeline(job_id: str) -> None:
    """OMR + fingering. Runs after the upload response is sent."""
    meta = read_meta(job_id)
    job_dir = UPLOADS / job_id
    processed_path = job_dir / "processed.png"
    audiveris_dir = job_dir / "audiveris"
    span_left = float(meta["hand_span_left_cm"])
    span_right = float(meta["hand_span_right_cm"])
    goal_choice = meta.get("goal") or "expression"
    tempo_override = meta.get("tempo_bpm")

    try:
        update_meta(job_id, status="processing", stage="omr", pipeline_error=None)
        omr_result = recognize_with_recovery(processed_path, audiveris_dir)

        update_meta(job_id, stage="parse")
        book_omr = find_book_omr(audiveris_dir)
        music_data = (
            parse_omr_heads(book_omr, processed_path, tempo_bpm=tempo_override)
            if book_omr
            else None
        )
        if music_data is None:
            music_data = parse_musicxml(omr_result.musicxml_path, tempo_bpm=tempo_override)
        music_data["settings"] = {
            "hand_span_left_cm": span_left,
            "hand_span_right_cm": span_right,
            "goal": goal_choice,
            "tempo_override_bpm": tempo_override,
        }

        update_meta(job_id, stage="fingering")
        finger_hand(music_data, "right", span_right, goal_choice)
        finger_hand(music_data, "left", span_left, goal_choice)

        notes_path = job_dir / "notes.json"
        notes_path.write_text(json.dumps(music_data, indent=2), encoding="utf-8")

        musicxml_url = None
        try:
            fingered_path = job_dir / "fingered.musicxml"
            write_fingered_musicxml(omr_result.musicxml_path, music_data, fingered_path)
            musicxml_url = f"/api/uploads/{job_id}/fingered.musicxml"
        except Exception as exc:  # noqa: BLE001
            print(f"Fingered MusicXML export failed for job {job_id}: {exc}")

        update_meta(
            job_id,
            status="complete",
            stage="complete",
            notes_url=f"/api/uploads/{job_id}/notes.json",
            musicxml_url=musicxml_url,
            pipeline_error=None,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"OMR or fingering failed for job {job_id}: {exc}")
        update_meta(
            job_id,
            status="error",
            stage="error",
            notes_url=None,
            musicxml_url=None,
            pipeline_error=str(exc),
        )


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "weights": "learned" if LEARNED_WEIGHTS else "default"}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    if not job_id.isalnum() or len(job_id) > 64:
        raise HTTPException(status_code=404, detail="Job not found.")
    return read_meta(job_id)


@app.post("/api/preprocess")
async def preprocess(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    piece_name: str = Form(""),
    hand_span: str = Form(""),
    hand_span_left_cm: str = Form(""),
    hand_span_right_cm: str = Form(""),
    tempo_bpm: str = Form(""),
    goal: str = Form("expression"),
) -> dict:
    content_type = (file.content_type or "").lower()
    if content_type not in ACCEPTED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Please upload a PNG or JPG image of your sheet music.",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        processed_bytes, deskew_angle = preprocess_bytes(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Preprocessing failed: {exc}",
        ) from exc

    # Hand spans: per-hand values win; the legacy single ``hand_span``
    # field applies to both hands; otherwise a default adult span.
    shared_span = _parse_span(hand_span, DEFAULT_HAND_SPAN_CM)
    span_left = _parse_span(hand_span_left_cm, shared_span)
    span_right = _parse_span(hand_span_right_cm, shared_span)
    goal_choice = goal.strip().lower() if goal.strip().lower() in GOALS else "expression"
    tempo_override = _parse_tempo(tempo_bpm)

    job_id = uuid.uuid4().hex
    job_dir = UPLOADS / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    original_suffix = Path(file.filename or "score.jpg").suffix.lower() or ".jpg"
    if original_suffix not in {".png", ".jpg", ".jpeg"}:
        original_suffix = ".jpg"

    original_path = job_dir / f"original{original_suffix}"
    processed_path = job_dir / "processed.png"

    original_path.write_bytes(data)
    processed_path.write_bytes(processed_bytes)

    meta = {
        "job_id": job_id,
        "status": "processing",
        "stage": "omr",
        "original_filename": file.filename,
        "piece_name": piece_name.strip(),
        "hand_span_left_cm": span_left,
        "hand_span_right_cm": span_right,
        "tempo_bpm": tempo_override,
        "goal": goal_choice,
        "deskew_angle_deg": round(deskew_angle, 4),
        "original_url": f"/api/uploads/{job_id}/original{original_suffix}",
        "processed_url": f"/api/uploads/{job_id}/processed.png",
        "notes_url": None,
        "musicxml_url": None,
        "pipeline_error": None,
    }
    write_meta(job_id, meta)
    background_tasks.add_task(run_recognition_pipeline, job_id)
    return meta
