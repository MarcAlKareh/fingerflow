from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from engine import Weights, assign_fingering
from omr.audiveris import recognize_score
from omr.parser import parse_musicxml

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


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "weights": "learned" if LEARNED_WEIGHTS else "default"}


@app.post("/api/preprocess")
async def preprocess(
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
    meta_path = job_dir / "meta.json"

    original_path.write_bytes(data)
    processed_path.write_bytes(processed_bytes)

    # --- OMR + fingering pipeline ---
    notes_url = None
    pipeline_error = None
    audiveris_dir = job_dir / "audiveris"

    try:
        omr_result = recognize_score(processed_path, audiveris_dir)
        music_data = parse_musicxml(omr_result.musicxml_path, tempo_bpm=tempo_override)
        music_data["settings"] = {
            "hand_span_left_cm": span_left,
            "hand_span_right_cm": span_right,
            "goal": goal_choice,
            "tempo_override_bpm": tempo_override,
        }
        finger_hand(music_data, "right", span_right, goal_choice)
        finger_hand(music_data, "left", span_left, goal_choice)

        notes_path = job_dir / "notes.json"
        notes_path.write_text(json.dumps(music_data, indent=2), encoding="utf-8")
        notes_url = f"/api/uploads/{job_id}/notes.json"
    except Exception as exc:  # noqa: BLE001
        pipeline_error = str(exc)
        print(f"OMR or fingering failed for job {job_id}: {exc}")

    meta = {
        "job_id": job_id,
        "original_filename": file.filename,
        "piece_name": piece_name.strip(),
        "hand_span_left_cm": span_left,
        "hand_span_right_cm": span_right,
        "tempo_bpm": tempo_override,
        "goal": goal_choice,
        "deskew_angle_deg": round(deskew_angle, 4),
        "original_url": f"/api/uploads/{job_id}/original{original_suffix}",
        "processed_url": f"/api/uploads/{job_id}/processed.png",
        "notes_url": notes_url,
        "pipeline_error": pipeline_error,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return meta
