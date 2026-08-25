from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

from omr.audiveris import recognize_score
from omr.parser import parse_musicxml
from engine.viterbi import run_2nd_order_viterbi
import json

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
UPLOADS = Path(__file__).resolve().parent / "uploads"

sys.path.insert(0, str(SCRIPTS))

from preprocess_score import preprocess_bytes  # noqa: E402

UPLOADS.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="FingerFlow API", version="0.1.0")

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


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/preprocess")
async def preprocess(
    file: UploadFile = File(...),
    piece_name: str = Form(""),
    hand_span: str = Form("8.5"),
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

# --- AI PIPELINE INTEGRATION ---
    notes_url = None
    audiveris_dir = job_dir / "audiveris"
    
    try:
        # 1. Run Audiveris
        omr_result = recognize_score(processed_path, audiveris_dir)
        
        # 2. Parse MusicXML
        music_data = parse_musicxml(omr_result.musicxml_path)
        
        # 3. Optimize Right Hand
        if music_data.get("right_hand"):
            try:
                rh_fingers = run_2nd_order_viterbi(music_data["right_hand"], hand="right")
                for idx, note in enumerate(music_data["right_hand"]):
                    note["finger"] = rh_fingers[idx] if idx < len(rh_fingers) else None
            except Exception as e:
                print(f"Right hand optimization failed: {e}")
                for note in music_data["right_hand"]: note["finger"] = None

        # 4. Optimize Left Hand
        if music_data.get("left_hand"):
            try:
                lh_fingers = run_2nd_order_viterbi(music_data["left_hand"], hand="left")
                for idx, note in enumerate(music_data["left_hand"]):
                    note["finger"] = lh_fingers[idx] if idx < len(lh_fingers) else None
            except Exception as e:
                print(f"Left hand optimization failed: {e}")
                for note in music_data["left_hand"]: note["finger"] = None
        
        # 5. Save and link JSON
        notes_path = job_dir / "notes.json"
        notes_path.write_text(json.dumps(music_data, indent=2), encoding="utf-8")
        notes_url = f"/api/uploads/{job_id}/notes.json"
        
    except Exception as exc:
        print(f"OMR or Parsing failed for job {job_id}: {exc}")
    # -----------------------------------

    meta = {
        "job_id": job_id,
        "original_filename": file.filename,
        "piece_name": piece_name.strip(),
        "hand_span": hand_span,
        "goal": goal,
        "deskew_angle_deg": round(deskew_angle, 4),
        "original_url": f"/api/uploads/{job_id}/original{original_suffix}",
        "processed_url": f"/api/uploads/{job_id}/processed.png",
        "notes_url": notes_url,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return meta