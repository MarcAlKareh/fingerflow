FingerFlow is a tool designed for pianists without teachers to generate fingerings for passages where good fingering choices are hard to find. It is built to generate custom fingerings based on a user's hand biology.

## Project structure

```
FingerFlow/
  frontend/   Next.js app
  backend/    FastAPI API, OMR integration and the fingering engine
    engine/     biomechanical cost model + second-order Viterbi decoder
    engine/training/   learn weights from the PIG human-fingering dataset
    omr/        Audiveris wrapper and MusicXML parser
    docs/       fingering-model.md (the mathematics), audiveris-setup.md
    tests/      pytest suite
  scripts/    CLI utilities
```

## Run locally

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

When you click **Generate Fingering** on `/upload`, the image is sent to `POST /api/preprocess`, cleaned with the OpenCV pipeline, saved under `backend/uploads/<job_id>/`, recognised by Audiveris, parsed into timed note events, fingered by the engine, and the app navigates to `/processing`. The result is `notes.json` in the job folder: every note carries `finger` and a short list of `reasons`.

### Form fields accepted by `POST /api/preprocess`

| field | meaning |
| --- | --- |
| `file` | PNG or JPG of the score (required) |
| `piece_name` | free text |
| `hand_span_left_cm`, `hand_span_right_cm` | maximum spread from little finger to thumb, in centimetres, per hand |
| `hand_span` | legacy single value applied to both hands when the per-hand fields are absent (default 20 cm) |
| `tempo_bpm` | the tempo the player intends to practise at; overrides the score's initial tempo and scales later tempo changes |
| `goal` | `expression` (favours legato connection and a settled hand) or `speed` (favours fewer hand movements and strong fingers at tempo) |

## Fingering engine

`backend/engine` replaces the original semitone-based cost function. The model works on physical keyboard geometry, signed finger-pair spans scaled to the player's hand, legato coupling from note durations, tempo through inter-onset intervals and Fitts's law, and proper chords with held-note constraints. `backend/docs/fingering-model.md` gives the full definition.

Quick use from Python:

```python
from engine import assign_fingering
result = assign_fingering(notes, hand="right", hand_span_cm=23.0, goal="expression")
result.fingers          # {note_id: finger}
result.explanations     # {note_id: ["thumb passes 3rd finger", ...]}
```

Run the tests:

```bash
cd backend
python -m pytest
```

### Learning weights from human fingerings

Download the PIG dataset (Nakamura, Saito and Yoshii; link in `engine/training/pig.py`), then:

```bash
cd backend
python -m engine.training.train --pig-dir path/to/PIG/FingeringFiles \
    --out weights_learned.json --epochs 8 --cache .feature_cache --workers 4
python -m engine.training.evaluate --pig-dir path/to/PIG/FingeringFiles --weights weights_learned.json
```

The API loads `backend/weights_learned.json` automatically if it exists (or the file named by `FINGERFLOW_WEIGHTS`); otherwise it uses the hand-tuned defaults in `engine/weights.py`.

## Sheet music preprocessing CLI

You can still run preprocessing by hand:

```bash
python scripts/preprocess_score.py input.jpg output.png --debug-dir debug
```

### What it does

- converts to grayscale
- resizes to a consistent working resolution
- boosts local contrast
- lightly denoises
- deskews the page
- crops to musical content
- binarizes to clean black and white

## Audiveris OMR

The reusable `backend.omr` module runs a preprocessed image through Audiveris
and validates the resulting MusicXML. Audiveris must be installed separately
and configured through `AUDIVERIS_CMD`.

See [backend/docs/audiveris-setup.md](backend/docs/audiveris-setup.md) for
installation, configuration, and troubleshooting instructions.

Run the complete preprocessing → Audiveris smoke test with:

```bash
python scripts/test_omr_pipeline.py "path/to/score.jpg"
```
