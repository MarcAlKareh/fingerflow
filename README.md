FingerFlow is a tool designed for pianists without teachers to generate fingerings for passages where good fingering choices are hard to find. It is built to generate custom fingerings based on a user's hand biology.

## Project structure

```
FingerFlow/
  frontend/   Next.js app
  backend/    FastAPI API (image preprocess for now)
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

When you click **Generate Fingering** on `/upload`, the image is sent to `POST /api/preprocess`, cleaned with the OpenCV pipeline, saved under `backend/uploads/<job_id>/`, then the app navigates to `/processing`.

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
