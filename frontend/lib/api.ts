export type JobStatus = "processing" | "complete" | "error";
export type JobStage = "omr" | "parse" | "fingering" | "complete" | "error";

export type JobMeta = {
  job_id: string;
  status: JobStatus;
  stage: JobStage;
  original_filename: string | null;
  piece_name: string;
  hand_span_left_cm: number;
  hand_span_right_cm: number;
  tempo_bpm: number | null;
  goal: string;
  deskew_angle_deg: number;
  original_url: string;
  processed_url: string;
  notes_url: string | null;
  musicxml_url: string | null;
  pipeline_error: string | null;
};

export type ScoreNote = {
  note_id: number;
  pitch: number;
  start_time_sec: number;
  duration_sec: number;
  measure?: number | null;
  finger?: number | null;
  reasons?: string[];
  x?: number;
  y?: number;
};

export type NotesPayload = {
  tempo_bpm?: number;
  score_tempo_bpm?: number;
  hand_split?: string;
  right_hand: ScoreNote[];
  left_hand: ScoreNote[];
  settings?: {
    hand_span_left_cm?: number;
    hand_span_right_cm?: number;
    goal?: string;
    tempo_override_bpm?: number | null;
  };
  engine?: Record<string, unknown>;
};

export const JOB_STORAGE_KEY = "fingerflow:last-job";

async function readError(response: Response, fallback: string): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: string };
    if (payload.detail) return payload.detail;
  } catch {
    // ignore JSON parse errors
  }
  return fallback;
}

export async function preprocessScore(input: {
  file: File;
  pieceName: string;
  handSpanLeftCm: string;
  handSpanRightCm: string;
  tempoBpm: string;
  goal: string;
}): Promise<JobMeta> {
  const body = new FormData();
  body.append("file", input.file);
  body.append("piece_name", input.pieceName);
  body.append("hand_span_left_cm", input.handSpanLeftCm);
  body.append("hand_span_right_cm", input.handSpanRightCm);
  body.append("tempo_bpm", input.tempoBpm);
  body.append("goal", input.goal);

  const response = await fetch("/api/preprocess", {
    method: "POST",
    body,
  });

  if (!response.ok) {
    throw new Error(
      await readError(
        response,
        "Could not preprocess your score. Is the backend running?"
      )
    );
  }

  return (await response.json()) as JobMeta;
}

export async function fetchJob(jobId: string): Promise<JobMeta> {
  const response = await fetch(`/api/jobs/${jobId}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(await readError(response, "Job not found."));
  }
  return (await response.json()) as JobMeta;
}

export async function fetchNotes(notesUrl: string): Promise<NotesPayload> {
  const response = await fetch(notesUrl, { cache: "no-store" });
  if (!response.ok) {
    throw new Error("Could not load fingerings for this score.");
  }
  return (await response.json()) as NotesPayload;
}

export function saveJob(job: JobMeta) {
  if (typeof window === "undefined") return;
  sessionStorage.setItem(JOB_STORAGE_KEY, JSON.stringify(job));
}

export function loadJob(): JobMeta | null {
  if (typeof window === "undefined") return null;
  const raw = sessionStorage.getItem(JOB_STORAGE_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as JobMeta;
  } catch {
    return null;
  }
}
