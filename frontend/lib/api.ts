export type PreprocessResponse = {
  job_id: string;
  original_filename: string | null;
  piece_name: string;
  hand_span: string;
  goal: string;
  deskew_angle_deg: number;
  original_url: string;
  processed_url: string;
};

export const JOB_STORAGE_KEY = "fingerflow:last-job";

export async function preprocessScore(input: {
  file: File;
  pieceName: string;
  handSpan: string;
  goal: string;
}): Promise<PreprocessResponse> {
  const body = new FormData();
  body.append("file", input.file);
  body.append("piece_name", input.pieceName);
  body.append("hand_span", input.handSpan);
  body.append("goal", input.goal);

  const response = await fetch("/api/preprocess", {
    method: "POST",
    body,
  });

  if (!response.ok) {
    let detail = "Could not preprocess your score. Is the backend running?";
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) detail = payload.detail;
    } catch {
      // ignore JSON parse errors
    }
    throw new Error(detail);
  }

  return (await response.json()) as PreprocessResponse;
}

export function saveJob(job: PreprocessResponse) {
  if (typeof window === "undefined") return;
  sessionStorage.setItem(JOB_STORAGE_KEY, JSON.stringify(job));
}

export function loadJob(): PreprocessResponse | null {
  if (typeof window === "undefined") return null;
  const raw = sessionStorage.getItem(JOB_STORAGE_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as PreprocessResponse;
  } catch {
    return null;
  }
}
