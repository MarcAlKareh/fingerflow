"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { fetchJob, fetchNotes, type JobMeta, type NotesPayload } from "@/lib/api";
import { cn } from "@/lib/utils";
import { ScoreOverlay } from "./score-overlay";
import { ScoreRenderer } from "./score-renderer";

export function ResultsView() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const jobId = searchParams.get("job");
  const [job, setJob] = useState<JobMeta | null>(null);
  const [notes, setNotes] = useState<NotesPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const hasOverlay = Boolean(
    (notes?.right_hand ?? []).some((note) => note.x != null) ||
      (notes?.left_hand ?? []).some((note) => note.x != null)
  );
  const [view, setView] = useState<"score" | "engraved">("score");

  useEffect(() => {
    if (!jobId) {
      setLoading(false);
      setError("No job specified.");
      return;
    }

    let cancelled = false;

    const load = async () => {
      try {
        const meta = await fetchJob(jobId);
        if (cancelled) return;
        setJob(meta);

        if (meta.status === "processing") {
          router.replace(`/processing?job=${meta.job_id}`);
          return;
        }

        if (meta.status === "error" || !meta.notes_url) {
          setError(
            meta.pipeline_error || "No fingerings were produced for this score."
          );
          setLoading(false);
          return;
        }

        const payload = await fetchNotes(meta.notes_url);
        if (cancelled) return;
        setNotes(payload);
        setLoading(false);
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Could not load results.");
        setLoading(false);
      }
    };

    void load();
    return () => {
      cancelled = true;
    };
  }, [jobId, router]);

  if (loading) {
    return (
      <div className="mx-auto max-w-xl px-6 py-20">
        <h1 className="text-2xl text-foreground">Loading fingerings…</h1>
      </div>
    );
  }

  if (error || !job) {
    return (
      <div className="mx-auto max-w-xl px-6 py-20">
        <h1 className="text-2xl text-foreground">Nothing to show yet</h1>
        <p className="mt-3 text-sm text-muted-foreground">{error}</p>
        <Link
          href="/upload"
          className="mt-6 inline-flex h-9 items-center rounded-md bg-foreground px-4 text-sm text-background"
        >
          Upload a score
        </Link>
      </div>
    );
  }

  const title = job.piece_name || "Untitled score";

  return (
    <div className="mx-auto max-w-5xl px-6 pt-12 pb-16 sm:pt-16">
      <div className="mb-8 flex flex-col gap-5 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-3xl text-foreground sm:text-4xl">{title}</h1>
          <p className="mt-3 text-sm text-muted-foreground">
            {job.goal === "speed" ? "Speed" : "Expression"} · LH {job.hand_span_left_cm}{" "}
            cm · RH {job.hand_span_right_cm} cm
            {job.tempo_bpm ? ` · ${job.tempo_bpm} BPM` : ""}
          </p>
        </div>
        <div className="flex flex-wrap gap-3">
          <Link
            href="/upload"
            className="inline-flex h-9 items-center rounded-md bg-foreground px-3 text-sm text-background"
          >
            Another score
          </Link>
          <Link
            href="/"
            className="inline-flex h-9 items-center rounded-md border border-border px-3 text-sm text-foreground"
          >
            Home
          </Link>
        </div>
      </div>

      <div className="overflow-hidden rounded-md border border-border bg-card p-3">
        {hasOverlay && job.musicxml_url ? (
          <div className="mb-3 flex gap-4 border-b border-border pb-2 text-sm">
            <button
              type="button"
              onClick={() => setView("score")}
              className={cn(
                view === "score"
                  ? "text-foreground"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              Your score
            </button>
            <button
              type="button"
              onClick={() => setView("engraved")}
              className={cn(
                view === "engraved"
                  ? "text-foreground"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              Digital copy
            </button>
          </div>
        ) : null}

        {hasOverlay && view === "score" ? (
          <ScoreOverlay
            imageUrl={job.processed_url || job.original_url}
            rightHand={notes?.right_hand ?? []}
            leftHand={notes?.left_hand ?? []}
          />
        ) : job.musicxml_url ? (
          <ScoreRenderer musicxmlUrl={job.musicxml_url} />
        ) : (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={job.original_url}
            alt="Uploaded sheet music"
            className="mx-auto max-h-[720px] w-full object-contain"
          />
        )}
      </div>
    </div>
  );
}
