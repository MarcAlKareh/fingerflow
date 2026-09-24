"use client";

import { useCallback, useEffect, useId, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { ChevronDown, FileImage, LoaderCircle, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { preprocessScore, saveJob } from "@/lib/api";

const HAND_SPANS = [
  { value: "18", label: "18 cm — Small" },
  { value: "20", label: "20 cm — Medium" },
  { value: "21.5", label: "21.5 cm — Average" },
  { value: "23", label: "23 cm — Large" },
  { value: "25", label: "25 cm — Extra large" },
] as const;

const OPTIMIZATION_GOALS = [
  { value: "expression", label: "Expression" },
  { value: "speed", label: "Speed" },
] as const;

const ACCEPTED_TYPES = new Set(["image/png", "image/jpeg", "image/jpg"]);

function formatFileSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

const fieldClass =
  "h-10 w-full appearance-none rounded-md border border-border bg-card px-3 text-sm text-foreground outline-none placeholder:text-muted-foreground focus:border-foreground disabled:opacity-60";

export function UploadForm() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const pieceNameId = useId();
  const leftSpanId = useId();
  const rightSpanId = useId();
  const tempoId = useId();
  const goalId = useId();

  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [pieceName, setPieceName] = useState("");
  const [handSpanLeft, setHandSpanLeft] = useState("20");
  const [handSpanRight, setHandSpanRight] = useState("20");
  const [tempoBpm, setTempoBpm] = useState("");
  const [goal, setGoal] = useState("expression");
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const assignFile = useCallback((next: File | null) => {
    setError(null);

    if (!next) {
      setFile(null);
      setPreviewUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return null;
      });
      return;
    }

    if (!ACCEPTED_TYPES.has(next.type)) {
      setError("Please upload a PNG or JPG image of your sheet music.");
      return;
    }

    setFile(next);
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return URL.createObjectURL(next);
    });

    const inferred = next.name
      .replace(/\.[^.]+$/, "")
      .replace(/[-_]+/g, " ")
      .trim();
    setPieceName((current) => (current.trim() ? current : inferred));
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      setDragging(false);
      const dropped = event.dataTransfer.files?.[0];
      if (dropped) assignFile(dropped);
    },
    [assignFile]
  );

  const onGenerate = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!file || submitting) {
      if (!file) setError("Add a sheet music image to continue.");
      return;
    }

    setSubmitting(true);
    setError(null);

    try {
      const job = await preprocessScore({
        file,
        pieceName,
        handSpanLeftCm: handSpanLeft,
        handSpanRightCm: handSpanRight,
        tempoBpm,
        goal,
      });
      saveJob(job);
      router.push(`/processing?job=${job.job_id}`);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not preprocess your score. Is the backend running?"
      );
      setSubmitting(false);
    }
  };

  return (
    <form onSubmit={onGenerate} className="w-full">
      <div
        onDragEnter={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={(e) => {
          e.preventDefault();
          if (e.currentTarget.contains(e.relatedTarget as Node)) return;
          setDragging(false);
        }}
        onDrop={onDrop}
        className={cn(
          "rounded-md border bg-card",
          dragging ? "border-foreground" : "border-border"
        )}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg,.png,.jpg,.jpeg"
          className="sr-only"
          onChange={(e) => assignFile(e.target.files?.[0] ?? null)}
        />

        {previewUrl && file ? (
          <div className="p-3">
            <div className="relative overflow-hidden rounded border border-border bg-muted">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={previewUrl}
                alt="Sheet music preview"
                className="mx-auto max-h-[420px] w-full object-contain"
              />
              <button
                type="button"
                onClick={() => {
                  assignFile(null);
                  if (inputRef.current) inputRef.current.value = "";
                }}
                className="absolute top-2 right-2 inline-flex size-8 items-center justify-center rounded-md border border-border bg-card text-foreground"
                aria-label="Remove image"
              >
                <X className="size-4" />
              </button>
            </div>
            <div className="mt-3 flex items-center justify-between gap-3">
              <div className="flex min-w-0 items-center gap-2">
                <FileImage className="size-4 shrink-0 text-muted-foreground" />
                <div className="min-w-0">
                  <p className="truncate text-sm text-foreground">{file.name}</p>
                  <p className="text-xs text-muted-foreground">
                    {formatFileSize(file.size)}
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => inputRef.current?.click()}
                disabled={submitting}
                className="shrink-0 text-sm text-muted-foreground hover:text-foreground disabled:opacity-50"
              >
                Replace
              </button>
            </div>
          </div>
        ) : (
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            className="flex w-full flex-col items-center justify-center px-6 py-14 text-center"
          >
            <p className="text-sm font-medium text-foreground">
              {dragging ? "Drop the page here" : "Drop a photo of the page"}
            </p>
            <p className="mt-2 max-w-sm text-sm leading-relaxed text-muted-foreground">
              PNG or JPG. A clear printed page works better than a dark phone
              shot.
            </p>
            <span className="mt-5 inline-flex h-8 items-center rounded-md border border-border px-3 text-sm text-foreground">
              Choose file
            </span>
          </button>
        )}
      </div>

      {error ? (
        <p className="mt-3 text-sm text-destructive" role="alert">
          {error}
        </p>
      ) : null}

      <div className="mt-8 space-y-5">
        <div>
          <label htmlFor={pieceNameId} className="mb-1.5 block text-sm text-foreground">
            Piece name{" "}
            <span className="text-muted-foreground">(optional)</span>
          </label>
          <input
            id={pieceNameId}
            type="text"
            value={pieceName}
            onChange={(e) => setPieceName(e.target.value)}
            placeholder="e.g. Chopin Op. 10 No. 4"
            disabled={submitting}
            className={fieldClass}
          />
        </div>

        <div className="grid gap-5 sm:grid-cols-2">
          <div>
            <label htmlFor={leftSpanId} className="mb-1.5 block text-sm text-foreground">
              Left-hand span
            </label>
            <div className="relative">
              <select
                id={leftSpanId}
                value={handSpanLeft}
                onChange={(e) => setHandSpanLeft(e.target.value)}
                disabled={submitting}
                className={cn(fieldClass, "pr-9")}
              >
                {HAND_SPANS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
              <ChevronDown className="pointer-events-none absolute top-1/2 right-3 size-4 -translate-y-1/2 text-muted-foreground" />
            </div>
          </div>

          <div>
            <label htmlFor={rightSpanId} className="mb-1.5 block text-sm text-foreground">
              Right-hand span
            </label>
            <div className="relative">
              <select
                id={rightSpanId}
                value={handSpanRight}
                onChange={(e) => setHandSpanRight(e.target.value)}
                disabled={submitting}
                className={cn(fieldClass, "pr-9")}
              >
                {HAND_SPANS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
              <ChevronDown className="pointer-events-none absolute top-1/2 right-3 size-4 -translate-y-1/2 text-muted-foreground" />
            </div>
          </div>
        </div>

        <div className="grid gap-5 sm:grid-cols-2">
          <div>
            <label htmlFor={tempoId} className="mb-1.5 block text-sm text-foreground">
              Practice tempo{" "}
              <span className="text-muted-foreground">(optional)</span>
            </label>
            <input
              id={tempoId}
              type="number"
              min={10}
              max={400}
              value={tempoBpm}
              onChange={(e) => setTempoBpm(e.target.value)}
              placeholder="BPM"
              disabled={submitting}
              className={fieldClass}
            />
          </div>

          <div>
            <label htmlFor={goalId} className="mb-1.5 block text-sm text-foreground">
              Optimization goal
            </label>
            <div className="relative">
              <select
                id={goalId}
                value={goal}
                onChange={(e) => setGoal(e.target.value)}
                disabled={submitting}
                className={cn(fieldClass, "pr-9")}
              >
                {OPTIMIZATION_GOALS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
              <ChevronDown className="pointer-events-none absolute top-1/2 right-3 size-4 -translate-y-1/2 text-muted-foreground" />
            </div>
          </div>
        </div>
      </div>

      <button
        type="submit"
        disabled={!file || submitting}
        className={cn(
          "mt-8 inline-flex h-10 w-full items-center justify-center rounded-md text-sm",
          file && !submitting
            ? "bg-foreground text-background"
            : "cursor-not-allowed bg-muted text-muted-foreground"
        )}
      >
        {submitting ? (
          <>
            <LoaderCircle className="mr-2 size-4 animate-spin" />
            Preparing score…
          </>
        ) : (
          "Generate fingering"
        )}
      </button>
    </form>
  );
}
