"use client";

import { useCallback, useEffect, useId, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import {
  ArrowRight,
  ChevronDown,
  FileImage,
  LoaderCircle,
  Sparkles,
  Upload,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { preprocessScore, saveJob } from "@/lib/api";
import { easeOutExpo } from "@/components/landing/motion";

const HAND_SPANS = [
  { value: "7.5", label: "7.5″ — Small" },
  { value: "8.0", label: "8.0″ — Petite" },
  { value: "8.5", label: "8.5″ — Medium" },
  { value: "9.0", label: "9.0″ — Large" },
  { value: "9.5", label: "9.5″ — Extra large" },
  { value: "10.0", label: "10.0″ — Concert" },
] as const;

const OPTIMIZATION_GOALS = [
  { value: "expression", label: "Expression" },
  { value: "speed", label: "Speed" },
  { value: "endurance", label: "Endurance" },
  { value: "accuracy", label: "Technical accuracy" },
  { value: "balanced", label: "Balanced" },
] as const;

const ACCEPTED_TYPES = new Set(["image/png", "image/jpeg", "image/jpg"]);

function formatFileSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function UploadForm() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const pieceNameId = useId();
  const handSpanId = useId();
  const goalId = useId();

  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [pieceName, setPieceName] = useState("");
  const [handSpan, setHandSpan] = useState("8.5");
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
        handSpan,
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
    <motion.form
      onSubmit={onGenerate}
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.65, delay: 0.1, ease: easeOutExpo }}
      className="mx-auto w-full max-w-2xl"
    >
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
          "relative overflow-hidden rounded-2xl border bg-[#0d1016] transition-all duration-300",
          dragging
            ? "border-brand/40 bg-brand/[0.04] shadow-[0_0_0_1px_rgba(94,234,212,0.12)]"
            : "border-white/10 shadow-[0_24px_80px_-40px_rgba(0,0,0,0.8)]",
          previewUrl ? "p-3 sm:p-4" : "p-1"
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
          <div className="space-y-3">
            <div className="relative overflow-hidden rounded-xl border border-white/[0.06] bg-[#12151c]">
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
                className="absolute top-3 right-3 inline-flex size-8 items-center justify-center rounded-full border border-white/10 bg-[#0a0c10]/85 text-zinc-300 backdrop-blur transition hover:bg-[#0a0c10] hover:text-white"
                aria-label="Remove image"
              >
                <X className="size-4" />
              </button>
            </div>
            <div className="flex items-center justify-between gap-3 px-1">
              <div className="flex min-w-0 items-center gap-2.5">
                <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-brand/10 text-brand ring-1 ring-brand/20">
                  <FileImage className="size-3.5" />
                </span>
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium text-white">
                    {file.name}
                  </p>
                  <p className="text-[11px] text-zinc-500">
                    {formatFileSize(file.size)} · PNG / JPG
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => inputRef.current?.click()}
                disabled={submitting}
                className="shrink-0 text-sm text-zinc-400 transition hover:text-white disabled:opacity-50"
              >
                Replace
              </button>
            </div>
          </div>
        ) : (
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            className={cn(
              "flex w-full flex-col items-center justify-center rounded-[14px] px-6 py-16 text-center transition sm:py-20",
              dragging ? "bg-brand/[0.03]" : "hover:bg-white/[0.02]"
            )}
          >
            <span
              className={cn(
                "mb-5 flex size-14 items-center justify-center rounded-2xl ring-1 transition",
                dragging
                  ? "bg-brand/15 text-brand ring-brand/30"
                  : "bg-white/[0.04] text-zinc-300 ring-white/10"
              )}
            >
              <Upload className="size-6" />
            </span>
            <p className="text-base font-medium text-white sm:text-lg">
              {dragging ? "Drop your score here" : "Drag & drop your sheet music"}
            </p>
            <p className="mt-2 max-w-sm text-sm leading-relaxed text-zinc-500">
              PNG or JPG of a printed or handwritten page. We&apos;ll read the
              notation and prepare fingerings.
            </p>
            <span className="mt-6 inline-flex h-9 items-center rounded-full border border-white/12 bg-white/5 px-4 text-sm font-medium text-zinc-200 transition hover:border-white/20 hover:bg-white/10">
              Browse files
            </span>
          </button>
        )}
      </div>

      {error ? (
        <p className="mt-3 text-sm text-rose-400" role="alert">
          {error}
        </p>
      ) : null}

      <div className="mt-8 space-y-5">
        <div>
          <label
            htmlFor={pieceNameId}
            className="mb-2 block text-[13px] font-medium text-zinc-300"
          >
            Piece name{" "}
            <span className="font-normal text-zinc-600">(optional)</span>
          </label>
          <input
            id={pieceNameId}
            type="text"
            value={pieceName}
            onChange={(e) => setPieceName(e.target.value)}
            placeholder="e.g. Chopin Nocturne Op. 9 No. 2"
            disabled={submitting}
            className="h-11 w-full rounded-xl border border-white/10 bg-white/[0.03] px-3.5 text-sm text-white outline-none transition placeholder:text-zinc-600 focus:border-brand/40 focus:bg-white/[0.04] focus:ring-2 focus:ring-brand/15 disabled:opacity-60"
          />
        </div>

        <div className="grid gap-5 sm:grid-cols-2">
          <div>
            <label
              htmlFor={handSpanId}
              className="mb-2 block text-[13px] font-medium text-zinc-300"
            >
              Hand span
            </label>
            <div className="relative">
              <select
                id={handSpanId}
                value={handSpan}
                onChange={(e) => setHandSpan(e.target.value)}
                disabled={submitting}
                className="h-11 w-full appearance-none rounded-xl border border-white/10 bg-white/[0.03] px-3.5 pr-10 text-sm text-white outline-none transition focus:border-brand/40 focus:bg-white/[0.04] focus:ring-2 focus:ring-brand/15 disabled:opacity-60"
              >
                {HAND_SPANS.map((option) => (
                  <option
                    key={option.value}
                    value={option.value}
                    className="bg-[#111318] text-white"
                  >
                    {option.label}
                  </option>
                ))}
              </select>
              <ChevronDown className="pointer-events-none absolute top-1/2 right-3.5 size-4 -translate-y-1/2 text-zinc-500" />
            </div>
          </div>

          <div>
            <label
              htmlFor={goalId}
              className="mb-2 block text-[13px] font-medium text-zinc-300"
            >
              Optimization goal
            </label>
            <div className="relative">
              <select
                id={goalId}
                value={goal}
                onChange={(e) => setGoal(e.target.value)}
                disabled={submitting}
                className="h-11 w-full appearance-none rounded-xl border border-white/10 bg-white/[0.03] px-3.5 pr-10 text-sm text-white outline-none transition focus:border-brand/40 focus:bg-white/[0.04] focus:ring-2 focus:ring-brand/15 disabled:opacity-60"
              >
                {OPTIMIZATION_GOALS.map((option) => (
                  <option
                    key={option.value}
                    value={option.value}
                    className="bg-[#111318] text-white"
                  >
                    {option.label}
                  </option>
                ))}
              </select>
              <ChevronDown className="pointer-events-none absolute top-1/2 right-3.5 size-4 -translate-y-1/2 text-zinc-500" />
            </div>
          </div>
        </div>
      </div>

      <motion.button
        type="submit"
        whileHover={file && !submitting ? { y: -2 } : undefined}
        whileTap={file && !submitting ? { y: 0 } : undefined}
        transition={{ duration: 0.2, ease: easeOutExpo }}
        disabled={!file || submitting}
        className={cn(
          "mt-9 inline-flex h-12 w-full items-center justify-center gap-2 rounded-full text-sm font-medium transition",
          file && !submitting
            ? "bg-white text-zinc-950 hover:bg-zinc-200"
            : "cursor-not-allowed bg-white/10 text-zinc-500"
        )}
      >
        {submitting ? (
          <>
            <LoaderCircle className="size-4 animate-spin" />
            Preparing score…
          </>
        ) : (
          <>
            <Sparkles className="size-4" />
            Generate Fingering
            <ArrowRight className="size-4" />
          </>
        )}
      </motion.button>
    </motion.form>
  );
}
