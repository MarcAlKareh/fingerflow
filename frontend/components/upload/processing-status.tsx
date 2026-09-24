"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { fetchJob, saveJob, type JobStage } from "@/lib/api";

const STAGES = [
  { id: "upload", label: "Uploading the image" },
  { id: "detect", label: "Reading the notation" },
  { id: "understand", label: "Parsing the passage" },
  { id: "optimize", label: "Assigning fingerings" },
  { id: "render", label: "Placing the numbers" },
] as const;

function stageIndexFromJob(stage: JobStage | null, status: string | null): number {
  if (status === "complete" || stage === "complete") return STAGES.length - 1;
  if (status === "error" || stage === "error") return 1;
  switch (stage) {
    case "omr":
      return 1;
    case "parse":
      return 2;
    case "fingering":
      return 3;
    default:
      return 1;
  }
}

export function ProcessingStatus() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const jobId = searchParams.get("job");

  const [activeIndex, setActiveIndex] = useState(1);
  const [error, setError] = useState<string | null>(null);
  const [missingJob, setMissingJob] = useState(!jobId);

  useEffect(() => {
    if (!jobId) {
      setMissingJob(true);
      return;
    }

    let cancelled = false;
    let intervalId = 0;

    const poll = async () => {
      try {
        const job = await fetchJob(jobId);
        if (cancelled) return;
        saveJob(job);
        setActiveIndex(stageIndexFromJob(job.stage, job.status));

        if (job.status === "complete") {
          window.clearInterval(intervalId);
          router.push(`/results?job=${job.job_id}`);
          return;
        }

        if (job.status === "error") {
          window.clearInterval(intervalId);
          setError(
            job.pipeline_error ||
              "Recognition failed. Try a clearer photo of the score."
          );
        }
      } catch (err) {
        if (cancelled) return;
        window.clearInterval(intervalId);
        setError(
          err instanceof Error ? err.message : "Could not check job status."
        );
      }
    };

    void poll();
    intervalId = window.setInterval(poll, 1500);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [jobId, router]);

  if (missingJob) {
    return (
      <div className="mx-auto max-w-md text-center">
        <h1 className="text-2xl text-foreground">No score in progress</h1>
        <p className="mt-3 text-sm text-muted-foreground">
          Upload a page of sheet music to generate fingerings.
        </p>
        <Link
          href="/upload"
          className="mt-6 inline-flex h-9 items-center rounded-md bg-foreground px-4 text-sm text-background"
        >
          Go to upload
        </Link>
      </div>
    );
  }

  return (
    <div className="mx-auto w-full max-w-md">
      <h1 className="text-2xl text-foreground sm:text-3xl">
        {error ? "Could not finish" : "Reading the score"}
      </h1>
      <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
        {error
          ? error
          : "This usually takes a minute or two. Fingerings are written onto the page you uploaded."}
      </p>

      {error ? (
        <div className="mt-8 flex flex-wrap gap-3">
          <Link
            href="/upload"
            className="inline-flex h-9 items-center rounded-md bg-foreground px-4 text-sm text-background"
          >
            Try another image
          </Link>
          <Link
            href="/"
            className="inline-flex h-9 items-center rounded-md border border-border px-4 text-sm text-foreground"
          >
            Home
          </Link>
        </div>
      ) : (
        <ol className="mt-8 space-y-2">
          {STAGES.map((stage, index) => {
            const done = index < activeIndex;
            const active = index === activeIndex;
            return (
              <li
                key={stage.id}
                className={
                  active
                    ? "text-sm text-foreground"
                    : done
                      ? "text-sm text-muted-foreground"
                      : "text-sm text-muted-foreground/60"
                }
              >
                {done ? "✓" : active ? "·" : " "} {stage.label}
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}
