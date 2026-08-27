"use client";

import { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";
import {
  Check,
  Eye,
  Hand,
  Music2,
  ScanLine,
  Upload,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { easeOutExpo } from "@/components/landing/motion";

const STAGES = [
  {
    id: "upload",
    label: "Uploading image",
    detail: "Securely transferring your score",
    icon: Upload,
  },
  {
    id: "detect",
    label: "Detecting musical notation",
    detail: "Locating staves, notes, and markings",
    icon: ScanLine,
  },
  {
    id: "understand",
    label: "Understanding the musical passage",
    detail: "Parsing rhythm, phrasing, and voicing",
    icon: Music2,
  },
  {
    id: "optimize",
    label: "Optimizing fingering",
    detail: "Matching span, technique, and goals",
    icon: Hand,
  },
  {
    id: "render",
    label: "Rendering recommendations",
    detail: "Preparing confidence-aware markings",
    icon: Eye,
  },
] as const;

const TOTAL_MS = 4000;
const STAGE_MS = TOTAL_MS / STAGES.length;

type StageState = "pending" | "active" | "complete";

function getStageState(index: number, activeIndex: number): StageState {
  if (index < activeIndex) return "complete";
  if (index === activeIndex) return "active";
  return "pending";
}

export function ProcessingStatus() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const jobId = searchParams.get("job_id");
  const [activeIndex, setActiveIndex] = useState(0);

  useEffect(() => {
    const timers = STAGES.map((_, index) =>
      window.setTimeout(() => setActiveIndex(index), index * STAGE_MS)
    );

    const navigateTimer = window.setTimeout(() => {
      if (jobId) {
        router.push(`/results/${jobId}`);
      } else {
        router.push("/results"); // Fallback just in case
      }
    }, TOTAL_MS);

    return () => {
      timers.forEach((id) => window.clearTimeout(id));
      window.clearTimeout(navigateTimer);
    };
  }, [router, jobId]);

  const progress = ((activeIndex + 1) / STAGES.length) * 100;
  const ActiveIcon = STAGES[activeIndex]?.icon ?? Upload;

  return (
    <div className="relative mx-auto w-full max-w-lg">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: easeOutExpo }}
        className="text-center"
      >
        <p className="mb-5 text-sm font-medium tracking-[0.22em] text-brand uppercase">
          Analysis
        </p>
        <h1 className="text-balance text-2xl font-semibold tracking-tight text-white sm:text-3xl">
          Generating fingerings
        </h1>
        <p className="mx-auto mt-3 max-w-md text-sm leading-relaxed text-zinc-400 sm:text-base">
          FingerFlow is reading your score and shaping markings for your hands.
        </p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.65, delay: 0.1, ease: easeOutExpo }}
        className="mt-10 overflow-hidden rounded-2xl border border-white/10 bg-[#0d1016] shadow-[0_24px_80px_-40px_rgba(0,0,0,0.8)]"
      >
        <div className="flex items-center gap-4 border-b border-white/[0.06] px-5 py-4">
          <div className="relative flex size-11 shrink-0 items-center justify-center overflow-hidden rounded-xl bg-brand/15 ring-1 ring-brand/25">
            <AnimatePresence mode="wait">
              <motion.span
                key={STAGES[activeIndex]?.id}
                initial={{ opacity: 0, scale: 0.7, y: 6 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.7, y: -6 }}
                transition={{ duration: 0.28, ease: easeOutExpo }}
                className="absolute inset-0 flex items-center justify-center"
              >
                <ActiveIcon className="size-5 text-brand" />
              </motion.span>
            </AnimatePresence>
            <motion.span
              aria-hidden
              className="absolute inset-0 rounded-xl bg-brand/10"
              animate={{ opacity: [0.35, 0.7, 0.35] }}
              transition={{ duration: 1.6, repeat: Infinity, ease: "easeInOut" }}
            />
          </div>

          <div className="min-w-0 flex-1 text-left">
            <AnimatePresence mode="wait">
              <motion.div
                key={STAGES[activeIndex]?.id}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.28, ease: easeOutExpo }}
              >
                <p className="truncate text-sm font-medium text-white">
                  {STAGES[activeIndex]?.label}
                </p>
                <p className="mt-0.5 truncate text-[12px] text-zinc-500">
                  {STAGES[activeIndex]?.detail}
                </p>
              </motion.div>
            </AnimatePresence>
          </div>

          <span className="font-mono text-[11px] tabular-nums text-zinc-500">
            {activeIndex + 1}/{STAGES.length}
          </span>
        </div>

        <div className="h-1 bg-white/[0.04]">
          <motion.div
            className="h-full origin-left bg-gradient-to-r from-teal-500 to-teal-300"
            initial={{ scaleX: 0 }}
            animate={{ scaleX: progress / 100 }}
            transition={{ duration: 0.55, ease: easeOutExpo }}
          />
        </div>

        <ol className="space-y-1 p-3 sm:p-4">
          {STAGES.map((stage, index) => {
            const state = getStageState(index, activeIndex);
            const Icon = stage.icon;

            return (
              <motion.li
                key={stage.id}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{
                  duration: 0.45,
                  delay: 0.15 + index * 0.05,
                  ease: easeOutExpo,
                }}
              >
                <div
                  className={cn(
                    "flex items-center gap-3 rounded-xl px-3 py-2.5 transition-colors duration-300",
                    state === "active" && "bg-brand/[0.07] ring-1 ring-brand/20",
                    state === "complete" && "bg-transparent",
                    state === "pending" && "opacity-45"
                  )}
                >
                  <div className="relative flex size-7 shrink-0 items-center justify-center">
                    <AnimatePresence mode="wait" initial={false}>
                      {state === "complete" ? (
                        <motion.span
                          key="check"
                          initial={{ scale: 0.5, opacity: 0 }}
                          animate={{ scale: 1, opacity: 1 }}
                          exit={{ scale: 0.5, opacity: 0 }}
                          transition={{ duration: 0.25, ease: easeOutExpo }}
                          className="flex size-7 items-center justify-center rounded-full bg-teal-500/15 text-teal-400 ring-1 ring-teal-500/25"
                        >
                          <Check className="size-3.5" strokeWidth={2.5} />
                        </motion.span>
                      ) : state === "active" ? (
                        <motion.span
                          key="active"
                          initial={{ scale: 0.7, opacity: 0 }}
                          animate={{ scale: 1, opacity: 1 }}
                          exit={{ scale: 0.7, opacity: 0 }}
                          transition={{ duration: 0.25, ease: easeOutExpo }}
                          className="relative flex size-7 items-center justify-center rounded-full bg-brand/15 text-brand ring-1 ring-brand/30"
                        >
                          <Icon className="size-3.5" />
                          <motion.span
                            aria-hidden
                            className="absolute inset-0 rounded-full ring-1 ring-brand/40"
                            animate={{ scale: [1, 1.35], opacity: [0.6, 0] }}
                            transition={{
                              duration: 1.2,
                              repeat: Infinity,
                              ease: "easeOut",
                            }}
                          />
                        </motion.span>
                      ) : (
                        <motion.span
                          key="pending"
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          exit={{ opacity: 0 }}
                          className="flex size-7 items-center justify-center rounded-full bg-white/[0.03] text-zinc-600 ring-1 ring-white/[0.06]"
                        >
                          <Icon className="size-3.5" />
                        </motion.span>
                      )}
                    </AnimatePresence>
                  </div>

                  <div className="min-w-0 flex-1">
                    <p
                      className={cn(
                        "text-[13px] font-medium transition-colors duration-300",
                        state === "active" && "text-white",
                        state === "complete" && "text-zinc-300",
                        state === "pending" && "text-zinc-500"
                      )}
                    >
                      {stage.label}
                    </p>
                  </div>

                  {state === "active" ? (
                    <motion.span
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="flex items-center gap-1"
                    >
                      {[0, 1, 2].map((dot) => (
                        <motion.span
                          key={dot}
                          className="size-1 rounded-full bg-brand"
                          animate={{ opacity: [0.25, 1, 0.25] }}
                          transition={{
                            duration: 0.9,
                            repeat: Infinity,
                            delay: dot * 0.18,
                            ease: "easeInOut",
                          }}
                        />
                      ))}
                    </motion.span>
                  ) : null}
                </div>
              </motion.li>
            );
          })}
        </ol>
      </motion.div>

      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5, duration: 0.5 }}
        className="mt-6 text-center text-[12px] text-zinc-600"
      >
        This usually takes a few seconds
      </motion.p>
    </div>
  );
}
