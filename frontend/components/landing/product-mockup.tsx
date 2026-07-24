"use client";

import { motion } from "framer-motion";
import {
  Check,
  ChevronDown,
  Hand,
  Maximize2,
  Minus,
  Plus,
  RefreshCw,
  Sparkles,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { easeOutExpo } from "./motion";

type Confidence = "high" | "medium" | "review";

type FingerMark = {
  x: number;
  y: number;
  finger: 1 | 2 | 3 | 4 | 5;
  confidence: Confidence;
  score: number;
  delay: number;
  selected?: boolean;
};

const fingerMarks: FingerMark[] = [
  { x: 22, y: 34, finger: 1, confidence: "high", score: 97, delay: 0.45 },
  { x: 31, y: 30, finger: 2, confidence: "high", score: 94, delay: 0.52 },
  { x: 40, y: 37, finger: 4, confidence: "medium", score: 81, delay: 0.59, selected: true },
  { x: 52, y: 27, finger: 1, confidence: "high", score: 96, delay: 0.66 },
  { x: 61, y: 32, finger: 3, confidence: "high", score: 92, delay: 0.73 },
  { x: 70, y: 25, finger: 5, confidence: "review", score: 68, delay: 0.8 },
  { x: 82, y: 35, finger: 2, confidence: "high", score: 95, delay: 0.87 },
  { x: 91, y: 29, finger: 1, confidence: "medium", score: 84, delay: 0.94 },
];

const confidenceStyles: Record<
  Confidence,
  { badge: string; ring: string; label: string; bar: string; glow: string }
> = {
  high: {
    badge: "bg-teal-500 text-white",
    ring: "ring-teal-400/40",
    label: "text-teal-400",
    bar: "bg-teal-400",
    glow: "shadow-[0_0_16px_rgba(45,212,191,0.45)]",
  },
  medium: {
    badge: "bg-amber-500 text-zinc-950",
    ring: "ring-amber-400/45",
    label: "text-amber-400",
    bar: "bg-amber-400",
    glow: "shadow-[0_0_22px_rgba(245,158,11,0.55)]",
  },
  review: {
    badge: "bg-rose-500 text-white",
    ring: "ring-rose-400/50 ring-dashed",
    label: "text-rose-400",
    bar: "bg-rose-400",
    glow: "shadow-[0_0_16px_rgba(244,63,94,0.4)]",
  },
};

const alternatives = [
  { fingers: "1–2–3–1–3–5", score: 81, active: true },
  { fingers: "1–2–4–1–3–5", score: 76, active: false },
  { fingers: "2–3–4–1–2–4", score: 62, active: false },
];

function WindowChrome() {
  return (
    <div className="flex items-center gap-3 border-b border-white/[0.06] bg-[#0a0c10]/95 px-4 py-2.5">
      <div className="flex items-center gap-1.5">
        <span className="size-2.5 rounded-full bg-[#ff5f57]/90 transition-opacity hover:opacity-100" />
        <span className="size-2.5 rounded-full bg-[#febc2e]/90" />
        <span className="size-2.5 rounded-full bg-[#28c840]/90" />
      </div>
      <div className="flex flex-1 items-center justify-center">
        <div className="flex h-7 max-w-md flex-1 items-center justify-center rounded-md bg-white/[0.035] px-3 ring-1 ring-white/[0.05] transition-colors hover:bg-white/[0.05]">
          <span className="truncate font-mono text-[11px] text-zinc-400">
            fingerflow.app / scores / chopin-op9-no2
          </span>
        </div>
      </div>
      <div className="hidden w-[52px] sm:block" />
    </div>
  );
}

function AppToolbar() {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-white/[0.06] bg-[#0d1016] px-3 py-2.5 sm:px-4">
      <div className="flex min-w-0 items-center gap-2.5 sm:gap-3">
        <div className="hidden size-7 shrink-0 items-center justify-center rounded-lg bg-brand/15 ring-1 ring-brand/25 sm:flex">
          <span className="font-mono text-[10px] font-semibold text-brand">
            ff
          </span>
        </div>
        <div className="min-w-0">
          <p className="truncate text-xs font-medium text-white sm:text-[13px]">
            Nocturne Op. 9 No. 2
          </p>
          <p className="truncate text-[10px] text-zinc-500">
            Chopin · mm. 1–8 · Right hand
          </p>
        </div>
      </div>

      <div className="flex items-center gap-1.5 sm:gap-2">
        <div className="hidden items-center rounded-lg bg-white/[0.035] p-0.5 ring-1 ring-white/[0.05] sm:flex">
          <button
            type="button"
            className="rounded-md bg-white/[0.08] px-2.5 py-1 text-[11px] font-medium text-white transition-colors hover:bg-white/[0.12]"
          >
            RH
          </button>
          <button
            type="button"
            className="rounded-md px-2.5 py-1 text-[11px] font-medium text-zinc-500 transition-colors hover:text-zinc-300"
          >
            LH
          </button>
        </div>

        <div className="flex items-center gap-0.5 rounded-lg bg-white/[0.035] p-0.5 ring-1 ring-white/[0.05]">
          <span className="flex size-7 items-center justify-center rounded-md text-zinc-400 transition-colors hover:bg-white/[0.06] hover:text-zinc-200">
            <Minus className="size-3.5" />
          </span>
          <span className="min-w-[2.75rem] text-center font-mono text-[11px] text-zinc-300">
            125%
          </span>
          <span className="flex size-7 items-center justify-center rounded-md text-zinc-400 transition-colors hover:bg-white/[0.06] hover:text-zinc-200">
            <Plus className="size-3.5" />
          </span>
          <span className="mx-0.5 hidden h-4 w-px bg-white/10 sm:block" />
          <span className="hidden size-7 items-center justify-center rounded-md text-zinc-400 transition-colors hover:bg-white/[0.06] hover:text-zinc-200 sm:flex">
            <Maximize2 className="size-3.5" />
          </span>
        </div>

        <span className="hidden items-center gap-1.5 rounded-full bg-brand/10 px-2.5 py-1 text-[10px] font-medium text-brand ring-1 ring-brand/20 md:inline-flex">
          <span className="size-1.5 animate-pulse rounded-full bg-brand" />
          Analyzed
        </span>
      </div>
    </div>
  );
}

function SheetMusicSvg() {
  return (
    <svg
      viewBox="0 0 640 360"
      className="h-full w-full"
      role="img"
      aria-label="Sheet music with FingerFlow fingering overlays"
    >
      <defs>
        <filter id="paperGrain">
          <feTurbulence
            type="fractalNoise"
            baseFrequency="0.9"
            numOctaves="3"
            stitchTiles="stitch"
          />
          <feColorMatrix type="saturate" values="0" />
          <feComponentTransfer>
            <feFuncA type="linear" slope="0.04" />
          </feComponentTransfer>
          <feBlend in="SourceGraphic" mode="multiply" />
        </filter>
        <linearGradient id="selectionFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="rgba(245,158,11,0.16)" />
          <stop offset="100%" stopColor="rgba(245,158,11,0.04)" />
        </linearGradient>
      </defs>

      <rect width="640" height="360" fill="#f3efe6" />
      <rect width="640" height="360" fill="#f3efe6" filter="url(#paperGrain)" />

      <rect x="0" y="0" width="640" height="18" fill="url(#selectionFill)" opacity="0" />

      <path
        d="M42 78 C28 120, 28 210, 42 252"
        fill="none"
        stroke="#1c1917"
        strokeWidth="2.25"
      />
      <line x1="48" y1="78" x2="48" y2="252" stroke="#1c1917" strokeWidth="1.5" />

      {[90, 102, 114, 126, 138].map((y) => (
        <line
          key={`t-${y}`}
          x1="48"
          y1={y}
          x2="600"
          y2={y}
          stroke="#292524"
          strokeWidth="1.1"
        />
      ))}
      {[192, 204, 216, 228, 240].map((y) => (
        <line
          key={`b-${y}`}
          x1="48"
          y1={y}
          x2="600"
          y2={y}
          stroke="#292524"
          strokeWidth="1.1"
        />
      ))}

      <text x="56" y="136" fontFamily="Georgia, 'Times New Roman', serif" fontSize="48" fill="#1c1917">
        𝄞
      </text>
      <text x="58" y="236" fontFamily="Georgia, 'Times New Roman', serif" fontSize="40" fill="#1c1917">
        𝄢
      </text>

      <text x="102" y="108" fontFamily="Georgia, serif" fontSize="18" fill="#1c1917">
        ♯
      </text>
      <text x="102" y="210" fontFamily="Georgia, serif" fontSize="18" fill="#1c1917">
        ♯
      </text>

      <text x="120" y="116" fontFamily="Georgia, serif" fontSize="18" fontWeight="700" fill="#1c1917">
        6
      </text>
      <text x="120" y="134" fontFamily="Georgia, serif" fontSize="18" fontWeight="700" fill="#1c1917">
        8
      </text>
      <text x="120" y="218" fontFamily="Georgia, serif" fontSize="18" fontWeight="700" fill="#1c1917">
        6
      </text>
      <text x="120" y="236" fontFamily="Georgia, serif" fontSize="18" fontWeight="700" fill="#1c1917">
        8
      </text>

      <rect
        x="228"
        y="78"
        width="72"
        height="60"
        rx="4"
        fill="url(#selectionFill)"
        stroke="rgba(245,158,11,0.45)"
        strokeWidth="1.25"
      />

      {(
        [
          [155, 126],
          [185, 120],
          [215, 114],
          [255, 126],
          [285, 108],
          [315, 120],
          [355, 102],
          [385, 114],
          [425, 96],
          [455, 108],
          [495, 120],
          [525, 102],
          [555, 114],
          [585, 108],
        ] as const
      ).map(([x, y], i) => (
        <g key={`tn-${i}`}>
          <ellipse
            cx={x}
            cy={y}
            rx="8"
            ry="5.8"
            fill="#1c1917"
            transform={`rotate(-20 ${x} ${y})`}
          />
          <line
            x1={x + 7.2}
            y1={y}
            x2={x + 7.2}
            y2={y - 38}
            stroke="#1c1917"
            strokeWidth="1.35"
          />
        </g>
      ))}

      <path
        d="M162 82 L222 70"
        stroke="#1c1917"
        strokeWidth="3.5"
        strokeLinecap="round"
      />
      <path
        d="M262 88 L322 74"
        stroke="#1c1917"
        strokeWidth="3.5"
        strokeLinecap="round"
      />

      {(
        [
          [155, 228],
          [215, 216],
          [285, 228],
          [355, 204],
          [425, 216],
          [495, 228],
          [555, 210],
        ] as const
      ).map(([x, y], i) => (
        <g key={`bn-${i}`}>
          <ellipse
            cx={x}
            cy={y}
            rx="8"
            ry="5.8"
            fill="#1c1917"
            transform={`rotate(-20 ${x} ${y})`}
          />
          <line
            x1={x - 7.2}
            y1={y}
            x2={x - 7.2}
            y2={y + 34}
            stroke="#1c1917"
            strokeWidth="1.35"
          />
        </g>
      ))}

      {[140, 340, 470, 600].map((x) => (
        <g key={`bar-${x}`}>
          <line x1={x} y1={90} x2={x} y2={138} stroke="#1c1917" strokeWidth="1.35" />
          <line x1={x} y1={192} x2={x} y2={240} stroke="#1c1917" strokeWidth="1.35" />
        </g>
      ))}

      <text x="148" y="72" fontFamily="ui-sans-serif, system-ui" fontSize="9" fill="#78716c">
        1
      </text>
      <text x="348" y="72" fontFamily="ui-sans-serif, system-ui" fontSize="9" fill="#78716c">
        2
      </text>
      <text x="478" y="72" fontFamily="ui-sans-serif, system-ui" fontSize="9" fill="#78716c">
        3
      </text>

      <text
        x="155"
        y="292"
        fontFamily="Georgia, serif"
        fontSize="13"
        fontStyle="italic"
        fill="#57534e"
      >
        Ped.
      </text>
      <line x1="180" y1="288" x2="320" y2="288" stroke="#57534e" strokeWidth="1" />
      <text
        x="325"
        y="292"
        fontFamily="Georgia, serif"
        fontSize="12"
        fill="#57534e"
      >
        ✕
      </text>
    </svg>
  );
}

function FingerOverlay({ mark }: { mark: FingerMark }) {
  const styles = confidenceStyles[mark.confidence];

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.55, y: 6 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      transition={{ delay: mark.delay, duration: 0.4, ease: easeOutExpo }}
      className="absolute -translate-x-1/2 -translate-y-1/2"
      style={{ left: `${mark.x}%`, top: `${mark.y}%` }}
    >
      <div className="group/finger relative cursor-default">
        {mark.selected ? (
          <motion.span
            aria-hidden
            className="absolute inset-0 -m-1 rounded-full bg-amber-400/30"
            animate={{ scale: [1, 1.35, 1], opacity: [0.55, 0.15, 0.55] }}
            transition={{ duration: 2.4, repeat: Infinity, ease: "easeInOut" }}
          />
        ) : null}
        <span
          className={cn(
            "relative flex size-6 items-center justify-center rounded-full text-[11px] font-bold ring-2 transition-transform duration-200 sm:size-7 sm:text-xs",
            styles.badge,
            styles.ring,
            mark.selected
              ? cn("z-10 scale-110 ring-[3px]", styles.glow)
              : "shadow-[0_2px_8px_rgba(0,0,0,0.18)] group-hover/finger:scale-105"
          )}
        >
          {mark.finger}
        </span>
        <span
          className={cn(
            "absolute -right-2.5 -top-2.5 hidden rounded-full bg-[#0d1016]/95 px-1.5 py-0.5 font-mono text-[8px] leading-none ring-1 ring-white/10 backdrop-blur-sm sm:inline",
            styles.label
          )}
        >
          {mark.score}
        </span>
      </div>
    </motion.div>
  );
}

function ZoomControls() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.7, duration: 0.45, ease: easeOutExpo }}
      className="absolute bottom-3 left-3 flex items-center gap-0.5 rounded-xl border border-zinc-200/70 bg-white/90 p-1 shadow-[0_8px_30px_-12px_rgba(0,0,0,0.35)] backdrop-blur-md sm:bottom-4 sm:left-4"
    >
      <span className="flex size-7 items-center justify-center rounded-lg text-zinc-600 transition-colors hover:bg-zinc-100 hover:text-zinc-900">
        <Minus className="size-3.5" />
      </span>
      <span className="min-w-[2.5rem] text-center font-mono text-[11px] font-medium text-zinc-800">
        125%
      </span>
      <span className="flex size-7 items-center justify-center rounded-lg text-zinc-600 transition-colors hover:bg-zinc-100 hover:text-zinc-900">
        <Plus className="size-3.5" />
      </span>
      <span className="mx-0.5 h-4 w-px bg-zinc-200" />
      <span className="flex size-7 items-center justify-center rounded-lg text-zinc-600 transition-colors hover:bg-zinc-100 hover:text-zinc-900">
        <Maximize2 className="size-3.5" />
      </span>
    </motion.div>
  );
}

function ConfidenceLegend() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.85, duration: 0.45, ease: easeOutExpo }}
      className="absolute bottom-3 right-3 hidden items-center gap-3.5 rounded-xl border border-zinc-200/70 bg-white/90 px-3.5 py-2 shadow-[0_8px_30px_-12px_rgba(0,0,0,0.35)] backdrop-blur-md sm:bottom-4 sm:right-4 md:flex"
    >
      {(
        [
          { key: "high" as const, label: "High" },
          { key: "medium" as const, label: "Medium" },
          { key: "review" as const, label: "Review" },
        ] as const
      ).map((item) => (
        <div key={item.key} className="flex items-center gap-1.5">
          <span
            className={cn(
              "flex size-3.5 items-center justify-center rounded-full shadow-sm",
              confidenceStyles[item.key].badge
            )}
          />
          <span className="text-[10px] font-medium text-zinc-600">{item.label}</span>
        </div>
      ))}
    </motion.div>
  );
}

function AnalysisSidebar() {
  return (
    <motion.aside
      initial={{ opacity: 0, x: 16 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: 0.4, duration: 0.55, ease: easeOutExpo }}
      className="flex w-full shrink-0 flex-col border-t border-white/[0.06] bg-[#0a0c10] md:w-[300px] md:border-t-0 md:border-l"
    >
      <div className="border-b border-white/[0.06] px-4 py-3.5">
        <div className="flex items-center justify-between">
          <p className="text-[13px] font-semibold tracking-tight text-white">
            Analysis
          </p>
          <span className="inline-flex items-center gap-1 rounded-md bg-white/[0.035] px-1.5 py-0.5 text-[10px] text-zinc-400 ring-1 ring-white/[0.05] transition-colors hover:bg-white/[0.06]">
            <RefreshCw className="size-2.5" />
            v3
          </span>
        </div>
        <p className="mt-1 text-[11px] text-zinc-500">Measure 1 · beat 3 · RH</p>
      </div>

      <div className="space-y-5 overflow-hidden p-4">
        <div>
          <div className="mb-2.5 flex items-center justify-between">
            <span className="text-[11px] font-medium text-zinc-400">
              Passage confidence
            </span>
            <span className="font-mono text-[11px] tabular-nums text-teal-400">
              91%
            </span>
          </div>
          <div className="h-1.5 overflow-hidden rounded-full bg-white/[0.05]">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: "91%" }}
              transition={{ delay: 0.9, duration: 0.8, ease: easeOutExpo }}
              className="h-full rounded-full bg-gradient-to-r from-teal-500 to-teal-300 shadow-[0_0_8px_rgba(45,212,191,0.35)]"
            />
          </div>
          <div className="mt-2.5 flex flex-wrap gap-1.5">
            <span className="rounded-md bg-teal-500/10 px-1.5 py-0.5 text-[10px] text-teal-400 ring-1 ring-teal-500/15">
              6 high
            </span>
            <span className="rounded-md bg-amber-500/10 px-1.5 py-0.5 text-[10px] text-amber-400 ring-1 ring-amber-500/15">
              1 medium
            </span>
            <span className="rounded-md bg-rose-500/10 px-1.5 py-0.5 text-[10px] text-rose-400 ring-1 ring-rose-500/15">
              1 review
            </span>
          </div>
        </div>

        <div className="rounded-xl border border-amber-500/20 bg-gradient-to-b from-amber-500/[0.08] to-amber-500/[0.02] p-3.5 shadow-[inset_0_1px_0_rgba(251,191,36,0.08)]">
          <div className="flex items-start justify-between gap-2.5">
            <div className="min-w-0">
              <p className="text-[11px] font-medium text-amber-300">
                Selected · finger 4
              </p>
              <p className="mt-1.5 text-[11px] leading-relaxed text-zinc-400">
                Crosses over 2 on black-key approach. Medium confidence — check
                hand position.
              </p>
            </div>
            <span className="flex size-8 shrink-0 items-center justify-center rounded-full bg-amber-500 text-sm font-bold text-zinc-950 shadow-[0_0_18px_rgba(245,158,11,0.45)]">
              4
            </span>
          </div>
          <div className="mt-3.5">
            <div className="mb-1.5 flex items-center justify-between">
              <span className="text-[10px] text-zinc-500">Model confidence</span>
              <span className="font-mono text-[10px] tabular-nums text-amber-400">
                81%
              </span>
            </div>
            <div className="h-1 overflow-hidden rounded-full bg-white/[0.06]">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: "81%" }}
                transition={{ delay: 1.05, duration: 0.7, ease: easeOutExpo }}
                className="h-full rounded-full bg-amber-400"
              />
            </div>
          </div>
        </div>

        <div>
          <div className="mb-2.5 flex items-center gap-1.5">
            <Sparkles className="size-3 text-brand" />
            <span className="text-[11px] font-medium text-zinc-300">
              Alternatives
            </span>
          </div>
          <div className="space-y-1.5">
            {alternatives.map((alt) => (
              <div
                key={alt.fingers}
                className={cn(
                  "flex items-center justify-between rounded-lg px-2.5 py-2 ring-1 transition-colors duration-150",
                  alt.active
                    ? "bg-white/[0.05] ring-white/10"
                    : "bg-transparent ring-white/[0.04] hover:bg-white/[0.03] hover:ring-white/[0.08]"
                )}
              >
                <div className="flex items-center gap-2">
                  {alt.active ? (
                    <Check className="size-3 text-brand" />
                  ) : (
                    <span className="size-3 rounded-full border border-white/15" />
                  )}
                  <span className="font-mono text-[11px] text-zinc-300">
                    {alt.fingers}
                  </span>
                </div>
                <span
                  className={cn(
                    "font-mono text-[10px] tabular-nums",
                    alt.score >= 80
                      ? "text-teal-400"
                      : alt.score >= 70
                        ? "text-amber-400"
                        : "text-zinc-500"
                  )}
                >
                  {alt.score}%
                </span>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-xl border border-white/[0.05] bg-white/[0.02] p-3.5 transition-colors hover:border-white/[0.08] hover:bg-white/[0.03]">
          <div className="flex items-center gap-2.5">
            <span className="flex size-7 items-center justify-center rounded-lg bg-white/[0.04] text-zinc-300 ring-1 ring-white/[0.05]">
              <Hand className="size-3.5" />
            </span>
            <div className="min-w-0 flex-1">
              <p className="text-[11px] font-medium text-zinc-200">
                Studio profile
              </p>
              <p className="text-[10px] text-zinc-500">Span 8.5″ · lyrical</p>
            </div>
            <ChevronDown className="size-3.5 text-zinc-600" />
          </div>
          <div className="mt-3 grid grid-cols-2 gap-2">
            <div className="rounded-lg bg-white/[0.025] px-2.5 py-2 ring-1 ring-white/[0.04]">
              <p className="text-[9px] uppercase tracking-wide text-zinc-600">
                Goal
              </p>
              <p className="mt-0.5 text-[11px] text-zinc-300">Expression</p>
            </div>
            <div className="rounded-lg bg-white/[0.025] px-2.5 py-2 ring-1 ring-white/[0.04]">
              <p className="text-[9px] uppercase tracking-wide text-zinc-600">
                Tempo
              </p>
              <p className="mt-0.5 text-[11px] text-zinc-300">♩ = 60</p>
            </div>
          </div>
        </div>
      </div>
    </motion.aside>
  );
}

export function ProductMockup() {
  return (
    <div className="relative mx-auto w-full">
      <div className="pointer-events-none absolute -inset-x-10 -top-24 bottom-10 bg-[radial-gradient(ellipse_at_center,rgba(94,234,212,0.14),transparent_58%)]" />

      <motion.div
        initial={{ opacity: 0, y: 40, scale: 0.985 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.9, delay: 0.2, ease: easeOutExpo }}
      >
        <motion.div
          animate={{ y: [0, -6, 0] }}
          transition={{
            duration: 6,
            delay: 1.2,
            repeat: Infinity,
            ease: "easeInOut",
          }}
          className="relative overflow-hidden rounded-2xl border border-white/[0.09] bg-[#0a0c10] shadow-[0_32px_100px_-36px_rgba(0,0,0,0.85),0_0_0_1px_rgba(255,255,255,0.03)]"
        >
          <WindowChrome />
          <AppToolbar />

          <div className="flex flex-col md:flex-row">
            <div className="relative min-w-0 flex-1 bg-[#12151c]">
              <div className="relative m-2.5 overflow-hidden rounded-xl border border-white/[0.05] bg-[#f3efe6] shadow-[inset_0_1px_0_rgba(255,255,255,0.4)] sm:m-3.5">
                <div className="aspect-[16/10] sm:aspect-[16/9]">
                  <SheetMusicSvg />
                </div>

                {fingerMarks.map((mark) => (
                  <FingerOverlay
                    key={`${mark.x}-${mark.finger}-${mark.delay}`}
                    mark={mark}
                  />
                ))}

                <ZoomControls />
                <ConfidenceLegend />
              </div>
            </div>

            <AnalysisSidebar />
          </div>
        </motion.div>
      </motion.div>
    </div>
  );
}
