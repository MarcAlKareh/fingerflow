"use client";

import { useMemo } from "react";
import type { ScoreNote } from "@/lib/api";

type ScoreOverlayProps = {
  imageUrl: string;
  rightHand: ScoreNote[];
  leftHand: ScoreNote[];
};

type Mark = ScoreNote & { hand: "right" | "left" };

export function ScoreOverlay({ imageUrl, rightHand, leftHand }: ScoreOverlayProps) {
  const marks = useMemo(() => {
    const all: Mark[] = [
      ...rightHand.map((note) => ({ ...note, hand: "right" as const })),
      ...leftHand.map((note) => ({ ...note, hand: "left" as const })),
    ];
    return all.filter(
      (note) =>
        note.finger &&
        typeof note.x === "number" &&
        typeof note.y === "number"
    );
  }, [rightHand, leftHand]);

  return (
    <div className="relative overflow-hidden bg-card [container-type:inline-size]">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={imageUrl}
        alt="Sheet music with fingerings"
        className="mx-auto block h-auto w-full"
      />
      {marks.map((note) => {
        const above = note.hand === "right";
        return (
          <span
            key={`${note.hand}-${note.note_id}`}
            className="pointer-events-none absolute -translate-x-1/2 select-none font-sans font-semibold tabular-nums text-foreground"
            style={{
              left: `${note.x! * 100}%`,
              top: `${note.y! * 100}%`,
              fontSize: "clamp(7px, 1.05cqi, 11px)",
              lineHeight: 1,
              transform: above
                ? "translate(-50%, calc(-100% - 0.15em))"
                : "translate(-50%, 0.45em)",
              textShadow:
                "0 0 2px #fbfaf6, 1px 0 0 #fbfaf6, -1px 0 0 #fbfaf6, 0 1px 0 #fbfaf6, 0 -1px 0 #fbfaf6",
            }}
          >
            {note.finger}
          </span>
        );
      })}
    </div>
  );
}
