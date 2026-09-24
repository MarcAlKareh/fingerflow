"use client";

import { useEffect, useRef, useState } from "react";

type ScoreRendererProps = {
  musicxmlUrl: string;
};

export function ScoreRenderer({ musicxmlUrl }: ScoreRendererProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let cancelled = false;
    let osmd: { clear?: () => void } | null = null;

    const render = async () => {
      setLoading(true);
      setError(null);
      container.replaceChildren();

      try {
        const [{ OpenSheetMusicDisplay }, xmlResponse] = await Promise.all([
          import("opensheetmusicdisplay"),
          fetch(musicxmlUrl, { cache: "no-store" }),
        ]);
        if (!xmlResponse.ok) {
          throw new Error("Could not load the fingered score.");
        }
        const xml = await xmlResponse.text();
        if (cancelled) return;

        const display = new OpenSheetMusicDisplay(container, {
          autoResize: true,
          backend: "svg",
          drawTitle: false,
          drawSubtitle: false,
          drawComposer: false,
          drawCredits: false,
          drawPartNames: false,
          drawFingerings: true,
          drawingParameters: "compacttight",
          pageFormat: "Endless",
        });
        osmd = display;
        const rules = (
          display as {
            EngravingRules?: {
              NewSystemAtXMLNewSystemAttribute?: boolean;
              NewPageAtXMLNewPageAttribute?: boolean;
              PageLeftMargin?: number;
              PageRightMargin?: number;
              PageTopMargin?: number;
              PageBottomMargin?: number;
            };
          }
        ).EngravingRules;
        if (rules) {
          rules.NewSystemAtXMLNewSystemAttribute = false;
          rules.NewPageAtXMLNewPageAttribute = false;
          rules.PageLeftMargin = 3;
          rules.PageRightMargin = 3;
          rules.PageTopMargin = 3;
          rules.PageBottomMargin = 3;
        }
        await display.load(xml);
        if (cancelled) return;
        display.render();
        const svg = container.querySelector("svg");
        if (svg) {
          svg.style.width = "100%";
          svg.style.height = "auto";
          svg.removeAttribute("height");
        }
        setLoading(false);
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Could not render the score.");
        setLoading(false);
      }
    };

    void render();

    return () => {
      cancelled = true;
      try {
        osmd?.clear?.();
      } catch {
        // OSMD may already be disposed
      }
      container.replaceChildren();
    };
  }, [musicxmlUrl]);

  return (
    <div className="relative min-h-[320px] overflow-auto bg-card">
      {loading ? (
        <p className="absolute inset-x-0 top-8 text-center text-sm text-muted-foreground">
          Engraving score…
        </p>
      ) : null}
      {error ? (
        <p className="p-6 text-center text-sm text-destructive">{error}</p>
      ) : null}
      <div ref={containerRef} className="osmd-score min-h-[200px] px-2 py-3 [&_svg]:h-auto [&_svg]:w-full" />
    </div>
  );
}
