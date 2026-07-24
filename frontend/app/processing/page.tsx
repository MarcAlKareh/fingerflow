import type { Metadata } from "next";
import { Navbar } from "@/components/landing";
import { ProcessingStatus } from "@/components/upload/processing-status";

export const metadata: Metadata = {
  title: "Processing — FingerFlow",
  description: "FingerFlow is generating personalized fingerings for your score.",
};

export default function ProcessingPage() {
  return (
    <div className="relative flex min-h-full flex-1 flex-col overflow-x-hidden bg-background">
      <Navbar />
      <main className="flex flex-1 items-center justify-center">
        <section className="relative w-full overflow-hidden px-6 py-28 sm:py-32">
          <div className="pointer-events-none absolute inset-0 bg-grid opacity-60 [mask-image:radial-gradient(ellipse_at_center,black_15%,transparent_70%)]" />
          <div className="pointer-events-none absolute inset-0 bg-noise opacity-40" />
          <div className="pointer-events-none absolute left-1/2 top-1/2 h-[480px] w-[780px] -translate-x-1/2 -translate-y-1/2 bg-[radial-gradient(ellipse_at_center,rgba(94,234,212,0.12),transparent_65%)]" />
          <ProcessingStatus />
        </section>
      </main>
    </div>
  );
}
