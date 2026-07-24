import type { Metadata } from "next";
import Link from "next/link";
import { Footer, Navbar } from "@/components/landing";

export const metadata: Metadata = {
  title: "Results — FingerFlow",
  description: "Your personalized piano fingerings are ready.",
};

export default function ResultsPage() {
  return (
    <div className="relative flex min-h-full flex-1 flex-col overflow-x-hidden bg-background">
      <Navbar />
      <main className="flex flex-1 items-center justify-center">
        <section className="relative w-full overflow-hidden px-6 py-28 sm:py-36">
          <div className="pointer-events-none absolute inset-0 bg-grid opacity-60 [mask-image:radial-gradient(ellipse_at_center,black_15%,transparent_70%)]" />
          <div className="pointer-events-none absolute inset-0 bg-noise opacity-40" />
          <div className="pointer-events-none absolute left-1/2 top-1/2 h-[420px] w-[720px] -translate-x-1/2 -translate-y-1/2 bg-[radial-gradient(ellipse_at_center,rgba(94,234,212,0.12),transparent_65%)]" />

          <div className="relative mx-auto max-w-lg text-center">
            <p className="mb-5 text-sm font-medium tracking-[0.22em] text-brand uppercase">
              Results
            </p>
            <h1 className="text-balance text-3xl font-semibold tracking-tight text-white sm:text-4xl">
              Your fingerings are ready
            </h1>
            <p className="mx-auto mt-4 max-w-md text-base leading-relaxed text-zinc-400">
              Personalized markings for your score. Full results view coming
              next.
            </p>
            <div className="mt-9 flex flex-col items-center justify-center gap-3 sm:flex-row">
              <Link
                href="/upload"
                className="inline-flex h-11 items-center rounded-full bg-white px-5 text-sm font-medium text-zinc-950 transition hover:bg-zinc-200"
              >
                Analyze another score
              </Link>
              <Link
                href="/"
                className="inline-flex h-11 items-center rounded-full border border-white/12 bg-white/5 px-5 text-sm font-medium text-white transition hover:border-white/20 hover:bg-white/10"
              >
                Back to home
              </Link>
            </div>
          </div>
        </section>
      </main>
      <Footer />
    </div>
  );
}
