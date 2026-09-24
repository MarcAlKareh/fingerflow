import type { Metadata } from "next";
import { Suspense } from "react";
import { Footer, Navbar } from "@/components/landing";
import { ResultsView } from "@/components/results/results-view";

export const metadata: Metadata = {
  title: "Results — FingerFlow",
  description: "Your personalized piano fingerings are ready.",
};

export default function ResultsPage() {
  return (
    <div className="flex min-h-full flex-1 flex-col bg-background">
      <Navbar />
      <main className="flex-1">
        <Suspense>
          <ResultsView />
        </Suspense>
      </main>
      <Footer />
    </div>
  );
}
