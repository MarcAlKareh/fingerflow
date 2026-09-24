import type { Metadata } from "next";
import { Suspense } from "react";
import { Navbar } from "@/components/landing";
import { ProcessingStatus } from "@/components/upload/processing-status";

export const metadata: Metadata = {
  title: "Reading score — FingerFlow",
  description: "FingerFlow is reading your score and assigning fingerings.",
};

export default function ProcessingPage() {
  return (
    <div className="flex min-h-full flex-1 flex-col bg-background">
      <Navbar />
      <main className="flex flex-1 items-center justify-center px-6 py-20">
        <Suspense>
          <ProcessingStatus />
        </Suspense>
      </main>
    </div>
  );
}
