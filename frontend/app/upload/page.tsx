import type { Metadata } from "next";
import { Footer, Navbar } from "@/components/landing";
import { UploadForm } from "@/components/upload/upload-form";
import { UploadHeader } from "@/components/upload/upload-header";

export const metadata: Metadata = {
  title: "Upload Score — FingerFlow",
  description:
    "Upload a photo of your sheet music and generate personalized piano fingerings.",
};

export default function UploadPage() {
  return (
    <div className="relative flex min-h-full flex-1 flex-col overflow-x-hidden bg-background">
      <Navbar />
      <main className="flex-1">
        <section className="relative overflow-hidden pt-28 pb-20 sm:pt-36 sm:pb-28">
          <div className="pointer-events-none absolute inset-0 bg-grid opacity-60 [mask-image:radial-gradient(ellipse_at_top,black_20%,transparent_70%)]" />
          <div className="pointer-events-none absolute inset-0 bg-noise opacity-40" />
          <div className="pointer-events-none absolute left-1/2 top-0 h-[480px] w-[820px] -translate-x-1/2 bg-[radial-gradient(ellipse_at_center,rgba(94,234,212,0.12),transparent_65%)]" />

          <div className="relative mx-auto max-w-6xl px-6">
            <UploadHeader />
            <div className="mt-12 sm:mt-14">
              <UploadForm />
            </div>
          </div>
        </section>
      </main>
      <Footer />
    </div>
  );
}
