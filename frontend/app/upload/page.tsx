import type { Metadata } from "next";
import { Footer, Navbar } from "@/components/landing";
import { UploadForm } from "@/components/upload/upload-form";
import { UploadHeader } from "@/components/upload/upload-header";

export const metadata: Metadata = {
  title: "Upload — FingerFlow",
  description: "Upload a photo of your sheet music and generate piano fingerings.",
};

export default function UploadPage() {
  return (
    <div className="flex min-h-full flex-1 flex-col bg-background">
      <Navbar />
      <main className="flex-1 px-6 pt-12 pb-20 sm:pt-16">
        <div className="mx-auto max-w-xl">
          <UploadHeader />
          <div className="mt-10">
            <UploadForm />
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
}
