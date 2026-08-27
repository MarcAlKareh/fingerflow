"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import SheetMusicViewer from "@/components/SheetMusicViewer";

export default function ResultsPage() {
  const params = useParams();
  const jobId = params.job_id as string;
  
  const [xmlUrl, setXmlUrl] = useState<string>("");
  const [notesData, setNotesData] = useState<any>(null);

  useEffect(() => {
    if (!jobId) return;

    const fetchResults = async () => {
      try {
        // Fetch the generated AI fingering data
        const notesRes = await fetch(`/api/uploads/${jobId}/notes.json`);
        const notesJson = await notesRes.json();
        setNotesData(notesJson);

        // Define where the backend saved the MusicXML file
        // (Adjust this filename if your Audiveris outputs a different default name)
        setXmlUrl(`/api/uploads/${jobId}/audiveris/score.mxl`);
        
      } catch (error) {
        console.error("Error loading results:", error);
      }
    };

    fetchResults();
  }, [jobId]);

  if (!xmlUrl || !notesData) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <h2 className="text-2xl font-semibold text-gray-700 animate-pulse">
          Loading your custom fingerings...
        </h2>
      </div>
    );
  }

  return (
    <main className="min-h-screen p-8 bg-gray-50">
      <div className="max-w-5xl mx-auto">
        <h1 className="text-3xl font-bold text-gray-900 mb-6">Your FingerFlow Score</h1>
        <SheetMusicViewer xmlUrl={xmlUrl} notesData={notesData} />
      </div>
    </main>
  );
}