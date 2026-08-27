"use client"; // This tells Next.js to only run this code in the user's browser

import React, { useEffect, useRef } from "react";
import { OpenSheetMusicDisplay } from "opensheetmusicdisplay";

interface SheetMusicViewerProps {
  xmlUrl: string;
  notesData?: any; // We will use this in the next step to draw the fingerings
}

export default function SheetMusicViewer({ xmlUrl, notesData }: SheetMusicViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const osmdRef = useRef<OpenSheetMusicDisplay | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    // Initialize the rendering engine
    osmdRef.current = new OpenSheetMusicDisplay(containerRef.current, {
      autoResize: true,
      drawTitle: true,
      drawPartNames: false,
    });

    // Load and render the sheet music
    osmdRef.current
      .load(xmlUrl)
      .then(() => {
        osmdRef.current?.render();
        
        // TODO: This is where we will inject the AI fingerings from notesData!
      })
      .catch((error) => {
        console.error("Error rendering sheet music:", error);
      });

    // Cleanup function when the user leaves the page
    return () => {
      osmdRef.current?.clear();
    };
  }, [xmlUrl]);

  return (
    <div className="w-full bg-white p-4 rounded shadow">
      <div ref={containerRef} />
    </div>
  );
}