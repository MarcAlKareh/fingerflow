"use client";

import { motion } from "framer-motion";
import { easeOutExpo } from "@/components/landing/motion";

export function UploadHeader() {
  return (
    <div className="mx-auto max-w-2xl text-center">
      <motion.p
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.55, ease: easeOutExpo }}
        className="mb-5 text-sm font-medium tracking-[0.22em] text-brand uppercase"
      >
        Upload
      </motion.p>
      <motion.h1
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.06, ease: easeOutExpo }}
        className="text-balance text-3xl font-semibold tracking-tight text-white sm:text-4xl md:text-5xl md:leading-[1.08]"
      >
        Analyze your score
      </motion.h1>
      <motion.p
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.12, ease: easeOutExpo }}
        className="mx-auto mt-5 max-w-xl text-pretty text-base leading-relaxed text-zinc-400 sm:text-lg"
      >
        Drop a photo of your sheet music, set your hand span and goals, and
        FingerFlow will generate fingerings tailored to you.
      </motion.p>
    </div>
  );
}
