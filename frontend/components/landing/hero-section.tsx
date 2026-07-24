"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowRight, Play } from "lucide-react";
import { easeOutExpo } from "./motion";
import { ProductMockup } from "./product-mockup";

export function HeroSection() {
  return (
    <section className="relative overflow-hidden pt-28 pb-24 sm:pt-40 sm:pb-32">
      <div className="pointer-events-none absolute inset-0 bg-grid opacity-60 [mask-image:radial-gradient(ellipse_at_top,black_20%,transparent_70%)]" />
      <div className="pointer-events-none absolute inset-0 bg-noise opacity-40" />
      <div className="pointer-events-none absolute left-1/2 top-0 h-[520px] w-[900px] -translate-x-1/2 bg-[radial-gradient(ellipse_at_center,rgba(94,234,212,0.14),transparent_65%)]" />

      <div className="relative mx-auto max-w-6xl px-6">
        <div className="mx-auto max-w-3xl text-center">
          <motion.p
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.55, ease: easeOutExpo }}
            className="mb-6 text-sm font-medium tracking-[0.22em] text-brand uppercase"
          >
            FingerFlow
          </motion.p>

          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.06, ease: easeOutExpo }}
            className="text-balance text-4xl font-semibold tracking-tight text-white sm:text-5xl md:text-6xl md:leading-[1.05]"
          >
            Every Note. The Right Finger.
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.12, ease: easeOutExpo }}
            className="mx-auto mt-8 max-w-2xl text-pretty text-base leading-relaxed text-zinc-400 sm:text-lg"
          >
            Unlike generic fingering charts, FingerFlow generates personalized
            markings shaped by your hand size, technical ability, and musical
            goals — so every passage feels natural under your hands.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.18, ease: easeOutExpo }}
            className="mt-11 flex flex-col items-center justify-center gap-3 sm:flex-row"
          >
            <motion.div
              whileHover={{ y: -2 }}
              whileTap={{ y: 0 }}
              transition={{ duration: 0.2, ease: easeOutExpo }}
            >
              <Link
                id="upload"
                href="/upload"
                className="inline-flex h-12 items-center gap-2 rounded-full bg-white px-6 text-sm font-medium text-zinc-950 shadow-[0_0_0_1px_rgba(255,255,255,0.08)] transition-colors hover:bg-zinc-200"
              >
                Analyze My Score
                <ArrowRight className="size-4" />
              </Link>
            </motion.div>
            <motion.a
              id="demo"
              href="#demo"
              whileHover={{ y: -2 }}
              whileTap={{ y: 0 }}
              transition={{ duration: 0.2, ease: easeOutExpo }}
              className="inline-flex h-12 items-center gap-2 rounded-full border border-white/12 bg-white/5 px-6 text-sm font-medium text-white backdrop-blur transition-colors hover:border-white/20 hover:bg-white/10"
            >
              <Play className="size-3.5 fill-current" />
              Try Demo
            </motion.a>
          </motion.div>
        </div>
      </div>

      <div className="relative mx-auto mt-20 max-w-7xl px-4 sm:mt-28 sm:px-6">
        <ProductMockup />
      </div>
    </section>
  );
}
