"use client";

import { motion } from "framer-motion";
import { easeOutExpo, viewportOnce } from "./motion";

type Step = {
  step: string;
  title: string;
  description: string;
};

const steps: Step[] = [
  {
    step: "01",
    title: "Upload your score",
    description:
      "Drop a photo or PDF of the passage you’re working on. We handle glare, angles, and imperfect scans.",
  },
  {
    step: "02",
    title: "Tune your profile",
    description:
      "Tell FingerFlow about your hand size, repertoire level, and what you’re optimizing for today.",
  },
  {
    step: "03",
    title: "Play the markings",
    description:
      "Get annotated fingerings with rationale. Edit any number, regenerate alternatives, and export to your PDF.",
  },
];

function HowItWorksCard({ step, index }: { step: Step; index: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-40px" }}
      transition={{
        duration: 0.55,
        delay: index * 0.1,
        ease: easeOutExpo,
      }}
      className="relative rounded-2xl border border-white/8 bg-gradient-to-b from-white/[0.04] to-transparent p-7"
    >
      <span className="font-mono text-xs tracking-widest text-brand">
        {step.step}
      </span>
      <h3 className="mt-4 text-lg font-semibold tracking-tight text-white">
        {step.title}
      </h3>
      <p className="mt-2 text-sm leading-relaxed text-zinc-400">
        {step.description}
      </p>
    </motion.div>
  );
}

export function HowItWorksSection() {
  return (
    <section id="how-it-works" className="relative py-24 sm:py-32">
      <div className="pointer-events-none absolute inset-x-0 top-1/2 h-64 -translate-y-1/2 bg-[radial-gradient(ellipse_at_center,rgba(94,234,212,0.06),transparent_70%)]" />

      <div className="relative mx-auto max-w-6xl px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={viewportOnce}
          transition={{ duration: 0.55, ease: easeOutExpo }}
          className="mx-auto max-w-2xl text-center"
        >
          <p className="mb-3 text-sm font-medium tracking-wide text-brand">
            How it works
          </p>
          <h2 className="text-balance text-3xl font-semibold tracking-tight text-white sm:text-4xl">
            Three steps to better fingerings
          </h2>
          <p className="mt-4 text-pretty text-base leading-relaxed text-zinc-400">
            No music theory PhD required — just your score and a few minutes.
          </p>
        </motion.div>

        <div className="mt-14 grid gap-4 md:grid-cols-3">
          {steps.map((step, index) => (
            <HowItWorksCard key={step.step} step={step} index={index} />
          ))}
        </div>
      </div>
    </section>
  );
}
