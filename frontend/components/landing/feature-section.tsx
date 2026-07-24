"use client";

import type { LucideIcon } from "lucide-react";
import { motion } from "framer-motion";
import { Camera, Hand, Sparkles, Target } from "lucide-react";
import { easeOutExpo, viewportOnce } from "./motion";

type Feature = {
  icon: LucideIcon;
  title: string;
  description: string;
};

const features: Feature[] = [
  {
    icon: Camera,
    title: "Photo to score",
    description:
      "Snap any printed or handwritten page. FingerFlow reads the notation and reconstructs a clean digital score in seconds.",
  },
  {
    icon: Hand,
    title: "Built for your hands",
    description:
      "Set your hand span, preferred fingering style, and injuries or limitations. Every suggestion adapts to you.",
  },
  {
    icon: Sparkles,
    title: "Technique-aware AI",
    description:
      "Trained on conservatory pedagogy — thumb unders, finger substitutions, and voicing choices that actually play.",
  },
  {
    icon: Target,
    title: "Goal-driven markings",
    description:
      "Optimize for speed, expression, or endurance. Switch profiles and regenerate fingerings without re-uploading.",
  },
];

const container = {
  hidden: {},
  show: {
    transition: { staggerChildren: 0.08 },
  },
};

const item = {
  hidden: { opacity: 0, y: 24 },
  show: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.55, ease: easeOutExpo },
  },
};

function FeatureCard({ feature }: { feature: Feature }) {
  const Icon = feature.icon;

  return (
    <motion.article
      variants={item}
      className="group rounded-2xl border border-white/8 bg-card/80 p-6 transition hover:border-brand/25 hover:bg-card"
    >
      <div className="mb-5 inline-flex size-10 items-center justify-center rounded-xl bg-brand/10 text-brand ring-1 ring-brand/20">
        <Icon className="size-5" strokeWidth={1.75} />
      </div>
      <h3 className="text-[15px] font-semibold tracking-tight text-white">
        {feature.title}
      </h3>
      <p className="mt-2 text-sm leading-relaxed text-zinc-400">
        {feature.description}
      </p>
    </motion.article>
  );
}

export function FeatureSection() {
  return (
    <section id="features" className="relative py-24 sm:py-32">
      <div className="mx-auto max-w-6xl px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={viewportOnce}
          transition={{ duration: 0.55, ease: easeOutExpo }}
          className="mx-auto max-w-2xl text-center"
        >
          <p className="mb-3 text-sm font-medium tracking-wide text-brand">
            Features
          </p>
          <h2 className="text-balance text-3xl font-semibold tracking-tight text-white sm:text-4xl">
            Fingerings that feel like yours
          </h2>
          <p className="mt-4 text-pretty text-base leading-relaxed text-zinc-400">
            From first sight-read to recital polish — markings shaped by how you
            actually play.
          </p>
        </motion.div>

        <motion.div
          variants={container}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, margin: "-60px" }}
          className="mt-14 grid gap-4 sm:grid-cols-2 lg:grid-cols-4"
        >
          {features.map((feature) => (
            <FeatureCard key={feature.title} feature={feature} />
          ))}
        </motion.div>
      </div>
    </section>
  );
}
