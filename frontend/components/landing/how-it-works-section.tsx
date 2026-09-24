const steps = [
  {
    title: "Upload a page",
    description: "A clear photo of one system works better than a tight crop or a dark phone shot.",
  },
  {
    title: "Set span and goal",
    description: "Hand span is the stretch from thumb to little finger, in centimetres.",
  },
  {
    title: "Read the numbers",
    description: "Fingerings are drawn on your picture. The right-hand list explains the awkward ones.",
  },
];

export function HowItWorksSection() {
  return (
    <section id="how-it-works" className="border-t border-border px-6 py-16 sm:py-20">
      <div className="mx-auto max-w-2xl">
        <h2 className="text-2xl text-foreground sm:text-3xl">How it works</h2>
        <ol className="mt-10 space-y-8">
          {steps.map((step, index) => (
            <li key={step.title}>
              <p className="text-sm text-muted-foreground">{index + 1}.</p>
              <h3 className="mt-1 font-sans text-sm font-medium text-foreground">
                {step.title}
              </h3>
              <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                {step.description}
              </p>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}
