const features = [
  {
    title: "From a photo",
    description:
      "A printed page is enough. We read the staves and notes; you do not need a MusicXML file.",
  },
  {
    title: "Fitted to your hands",
    description:
      "Left and right span are set in centimetres. The engine treats those as hard limits, not decoration.",
  },
  {
    title: "A physical model",
    description:
      "Fingerings come from a biomechanical cost model trained on the PIG dataset, not a generic chart.",
  },
  {
    title: "Speed or line",
    description:
      "Ask for fewer shifts at tempo, or a more connected voicing. Same score, different goal.",
  },
];

export function FeatureSection() {
  return (
    <section id="features" className="border-t border-border px-6 py-16 sm:py-20">
      <div className="mx-auto max-w-2xl">
        <h2 className="text-2xl text-foreground sm:text-3xl">What it actually does</h2>
        <dl className="mt-10 space-y-8">
          {features.map((feature) => (
            <div key={feature.title}>
              <dt className="text-sm font-medium text-foreground">{feature.title}</dt>
              <dd className="mt-1 text-sm leading-relaxed text-muted-foreground">
                {feature.description}
              </dd>
            </div>
          ))}
        </dl>
      </div>
    </section>
  );
}
