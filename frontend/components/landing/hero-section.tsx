import Link from "next/link";

export function HeroSection() {
  return (
    <section className="px-6 pt-20 pb-16 sm:pt-28 sm:pb-24">
      <div className="mx-auto max-w-2xl">
        <h1 className="text-4xl leading-tight text-foreground sm:text-5xl">
          Fingerings for the hands you have.
        </h1>
        <p className="mt-6 max-w-xl text-base leading-relaxed text-muted-foreground sm:text-lg">
          Photograph a page of piano music. FingerFlow reads the notes and
          suggests fingerings for your span and how you want the passage to
          feel.
        </p>
        <div className="mt-8 flex flex-wrap gap-3">
          <Link
            href="/upload"
            className="inline-flex h-10 items-center rounded-md bg-foreground px-4 text-sm text-background"
          >
            Upload a score
          </Link>
          <a
            href="#how-it-works"
            className="inline-flex h-10 items-center rounded-md border border-border px-4 text-sm text-foreground"
          >
            How it works
          </a>
        </div>
      </div>
    </section>
  );
}
