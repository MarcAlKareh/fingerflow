import {
  FeatureSection,
  Footer,
  HeroSection,
  HowItWorksSection,
  Navbar,
} from "@/components/landing";

export default function Home() {
  return (
    <div className="relative flex min-h-full flex-1 flex-col overflow-x-hidden bg-background">
      <Navbar />
      <main className="flex-1">
        <HeroSection />
        <FeatureSection />
        <HowItWorksSection />
      </main>
      <Footer />
    </div>
  );
}
