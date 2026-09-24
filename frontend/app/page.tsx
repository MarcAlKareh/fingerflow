import {
  FeatureSection,
  Footer,
  HeroSection,
  HowItWorksSection,
  Navbar,
} from "@/components/landing";

export default function Home() {
  return (
    <div className="flex min-h-full flex-1 flex-col bg-background">
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
