import Link from "next/link";
import { BrandLogo } from "./brand-logo";

const footerLinks = [
  { href: "/#features", label: "Features" },
  { href: "/#how-it-works", label: "How it works" },
  { href: "/upload", label: "Upload" },
  { href: "mailto:hello@fingerflow.ai", label: "Contact" },
];

export function Footer() {
  return (
    <footer className="border-t border-white/8 py-12">
      <div className="mx-auto flex max-w-6xl flex-col items-start justify-between gap-8 px-6 sm:flex-row sm:items-center">
        <div>
          <Link href="/">
            <BrandLogo />
          </Link>
          <p className="mt-3 max-w-sm text-sm text-zinc-500">
            Personalized piano fingerings from sheet music — powered by AI.
          </p>
        </div>

        <div className="flex flex-wrap gap-x-8 gap-y-3 text-sm text-zinc-400">
          {footerLinks.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className="transition hover:text-white"
            >
              {link.label}
            </Link>
          ))}
        </div>
      </div>

      <div className="mx-auto mt-10 max-w-6xl px-6">
        <p className="text-xs text-zinc-600">
          © {new Date().getFullYear()} FingerFlow. All rights reserved.
        </p>
      </div>
    </footer>
  );
}
