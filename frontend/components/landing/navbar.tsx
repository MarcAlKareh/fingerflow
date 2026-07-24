"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "framer-motion";
import { Menu, X } from "lucide-react";
import { BrandLogo } from "./brand-logo";
import { easeOutExpo } from "./motion";

const links = [
  { href: "/#features", label: "Features" },
  { href: "/#how-it-works", label: "How it works" },
  { href: "/#demo", label: "Demo" },
];

export function Navbar() {
  const pathname = usePathname();
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);
  const onAppPage = pathname !== "/";

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    setOpen(false);
  }, [pathname]);

  return (
    <motion.header
      initial={{ y: -16, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5, ease: easeOutExpo }}
      className="fixed inset-x-0 top-0 z-50"
    >
      <motion.div
        animate={{
          borderColor:
            scrolled || onAppPage
              ? "rgba(255,255,255,0.08)"
              : "rgba(255,255,255,0)",
          backgroundColor:
            scrolled || onAppPage
              ? "rgba(7,8,10,0.7)"
              : "rgba(7,8,10,0)",
          backdropFilter: scrolled || onAppPage ? "blur(20px)" : "blur(0px)",
        }}
        transition={{ duration: 0.35, ease: easeOutExpo }}
        className="border-b"
      >
        <nav className="mx-auto flex h-16 max-w-6xl items-center justify-between px-6">
          <Link href="/" className="group">
            <BrandLogo />
          </Link>

          <div className="hidden items-center gap-8 md:flex">
            {links.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className="text-sm text-zinc-400 transition-colors duration-200 hover:text-white"
              >
                {link.label}
              </Link>
            ))}
          </div>

          <div className="hidden items-center md:flex">
            <motion.div
              whileHover={{ y: -1 }}
              whileTap={{ y: 0 }}
              transition={{ duration: 0.2, ease: easeOutExpo }}
            >
              <Link
                href="/upload"
                className="inline-flex h-9 items-center rounded-full bg-white px-4 text-sm font-medium text-zinc-950 transition-colors hover:bg-zinc-200"
              >
                Get started
              </Link>
            </motion.div>
          </div>

          <button
            type="button"
            aria-label={open ? "Close menu" : "Open menu"}
            className="inline-flex size-9 items-center justify-center rounded-lg text-zinc-300 transition-colors hover:bg-white/5 md:hidden"
            onClick={() => setOpen((v) => !v)}
          >
            {open ? <X className="size-5" /> : <Menu className="size-5" />}
          </button>
        </nav>

        {open ? (
          <motion.div
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.25, ease: easeOutExpo }}
            className="border-t border-white/8 bg-[#07080a]/95 px-6 py-4 backdrop-blur-xl md:hidden"
          >
            <div className="flex flex-col gap-3">
              {links.map((link) => (
                <Link
                  key={link.href}
                  href={link.href}
                  onClick={() => setOpen(false)}
                  className="rounded-lg px-2 py-2 text-sm text-zinc-300 transition-colors hover:bg-white/5 hover:text-white"
                >
                  {link.label}
                </Link>
              ))}
              <Link
                href="/upload"
                onClick={() => setOpen(false)}
                className="mt-1 inline-flex h-10 items-center justify-center rounded-full bg-white text-sm font-medium text-zinc-950"
              >
                Get started
              </Link>
            </div>
          </motion.div>
        ) : null}
      </motion.div>
    </motion.header>
  );
}
