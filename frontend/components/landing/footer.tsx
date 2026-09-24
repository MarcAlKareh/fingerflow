import Link from "next/link";

export function Footer() {
  return (
    <footer className="border-t border-border px-6 py-10">
      <div className="mx-auto flex max-w-2xl flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <p className="text-sm text-muted-foreground">
          FingerFlow · piano fingering
        </p>
        <div className="flex gap-6 text-sm text-muted-foreground">
          <Link href="/upload" className="hover:text-foreground">
            Upload
          </Link>
          <Link href="/#how-it-works" className="hover:text-foreground">
            How it works
          </Link>
        </div>
      </div>
    </footer>
  );
}
