import { cn } from "@/lib/utils";

type BrandLogoProps = {
  className?: string;
  showName?: boolean;
};

export function BrandLogo({ className, showName = true }: BrandLogoProps) {
  return (
    <span className={cn("flex items-center gap-2.5", className)}>
      <span className="relative flex size-7 items-center justify-center rounded-lg bg-brand/15 ring-1 ring-brand/30">
        <span className="font-mono text-xs font-semibold text-brand">ff</span>
      </span>
      {showName ? (
        <span className="text-[15px] font-semibold tracking-tight text-white">
          FingerFlow
        </span>
      ) : null}
    </span>
  );
}
