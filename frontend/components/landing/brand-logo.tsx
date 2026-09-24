import { cn } from "@/lib/utils";

type BrandLogoProps = {
  className?: string;
  showName?: boolean;
};

export function BrandLogo({ className, showName = true }: BrandLogoProps) {
  return (
    <span className={cn("flex items-center", className)}>
      {showName ? (
        <span className="font-heading text-[17px] tracking-tight text-foreground">
          FingerFlow
        </span>
      ) : (
        <span className="font-heading text-[17px] text-foreground">F</span>
      )}
    </span>
  );
}
