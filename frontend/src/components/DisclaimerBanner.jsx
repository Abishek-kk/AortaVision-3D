import { TriangleAlert } from "lucide-react";

export default function DisclaimerBanner() {
  return (
    <aside
      aria-label="Clinical safety disclaimer"
      className="print-disclaimer-banner flex w-full items-center justify-center gap-2 bg-yellow px-4 py-2.5 text-center text-sm leading-snug text-deepNavy sm:text-base"
    >
      <TriangleAlert aria-hidden="true" className="h-5 w-5 shrink-0" />
      <p>
        <strong className="font-extrabold">CLINICAL SAFETY DISCLAIMER:</strong>{" "}
        Predictions are for decision support and educational purposes only and
        not a substitute for formal diagnostic imaging.
      </p>
    </aside>
  );
}
