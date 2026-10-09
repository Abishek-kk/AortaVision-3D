import { useEffect, useRef, useState } from "react";

const ACKNOWLEDGEMENT_KEY = "aortavision_disclaimer_ack";

export default function DisclaimerModal() {
  const [isOpen, setIsOpen] = useState(() => {
    try {
      return window.localStorage.getItem(ACKNOWLEDGEMENT_KEY) !== "true";
    } catch (error) {
      console.warn("Unable to read disclaimer acknowledgement from local storage.", error);
      return true;
    }
  });
  const dialogRef = useRef(null);
  const acknowledgeButtonRef = useRef(null);

  useEffect(() => {
    if (!isOpen) return undefined;

    acknowledgeButtonRef.current?.focus();

    function trapFocus(event) {
      if (event.key !== "Tab") return;
      const dialog = dialogRef.current;
      const focusableElements = dialogRef.current?.querySelectorAll(
        'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
      );
      if (!focusableElements?.length) return;

      const first = focusableElements[0];
      const last = focusableElements[focusableElements.length - 1];
      if (!dialog?.contains(document.activeElement)) {
        event.preventDefault();
        first.focus();
      } else if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }

    document.addEventListener("keydown", trapFocus);
    return () => document.removeEventListener("keydown", trapFocus);
  }, [isOpen]);

  function acknowledge() {
    try {
      window.localStorage.setItem(ACKNOWLEDGEMENT_KEY, "true");
    } catch (error) {
      console.warn("Unable to save disclaimer acknowledgement to local storage.", error);
    }
    setIsOpen(false);
  }

  if (!isOpen) return null;

  return (
    <div className="disclaimer-modal fixed inset-0 z-[100] flex items-center justify-center bg-slate-950/85 p-4">
      <section
        ref={dialogRef}
        aria-describedby="first-visit-disclaimer-copy"
        aria-labelledby="first-visit-disclaimer-title"
        aria-modal="true"
        className="w-full max-w-xl rounded-2xl border border-yellow/70 bg-deepNavy p-6 text-white shadow-2xl sm:p-8"
        role="dialog"
      >
        <h2 className="text-xl font-extrabold text-yellow sm:text-2xl" id="first-visit-disclaimer-title">
          Important clinical disclaimer
        </h2>
        <p className="mt-4 leading-relaxed text-slate-100" id="first-visit-disclaimer-copy">
          Predictions are for decision support and educational purposes only, not a substitute for
          formal diagnostic imaging or a clinician&apos;s judgment. The 3D artery positions are
          schematic and are not an imaging-based lesion map. The model was trained on a small
          single-centre dataset.
        </p>
        <button
          ref={acknowledgeButtonRef}
          className="mt-6 rounded-full border border-yellow bg-yellow px-6 py-3 font-extrabold text-deepNavy transition hover:brightness-110 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-yellow"
          onClick={acknowledge}
          type="button"
        >
          I understand
        </button>
      </section>
    </div>
  );
}
