export default function GlowButton({
  children,
  className = "",
  disabled = false,
  type = "button",
  ...props
}) {
  return (
    <button
      className={`glow-button inline-flex items-center justify-center rounded-full px-6 py-3 font-bold ${className}`}
      disabled={disabled}
      type={type}
      {...props}
    >
      {children}
    </button>
  );
}
