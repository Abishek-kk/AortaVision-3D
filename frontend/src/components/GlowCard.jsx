export default function GlowCard({
  children,
  className = "",
  headerClassName = "",
  title,
  ...props
}) {
  return (
    <section className={`glow-card bg-white/5 ${className}`} {...props}>
      {title && (
        <h2
          className={`glow-card__header px-5 py-4 text-sm font-extrabold tracking-widest ${headerClassName}`}
        >
          {title}
        </h2>
      )}
      {children}
    </section>
  );
}
