export default function Logo({ className = "h-10 w-10" }) {
  return (
    <svg
      aria-hidden="true"
      className={className}
      fill="none"
      viewBox="0 0 48 48"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M24 41S5 30.2 5 17.7C5 11.8 9.3 8 14.5 8c4 0 7.2 2.4 9.5 6.2C26.3 10.4 29.5 8 33.5 8 38.7 8 43 11.8 43 17.7 43 30.2 24 41 24 41Z"
        stroke="#0B1F4B"
        strokeWidth="3"
      />
      <path
        d="M8 24h8l3.2-6 5.2 13 4.1-8H40"
        stroke="#0F9DB8"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2.5"
      />
    </svg>
  );
}
