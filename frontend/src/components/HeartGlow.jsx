import { useState } from "react";

export default function HeartGlow({ className = "w-full" }) {
  const [hasImage, setHasImage] = useState(true);

  return (
    <div
      aria-hidden="true"
      className={className}
      style={{ filter: "drop-shadow(0 0 18px rgba(56, 189, 248, 0.65)) drop-shadow(0 0 30px rgba(255, 59, 92, 0.35))" }}
    >
      {hasImage ? (
        <img
          alt=""
          className="h-auto w-full object-contain"
          onError={() => setHasImage(false)}
          src="/heart.png"
        />
      ) : (
        <svg
          className="h-auto w-full"
          fill="none"
          viewBox="0 0 360 380"
          xmlns="http://www.w3.org/2000/svg"
        >
          <defs>
            <linearGradient id="heart-body" x1="82" x2="287" y1="84" y2="326" gradientUnits="userSpaceOnUse">
              <stop stopColor="#FF6478" />
              <stop offset=".48" stopColor="#F43F5E" />
              <stop offset="1" stopColor="#F97316" />
            </linearGradient>
            <linearGradient id="heart-vessel" x1="182" x2="185" y1="22" y2="164" gradientUnits="userSpaceOnUse">
              <stop stopColor="#38BDF8" />
              <stop offset="1" stopColor="#2563EB" />
            </linearGradient>
            <radialGradient id="heart-glow">
              <stop stopColor="#FF3B5C" stopOpacity=".55" />
              <stop offset="1" stopColor="#FF3B5C" stopOpacity="0" />
            </radialGradient>
            <filter id="heart-blur" x="0" y="0" width="360" height="380" filterUnits="userSpaceOnUse">
              <feGaussianBlur stdDeviation="15" />
            </filter>
          </defs>
          <ellipse cx="186" cy="205" fill="url(#heart-glow)" filter="url(#heart-blur)" rx="129" ry="146" />
          <path
            d="M185 138c-28-58-80-64-111-39-47 38-28 104 7 148 24 31 61 59 101 91 4 3 10 3 14 0 38-29 82-62 108-99 33-46 42-99 5-133-34-31-86-16-124 32Z"
            fill="url(#heart-body)"
            stroke="#FB7185"
            strokeWidth="5"
          />
          <path
            d="M190 145c-1-37 1-77-2-112-1-13-20-13-22 0l-5 50c-1 13-19 14-22 2l-5-19c-3-11-18-9-19 2l-5 49"
            stroke="url(#heart-vessel)"
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="18"
          />
          <path
            d="M215 142c7-30 12-58 21-84 5-15 26-12 26 4v35c0 12 16 17 23 7l13-19c7-10 23-4 21 9l-10 62"
            stroke="url(#heart-vessel)"
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="15"
          />
          <path
            d="M184 153c-9 38-7 76 3 116m34-111c-10 42-7 83 9 120m-91-135c13 22 22 46 27 73m94-66c-17 19-31 42-42 69"
            stroke="#BFDBFE"
            strokeLinecap="round"
            strokeOpacity=".68"
            strokeWidth="6"
          />
          <path
            d="M76 187c-17 28-9 70 15 99m175-102c19 32 13 72-12 103"
            stroke="#38BDF8"
            strokeLinecap="round"
            strokeOpacity=".8"
            strokeWidth="5"
          />
        </svg>
      )}
    </div>
  );
}
