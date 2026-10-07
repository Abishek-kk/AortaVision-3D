/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        navy: "#0B1F4B",
        deepNavy: "#04122E",
        midNavy: "#0A2A66",
        cyan: "#38BDF8",
        blue: "#1D4ED8",
        yellow: "#FFD54A",
        "risk-red": "#FF3B5C",
        "risk-yellow": "#FACC15",
        "risk-green": "#22C55E",
        lightBlue: "#DCEBFA",
      },
      fontFamily: {
        sans: ['"Source Sans 3"', "Inter", "sans-serif"],
      },
      boxShadow: {
        glow: "0 0 24px rgba(56, 189, 248, 0.35)",
        "glow-strong": "0 0 32px rgba(56, 189, 248, 0.55)",
      },
      borderRadius: {
        "4xl": "2rem",
      },
      transitionDuration: {
        250: "250ms",
      },
    },
  },
  plugins: [],
};
