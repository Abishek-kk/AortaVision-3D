import { Menu, X } from "lucide-react";
import { useState } from "react";
import { NavLink } from "react-router-dom";
import Logo from "./Logo.jsx";

const links = [
  { to: "/", label: "HOME", end: true },
  { to: "/analyze", label: "ANALYZE RISK" },
  { to: "/guide", label: "EDUCATIONAL GUIDE" },
  { to: "/about", label: "ABOUT" },
];

export default function Navbar() {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <header className="relative z-20 bg-white text-navy shadow-sm print-hide">
      <nav
        aria-label="Main navigation"
        className="mx-auto flex min-h-[76px] max-w-screen-2xl items-center justify-between gap-4 px-5 py-3 sm:px-8"
      >
        <NavLink
          aria-label="AortaVision 3D home"
          className="flex min-w-0 items-center gap-3"
          onClick={() => setMenuOpen(false)}
          to="/"
        >
          <Logo className="h-10 w-10 shrink-0 sm:h-11 sm:w-11" />
          <span className="max-w-[330px] text-sm font-extrabold leading-tight sm:text-base">
            AortaVision 3D · Multimodal AI Hackathon 2026 - Track A
          </span>
        </NavLink>

        <button
          aria-expanded={menuOpen}
          aria-label={menuOpen ? "Close navigation menu" : "Open navigation menu"}
          className="rounded-lg p-2 md:hidden"
          onClick={() => setMenuOpen((open) => !open)}
          type="button"
        >
          {menuOpen ? <X aria-hidden="true" /> : <Menu aria-hidden="true" />}
        </button>

        <div
          className={`${menuOpen ? "flex" : "hidden"} absolute left-0 right-0 top-full flex-col border-t border-slate-100 bg-white px-5 py-3 shadow-lg md:static md:flex md:flex-row md:items-center md:gap-7 md:border-0 md:bg-transparent md:p-0 md:shadow-none`}
        >
          {links.map(({ to, label, end }) => (
            <NavLink
              key={to}
              end={end}
              onClick={() => setMenuOpen(false)}
              className={({ isActive }) =>
                `border-b-2 py-3 text-sm font-bold tracking-wide transition-colors duration-200 md:py-2 ${
                  isActive
                    ? "border-blue text-blue"
                    : "border-transparent text-navy hover:text-blue"
                }`
              }
              to={to}
            >
              {label}
            </NavLink>
          ))}
        </div>
      </nav>
    </header>
  );
}
