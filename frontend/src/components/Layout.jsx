import { Outlet, useLocation } from "react-router-dom";
import DisclaimerBanner from "./DisclaimerBanner.jsx";
import DisclaimerModal from "./DisclaimerModal.jsx";
import Footer from "./Footer.jsx";
import Navbar from "./Navbar.jsx";

const lightRoutes = new Set(["/guide", "/about"]);

export default function Layout() {
  const { pathname } = useLocation();
  const theme = lightRoutes.has(pathname) ? "page-light" : "page-dark";

  return (
    <div className={`flex min-h-screen flex-col ${pathname === "/guide" ? "guide-shell" : ""}`}>
      <DisclaimerBanner />
      <DisclaimerModal />
      <Navbar />
      <div className={`flex flex-1 flex-col ${theme}`}>
        <main className="flex flex-1 flex-col">
          <Outlet />
        </main>
        <Footer />
      </div>
    </div>
  );
}
