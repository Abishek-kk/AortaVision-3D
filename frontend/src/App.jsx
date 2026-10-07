import { lazy, Suspense } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout.jsx";
import About from "./pages/About.jsx";
import Home from "./pages/Home.jsx";

const Analyze = lazy(() => import("./pages/Analyze.jsx"));
const Guide = lazy(() => import("./pages/Guide.jsx"));

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Home />} />
        <Route
          path="/analyze"
          element={
            <Suspense
              fallback={
                <div
                  aria-live="polite"
                  className="flex flex-1 items-center justify-center bg-deepNavy text-cyan"
                  role="status"
                >
                  Loading analysis dashboard…
                </div>
              }
            >
              <Analyze />
            </Suspense>
          }
        />
        <Route
          path="/guide"
          element={
            <Suspense
              fallback={
                <div
                  aria-live="polite"
                  className="flex flex-1 items-center justify-center bg-lightBlue text-navy"
                  role="status"
                >
                  Loading educational guide…
                </div>
              }
            >
              <Guide />
            </Suspense>
          }
        />
        <Route path="/about" element={<About />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
