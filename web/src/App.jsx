import { Routes, Route } from "react-router-dom";
import { AuthProvider } from "./hooks/useAuth";
import Nav from "./components/Nav";
import GamesPage from "./pages/GamesPage";
import PredictPage from "./pages/PredictPage";
import PricingPage from "./pages/PricingPage";
import AboutPage from "./pages/AboutPage";

export default function App() {
  return (
    <AuthProvider>
      <Nav />
      <Routes>
        <Route path="/" element={<GamesPage />} />
        <Route path="/predict" element={<PredictPage />} />
        <Route path="/pricing" element={<PricingPage />} />
        <Route path="/about" element={<AboutPage />} />
      </Routes>
    </AuthProvider>
  );
}
