import { useNavigate } from "react-router-dom";
import PageHeader from "../components/PageHeader";

export default function LandingPage() {
  const navigate = useNavigate();

  return (
    <main className="min-h-screen px-6 py-12 md:px-10 lg:px-20">
      <div className="mx-auto max-w-6xl">
        <PageHeader
          title="Formula 1 Statistical Analysis Platform"
          subtitle="Demonstrate Probability and Statistics concepts with real Formula 1 race data from FastF1."
        />

        <section className="card p-8 md:p-10 bg-gradient-to-br from-f1gray to-black">
          <h2 className="text-2xl font-bold">Academic Project Scope</h2>
          <p className="mt-3 max-w-3xl text-white/75">
            Analyze race lap times, random variables, regression, distributions, and season-level comparisons using real session data.
          </p>
          <div className="mt-8 flex flex-wrap gap-4">
            <button
              onClick={() => navigate("/analyze")}
              className="rounded-lg bg-f1red px-6 py-3 font-semibold text-white hover:opacity-90"
            >
              Start Analysis
            </button>
            <button
              onClick={() => navigate("/analyze?example=1")}
              className="rounded-lg border border-white/25 px-6 py-3 font-semibold text-white hover:bg-white/10"
            >
              View Example
            </button>
          </div>
        </section>
      </div>
    </main>
  );
}
