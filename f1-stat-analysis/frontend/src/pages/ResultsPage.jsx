import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import BoxPlotChart from "../charts/BoxPlotChart";
import DistributionCurveChart from "../charts/DistributionCurveChart";
import HistogramChart from "../charts/HistogramChart";
import LineChart from "../charts/LineChart";
import ScatterRegressionChart from "../charts/ScatterRegressionChart";
import FormulaBlock from "../components/FormulaBlock";
import PageHeader from "../components/PageHeader";
import StatCard from "../components/StatCard";
import { fetchResults } from "../services/api";

const tabs = [
  "Descriptive Statistics",
  "Correlation & Regression",
  "Probability Distributions",
  "Hypothesis Testing",
];

export default function ResultsPage() {
  const { jobId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();

  const [tab, setTab] = useState(tabs[0]);
  const [payload, setPayload] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    async function load() {
      try {
        const data = await fetchResults(jobId);
        if (data.status !== "completed") {
          throw new Error(data.error || "Analysis is not completed yet.");
        }
        setPayload(data.result);
      } catch (e) {
        setError(e?.response?.data?.detail || e.message);
      }
    }
    load();
  }, [jobId]);

  const meta = payload?.meta;
  const ds = payload?.descriptive_statistics;
  const rv = payload?.random_variables;
  const cr = payload?.correlation_regression;
  const pd = payload?.probability_distributions;
  const ht = payload?.hypothesis_testing;

  const distTraces = useMemo(() => {
    if (!pd) return [];
    return [
      { x: pd.normal.curve.x, y: pd.normal.curve.y, mode: "lines", name: "Normal", line: { color: "#e10600", width: 3 } },
      { x: pd.binomial.curve.x, y: pd.binomial.curve.y, mode: "lines+markers", name: "Binomial", line: { color: "#ffffff" } },
      { x: pd.poisson.curve.x, y: pd.poisson.curve.y, mode: "lines+markers", name: "Poisson", line: { color: "#f3c623" } },
      { x: pd.exponential.curve.x, y: pd.exponential.curve.y, mode: "lines", name: "Exponential", line: { color: "#3ac47d", width: 3 } },
    ];
  }, [pd]);

  if (error) {
    return (
      <main className="min-h-screen p-8">
        <p className="text-red-300">{error}</p>
        <button className="mt-4 rounded bg-f1red px-4 py-2" onClick={() => navigate("/analyze")}>Back</button>
      </main>
    );
  }

  if (!payload) {
    return <main className="min-h-screen p-8 text-white/80">Loading results...</main>;
  }

  return (
    <main className="min-h-screen px-6 py-8 md:px-10 lg:px-20">
      <div className="mx-auto max-w-7xl">
        <PageHeader
          title="Results Dashboard"
          subtitle={`${meta.season} ${meta.race} | Driver: ${meta.driver} | Comparison Driver: ${meta.comparison_driver || "N/A"} | Teams: ${meta.team1} vs ${meta.team2}`}
        />

        <div className="mb-6 grid gap-4 md:grid-cols-4">
          <StatCard title="Analyzed Laps" value={meta.laps_analyzed} />
          <StatCard title="Mean Lap Time" value={`${ds.summary.mean}s`} />
          <StatCard title="Std Deviation" value={`${ds.summary.std_dev}s`} />
          <StatCard title="Correlation (r)" value={cr.correlation_coefficient} subtitle={cr.equation} />
        </div>

        <div className="mb-6 flex flex-wrap gap-3">
          {tabs.map((name) => (
            <button
              key={name}
              onClick={() => setTab(name)}
              className={`tab-btn ${tab === name ? "tab-btn-active" : "tab-btn-inactive"}`}
            >
              {name}
            </button>
          ))}
        </div>

        {tab === "Descriptive Statistics" ? (
          <section className="space-y-5">
            <div className="grid gap-5 lg:grid-cols-2">
              <div className="card p-4"><HistogramChart values={ds.histogram.values} /></div>
              <div className="card p-4"><BoxPlotChart values={ds.boxplot.values} /></div>
            </div>
            <div className="grid gap-4 md:grid-cols-3">
              <FormulaBlock title="Mean" formula={ds.formulas.mean} />
              <FormulaBlock title="Variance" formula={ds.formulas.variance} />
              <FormulaBlock title="Standard Deviation" formula={ds.formulas.std_dev} />
            </div>
            <p className="card p-4 text-sm text-white/80">{ds.interpretation}</p>
            <p className="card p-4 text-sm text-white/80">Random Variable: {rv.definition}. E[X] = {rv.expectation}, Var(X) = {rv.variance}, Covariance with {rv.comparison_driver || "comparison driver"} = {rv.covariance_with_comparison_driver ?? "N/A"}</p>
          </section>
        ) : null}

        {tab === "Correlation & Regression" ? (
          <section className="space-y-5">
            <div className="card p-4">
              <ScatterRegressionChart scatter={cr.scatter} line={cr.regression_line} />
            </div>
            <div className="card p-4">
              <LineChart
                title="Lap Number vs Lap Time"
                x={cr.scatter.x}
                y={cr.scatter.y}
                xTitle="Lap Number"
                yTitle="Lap Time (s)"
              />
            </div>
            <div className="grid gap-4 md:grid-cols-2">
              <FormulaBlock title="Correlation Formula" formula={cr.formulas.correlation} note={`r = ${cr.correlation_coefficient}, p = ${cr.correlation_p_value}`} />
              <FormulaBlock title="Regression Formula" formula={cr.formulas.regression} note={cr.equation} />
            </div>
            <p className="card p-4 text-sm text-white/80">{cr.interpretation}</p>
          </section>
        ) : null}

        {tab === "Probability Distributions" ? (
          <section className="space-y-5">
            <div className="card p-4">
              <DistributionCurveChart title="Distribution Curves" traces={distTraces} />
            </div>
            <div className="grid gap-4 md:grid-cols-2">
              <FormulaBlock title="Normal Distribution" formula={pd.normal.formula} note={`mu = ${pd.normal.parameters.mu}, sigma = ${pd.normal.parameters.sigma}`} />
              <FormulaBlock title="Binomial Distribution" formula={pd.binomial.formula} note={`n = ${pd.binomial.parameters.n}, p = ${pd.binomial.parameters.p}`} />
              <FormulaBlock title="Poisson Distribution" formula={pd.poisson.formula} note={`lambda = ${pd.poisson.parameters.lambda}`} />
              <FormulaBlock title="Exponential Distribution" formula={pd.exponential.formula} note={`lambda = ${pd.exponential.parameters.lambda}`} />
            </div>
            <p className="card p-4 text-sm text-white/80">Distribution analysis models podium outcomes (Binomial), pit-stop count (Poisson), lap time spread (Normal), and failure intervals (Exponential).</p>
          </section>
        ) : null}

        {tab === "Hypothesis Testing" ? (
          <section className="space-y-5">
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              {Object.entries(ht)
                .filter(([k]) => ["z_test", "t_test", "chi_square", "anova"].includes(k))
                .map(([name, test]) => (
                  <div key={name} className="card p-4">
                    <p className="text-sm font-semibold text-f1red uppercase">{name.replace("_", " ")}</p>
                    {test.error ? (
                      <p className="mt-2 text-sm text-white/75">{test.error}</p>
                    ) : (
                      <>
                        <p className="mt-2 text-sm text-white/75">Statistic: {test.statistic}</p>
                        <p className="text-sm text-white/75">p-value: {test.p_value}</p>
                        <p className="text-sm text-white/75">Decision: {test.decision}</p>
                      </>
                    )}
                  </div>
                ))}
            </div>
            <p className="card p-4 text-sm text-white/80">{ht.decision_rule}</p>
          </section>
        ) : null}

        <button
          className="mt-8 rounded-lg border border-white/20 px-5 py-2 text-sm hover:bg-white/10"
          onClick={() => navigate("/analyze", { state: location.state })}
        >
          Run New Analysis
        </button>
      </div>
    </main>
  );
}
