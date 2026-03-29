import { useEffect, useMemo, useState } from "react";
import Plot from "react-plotly.js";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import BoxPlotChart from "../charts/BoxPlotChart";
import DistributionCurveChart from "../charts/DistributionCurveChart";
import HistogramChart from "../charts/HistogramChart";
import FormulaBlock from "../components/FormulaBlock";
import PageHeader from "../components/PageHeader";
import StatCard from "../components/StatCard";
import { fetchResults } from "../services/api";

const tabs = [
  "Descriptive Statistics",
  "Race Driver Comparison",
  "Team Comparison",
  "Season Comparison",
  "Probability Distributions",
];

const chartLayout = (title, xTitle, yTitle, extra = {}) => {
  const { xaxis: extraXaxis = {}, yaxis: extraYaxis = {}, ...rest } = extra;
  return {
    title,
    paper_bgcolor: "#1f1f1f",
    plot_bgcolor: "#1f1f1f",
    font: { color: "#ffffff" },
    margin: { l: 70, r: 25, t: 55, b: 70 },
    hovermode: "x unified",
    hoverlabel: {
      bgcolor: "#111111",
      bordercolor: "rgba(255,255,255,0.15)",
      font: { color: "#ffffff" },
    },
    dragmode: "pan",
    xaxis: {
      title: { text: xTitle, standoff: 10 },
      automargin: true,
      showgrid: true,
      gridcolor: "rgba(255,255,255,0.08)",
      showspikes: true,
      spikemode: "across",
      spikecolor: "rgba(255,255,255,0.35)",
      spikethickness: 1,
      rangeslider: { visible: false },
      ...extraXaxis,
    },
    yaxis: {
      title: { text: yTitle, standoff: 10 },
      automargin: true,
      showgrid: true,
      gridcolor: "rgba(255,255,255,0.08)",
      showspikes: true,
      spikemode: "across",
      spikecolor: "rgba(255,255,255,0.25)",
      spikethickness: 1,
      ...extraYaxis,
    },
    legend: { orientation: "h", y: 1.12 },
    ...rest,
  };
};

function PlotCard({ data, layout, height = 430 }) {
  return (
    <div className="card p-4">
      <Plot
        className="w-full"
        data={data}
        layout={layout}
        config={{
          responsive: true,
          displaylogo: false,
          scrollZoom: true,
          doubleClick: "reset+autosize",
          modeBarButtonsToAdd: ["hoverclosest", "hovercompare"],
        }}
        useResizeHandler
        revision={JSON.stringify({ title: layout?.title, traces: data?.length })}
        style={{ width: "100%", height: `${height}px` }}
      />
      <p className="mt-3 text-xs text-white/50">
        Drag to pan, scroll to zoom, double-click to reset, and use the modebar to switch hover modes.
      </p>
    </div>
  );
}

function EmptyStateCard({ message }) {
  return <p className="card p-4 text-sm text-white/80">{message}</p>;
}

function formatNumber(value, suffix = "") {
  if (value === null || value === undefined || Number.isNaN(value)) return "N/A";
  return `${Number(value).toFixed(4)}${suffix}`;
}

function average(values = []) {
  if (!values.length) return null;
  return values.reduce((sum, value) => sum + Number(value || 0), 0) / values.length;
}

function stdDev(values = []) {
  if (!values.length) return null;
  const mean = average(values);
  const variance = values.reduce((sum, value) => sum + (Number(value || 0) - mean) ** 2, 0) / values.length;
  return Math.sqrt(variance);
}

function linearSlope(values = []) {
  if (values.length < 2) return null;
  const xs = values.map((_, index) => index + 1);
  const xMean = average(xs);
  const yMean = average(values);
  const numerator = xs.reduce((sum, x, index) => sum + (x - xMean) * (Number(values[index] || 0) - yMean), 0);
  const denominator = xs.reduce((sum, x) => sum + (x - xMean) ** 2, 0);
  if (!denominator) return null;
  return numerator / denominator;
}

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
  const driverTrends = payload?.driver_trends;
  const teamComparison = payload?.team_comparison;
  const seasonComparison = payload?.season_comparison;

  const distTraces = useMemo(() => {
    if (!pd) return [];
    return [
      { x: pd.normal.curve.x, y: pd.normal.curve.y, mode: "lines", name: "Normal", line: { color: "#e10600", width: 3 } },
      { x: pd.binomial.curve.x, y: pd.binomial.curve.y, mode: "lines+markers", name: "Binomial", line: { color: "#ffffff" } },
      { x: pd.poisson.curve.x, y: pd.poisson.curve.y, mode: "lines+markers", name: "Poisson", line: { color: "#f3c623" } },
      { x: pd.exponential.curve.x, y: pd.exponential.curve.y, mode: "lines", name: "Exponential", line: { color: "#3ac47d", width: 3 } },
    ];
  }, [pd]);

  const selectedTrend = driverTrends?.selected_driver;
  const comparisonTrend = driverTrends?.comparison_driver;
  const comparisonSummary = driverTrends?.comparison_summary;

  const selectedFitData = selectedTrend
    ? [
        {
          x: selectedTrend.original_scatter.x,
          y: selectedTrend.original_scatter.y,
          mode: "markers",
          type: "scatter",
          name: `${meta.driver} raw laps`,
          marker: { color: "#9ca3af", size: 6, opacity: 0.7 },
        },
        {
          x: selectedTrend.smoothed_line.x,
          y: selectedTrend.smoothed_line.y,
          mode: "lines",
          type: "scatter",
          name: `${meta.driver} moving average`,
          line: { color: "#ffffff", width: 2 },
        },
        {
          x: selectedTrend.linear_fit.x,
          y: selectedTrend.linear_fit.y,
          mode: "lines",
          type: "scatter",
          name: `${meta.driver} linear fit`,
          line: { color: "#f97316", width: 2, dash: "dash" },
        },
        {
          x: selectedTrend.polynomial_fit.x,
          y: selectedTrend.polynomial_fit.y,
          mode: "lines",
          type: "scatter",
          name: `${meta.driver} piecewise fit`,
          line: { color: "#e10600", width: 3 },
        },
      ]
    : [];

  const comparisonFitData = comparisonTrend
    ? [
        {
          x: comparisonTrend.original_scatter.x,
          y: comparisonTrend.original_scatter.y,
          mode: "markers",
          type: "scatter",
          name: `${meta.comparison_driver} raw laps`,
          marker: { color: "#9ca3af", size: 6, opacity: 0.7 },
        },
        {
          x: comparisonTrend.smoothed_line.x,
          y: comparisonTrend.smoothed_line.y,
          mode: "lines",
          type: "scatter",
          name: `${meta.comparison_driver} moving average`,
          line: { color: "#ffffff", width: 2 },
        },
        {
          x: comparisonTrend.linear_fit.x,
          y: comparisonTrend.linear_fit.y,
          mode: "lines",
          type: "scatter",
          name: `${meta.comparison_driver} linear fit`,
          line: { color: "#f97316", width: 2, dash: "dash" },
        },
        {
          x: comparisonTrend.polynomial_fit.x,
          y: comparisonTrend.polynomial_fit.y,
          mode: "lines",
          type: "scatter",
          name: `${meta.comparison_driver} piecewise fit`,
          line: { color: "#3b82f6", width: 3 },
        },
      ]
    : [];

  const combinedDriverData =
    selectedTrend && comparisonTrend
      ? [
          {
            x: selectedTrend.smoothed_line.x,
            y: selectedTrend.smoothed_line.y,
            mode: "lines",
            type: "scatter",
            name: `${meta.driver} moving average`,
            line: { color: "#e10600", width: 3 },
          },
          {
            x: comparisonTrend.smoothed_line.x,
            y: comparisonTrend.smoothed_line.y,
            mode: "lines",
            type: "scatter",
            name: `${meta.comparison_driver} moving average`,
            line: { color: "#3b82f6", width: 3 },
          },
          {
            x: selectedTrend.polynomial_fit.x,
            y: selectedTrend.polynomial_fit.y,
            mode: "lines",
            type: "scatter",
            name: `${meta.driver} piecewise fit`,
            line: { color: "#f97316", width: 2, dash: "dot" },
          },
          {
            x: comparisonTrend.polynomial_fit.x,
            y: comparisonTrend.polynomial_fit.y,
            mode: "lines",
            type: "scatter",
            name: `${meta.comparison_driver} piecewise fit`,
            line: { color: "#22c55e", width: 2, dash: "dot" },
          },
        ]
      : [];

  const teamHistogramData = teamComparison
    ? [
        {
          x: teamComparison.lap_time_histogram.team1.values,
          type: "histogram",
          name: teamComparison.lap_time_histogram.team1.name,
          opacity: 0.65,
          marker: { color: "#e10600" },
        },
        {
          x: teamComparison.lap_time_histogram.team2.values,
          type: "histogram",
          name: teamComparison.lap_time_histogram.team2.name,
          opacity: 0.55,
          marker: { color: "#3b82f6" },
        },
      ]
    : [];

  const teamBoxplotData = teamComparison
    ? [
        {
          y: teamComparison.lap_time_boxplot.team1.values,
          type: "box",
          name: teamComparison.lap_time_boxplot.team1.name,
          marker: { color: "#e10600" },
          boxpoints: "outliers",
        },
        {
          y: teamComparison.lap_time_boxplot.team2.values,
          type: "box",
          name: teamComparison.lap_time_boxplot.team2.name,
          marker: { color: "#3b82f6" },
          boxpoints: "outliers",
        },
      ]
    : [];

  const seasonDrivers = seasonComparison?.drivers || {};
  const seasonA = seasonDrivers[meta?.driver];
  const seasonB = seasonDrivers[meta?.comparison_driver];
  const seasonMath = useMemo(() => {
    if (!seasonA || !seasonB) return null;

    const pointsGap = Number(seasonA.total_points || 0) - Number(seasonB.total_points || 0);
    const finishGap =
      seasonA.average_finish !== null && seasonB.average_finish !== null
        ? Number(seasonA.average_finish) - Number(seasonB.average_finish)
        : null;
    const aPointsPerRound = average(seasonA.points_by_round || []);
    const bPointsPerRound = average(seasonB.points_by_round || []);
    const aFinishStd = stdDev(seasonA.finishing_positions || []);
    const bFinishStd = stdDev(seasonB.finishing_positions || []);
    const aPointsSlope = linearSlope(seasonA.cumulative_points || []);
    const bPointsSlope = linearSlope(seasonB.cumulative_points || []);

    return {
      pointsGap,
      finishGap,
      aPointsPerRound,
      bPointsPerRound,
      aFinishStd,
      bFinishStd,
      aPointsSlope,
      bPointsSlope,
    };
  }, [seasonA, seasonB]);
  const hasSeasonComparisonData =
    Boolean(seasonA?.race_names?.length) &&
    Boolean(seasonB?.race_names?.length) &&
    Boolean(seasonA?.cumulative_points?.length) &&
    Boolean(seasonB?.cumulative_points?.length);
  const seasonPointsData =
    seasonA && seasonB
      ? [
          {
            x: seasonA.race_names,
            y: seasonA.cumulative_points,
            mode: "lines+markers",
            type: "scatter",
            name: `${meta.driver} cumulative points`,
            line: { color: "#e10600", width: 3 },
          },
          {
            x: seasonB.race_names,
            y: seasonB.cumulative_points,
            mode: "lines+markers",
            type: "scatter",
            name: `${meta.comparison_driver} cumulative points`,
            line: { color: "#3b82f6", width: 3 },
          },
        ]
      : [];

  const seasonFinishData =
    seasonA && seasonB
      ? [
          {
            x: seasonA.race_names,
            y: seasonA.finishing_positions,
            mode: "lines+markers",
            type: "scatter",
            name: `${meta.driver} finish`,
            line: { color: "#e10600", width: 3 },
          },
          {
            x: seasonB.race_names,
            y: seasonB.finishing_positions,
            mode: "lines+markers",
            type: "scatter",
            name: `${meta.comparison_driver} finish`,
            line: { color: "#3b82f6", width: 3 },
          },
        ]
      : [];

  if (error) {
    return (
      <main className="min-h-screen p-8">
        <p className="text-red-300">{error}</p>
        <button className="mt-4 rounded bg-f1red px-4 py-2" onClick={() => navigate("/analyze")}>
          Back
        </button>
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
          <StatCard title="Smoothed Correlation (r)" value={cr.correlation_coefficient} subtitle={cr.equation} />
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
            <p className="card p-4 text-sm text-white/80">
              Random Variable: {rv.definition}. E[X] = {rv.expectation}, Var(X) = {rv.variance}, Covariance with {rv.comparison_driver || "comparison driver"} = {rv.covariance_with_comparison_driver ?? "N/A"}
            </p>
          </section>
        ) : null}

        {tab === "Race Driver Comparison" ? (
          <section className="space-y-5">
            <div className="grid gap-4 md:grid-cols-4">
              <StatCard title={`${meta.driver} Mean`} value={`${formatNumber(comparisonSummary?.summary?.[meta.driver]?.mean, "s")}`} />
              <StatCard title={`${meta.comparison_driver} Mean`} value={`${formatNumber(comparisonSummary?.summary?.[meta.comparison_driver]?.mean, "s")}`} />
              <StatCard title="Mean Delta" value={`${formatNumber(comparisonSummary?.mean_delta_seconds, "s")}`} subtitle={`${meta.driver} - ${meta.comparison_driver}`} />
              <StatCard title="Outliers Removed" value={selectedTrend.parameters.removed_outliers + comparisonTrend.parameters.removed_outliers} subtitle={`Threshold ${selectedTrend.parameters.outlier_threshold}s`} />
            </div>

            <PlotCard
              data={combinedDriverData}
              layout={chartLayout("Race Pace Comparison", "Lap Number", "Lap Time (seconds)")}
            />

            <div className="grid gap-5">
              <PlotCard
                data={selectedFitData}
                layout={chartLayout(`${meta.driver}: Raw vs Smoothed vs Linear vs Piecewise Fit`, "Lap Number", "Lap Time (seconds)")}
                height={500}
              />
              <PlotCard
                data={comparisonFitData}
                layout={chartLayout(`${meta.comparison_driver}: Raw vs Smoothed vs Linear vs Piecewise Fit`, "Lap Number", "Lap Time (seconds)")}
                height={500}
              />
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <FormulaBlock
                title={`${meta.driver} Stint-Aware Model`}
                formula={selectedTrend.polynomial_fit.equation}
                note={`Linear R^2 = ${selectedTrend.linear_fit.r_squared}, Piecewise R^2 = ${selectedTrend.polynomial_fit.r_squared}, Segments = ${selectedTrend.parameters.stint_segments}`}
              />
              <FormulaBlock
                title={`${meta.comparison_driver} Stint-Aware Model`}
                formula={comparisonTrend.polynomial_fit.equation}
                note={`Linear R^2 = ${comparisonTrend.linear_fit.r_squared}, Piecewise R^2 = ${comparisonTrend.polynomial_fit.r_squared}, Segments = ${comparisonTrend.parameters.stint_segments}`}
              />
            </div>
            <p className="card p-4 text-sm text-white/80">{comparisonSummary?.interpretation || "Driver comparison summary is unavailable."}</p>
            <p className="card p-4 text-sm text-white/80">
              Smoothing used a moving average window of {selectedTrend.parameters.smoothing_window}; laps above {selectedTrend.parameters.outlier_threshold}s were removed before fitting a piecewise polynomial model separately to each stint.
            </p>
          </section>
        ) : null}

        {tab === "Team Comparison" ? (
          <section className="space-y-5">
            <div className="grid gap-4 md:grid-cols-4">
              <StatCard title={`${meta.team1} Mean Lap`} value={`${formatNumber(teamComparison?.summary?.[meta.team1]?.mean_lap, "s")}`} />
              <StatCard title={`${meta.team2} Mean Lap`} value={`${formatNumber(teamComparison?.summary?.[meta.team2]?.mean_lap, "s")}`} />
              <StatCard title={`${meta.team1} Avg Pit`} value={formatNumber(teamComparison?.summary?.[meta.team1]?.avg_pit_time, "s")} />
              <StatCard title={`${meta.team2} Avg Pit`} value={formatNumber(teamComparison?.summary?.[meta.team2]?.avg_pit_time, "s")} />
            </div>
            <div className="grid gap-5 lg:grid-cols-2">
              <PlotCard
                data={teamHistogramData}
                layout={chartLayout("Team Lap Time Distribution", "Lap Time (seconds)", "Frequency", { barmode: "overlay" })}
              />
              <PlotCard
                data={teamBoxplotData}
                layout={chartLayout("Team Lap Time Box Plot", "Team", "Lap Time (seconds)")}
              />
            </div>
            <p className="card p-4 text-sm text-white/80">{teamComparison?.interpretation || "Team comparison summary is unavailable."}</p>
          </section>
        ) : null}

        {tab === "Season Comparison" ? (
          <section className="space-y-5">
            <div className="grid gap-4 md:grid-cols-4">
              <StatCard title={`${meta.driver} Total Points`} value={seasonA?.total_points ?? "N/A"} />
              <StatCard title={`${meta.comparison_driver} Total Points`} value={seasonB?.total_points ?? "N/A"} />
              <StatCard title={`${meta.driver} Avg Finish`} value={seasonA?.average_finish ?? "N/A"} />
              <StatCard title={`${meta.comparison_driver} Avg Finish`} value={seasonB?.average_finish ?? "N/A"} />
            </div>
            {hasSeasonComparisonData && seasonMath ? (
              <>
                <div className="grid gap-4 md:grid-cols-4">
                  <StatCard
                    title="Points Gap"
                    value={formatNumber(seasonMath.pointsGap)}
                    subtitle={`${meta.driver} - ${meta.comparison_driver}`}
                  />
                  <StatCard
                    title="Avg Finish Gap"
                    value={formatNumber(seasonMath.finishGap)}
                    subtitle={`${meta.driver} - ${meta.comparison_driver}`}
                  />
                  <StatCard
                    title={`${meta.driver} Pts/Race`}
                    value={formatNumber(seasonMath.aPointsPerRound)}
                  />
                  <StatCard
                    title={`${meta.comparison_driver} Pts/Race`}
                    value={formatNumber(seasonMath.bPointsPerRound)}
                  />
                </div>

                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                  <FormulaBlock
                    title="Points Gap"
                    formula={`Delta_P = P_${meta.driver} - P_${meta.comparison_driver} = ${formatNumber(seasonMath.pointsGap)}`}
                    note="Positive value means the selected driver leads on total points."
                  />
                  <FormulaBlock
                    title="Average Finish"
                    formula={`mu_f = (sum f_i) / n`}
                    note={`${meta.driver} = ${formatNumber(seasonA.average_finish)}, ${meta.comparison_driver} = ${formatNumber(seasonB.average_finish)}`}
                  />
                  <FormulaBlock
                    title="Finish Consistency"
                    formula={`sigma_f = sqrt(sum((f_i - mu_f)^2) / n)`}
                    note={`${meta.driver} = ${formatNumber(seasonMath.aFinishStd)}, ${meta.comparison_driver} = ${formatNumber(seasonMath.bFinishStd)}`}
                  />
                  <FormulaBlock
                    title="Points Trend"
                    formula={`m = Cov(round, cumulative_points) / Var(round)`}
                    note={`${meta.driver} slope = ${formatNumber(seasonMath.aPointsSlope)}, ${meta.comparison_driver} slope = ${formatNumber(seasonMath.bPointsSlope)}`}
                  />
                </div>
              </>
            ) : null}
            {hasSeasonComparisonData ? (
              <div className="grid gap-5">
                <PlotCard
                  data={seasonPointsData}
                  layout={chartLayout("Season Cumulative Points", "Race", "Cumulative Points")}
                  height={500}
                />
                <PlotCard
                  data={seasonFinishData}
                  layout={chartLayout("Season Finishing Positions", "Race", "Finishing Position", {
                    yaxis: { autorange: "reversed" },
                  })}
                  height={500}
                />
              </div>
            ) : (
              <EmptyStateCard message="Season comparison data is unavailable for the selected drivers. Run a fresh analysis after restarting the backend." />
            )}
            <p className="card p-4 text-sm text-white/80">
              {seasonComparison.interpretation}
              {hasSeasonComparisonData && seasonMath
                ? ` ${meta.driver} scores ${formatNumber(seasonMath.aPointsPerRound)} points per round versus ${formatNumber(seasonMath.bPointsPerRound)} for ${meta.comparison_driver}. Finish-position volatility is ${formatNumber(seasonMath.aFinishStd)} versus ${formatNumber(seasonMath.bFinishStd)}.`
                : ""}
            </p>
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
