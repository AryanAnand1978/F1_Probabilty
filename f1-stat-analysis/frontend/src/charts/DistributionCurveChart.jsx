import Plot from "react-plotly.js";

export default function DistributionCurveChart({ title, traces }) {
  return (
    <Plot
      className="w-full"
      data={traces}
      layout={{
        title,
        paper_bgcolor: "#1f1f1f",
        plot_bgcolor: "#1f1f1f",
        font: { color: "#ffffff" },
        margin: { l: 70, r: 25, t: 55, b: 70 },
        xaxis: {
          title: { text: "Value (seconds / count)", standoff: 10 },
          automargin: true,
          showgrid: true,
          gridcolor: "rgba(255,255,255,0.08)",
        },
        yaxis: {
          title: { text: "Probability / Density", standoff: 10 },
          automargin: true,
          showgrid: true,
          gridcolor: "rgba(255,255,255,0.08)",
        },
        legend: { title: { text: "Distribution" } },
      }}
      config={{ responsive: true, displaylogo: false }}
      useResizeHandler
      style={{ width: "100%", height: "430px" }}
    />
  );
}
