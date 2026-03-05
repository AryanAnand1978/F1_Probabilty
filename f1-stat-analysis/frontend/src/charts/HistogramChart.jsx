import Plot from "react-plotly.js";

export default function HistogramChart({ values, title = "Lap Time Histogram" }) {
  return (
    <Plot
      className="w-full"
      data={[
        {
          x: values,
          type: "histogram",
          marker: { color: "#e10600" },
          hovertemplate: "Lap Time: %{x:.3f}s<br>Count: %{y}<extra></extra>",
        },
      ]}
      layout={{
        title,
        paper_bgcolor: "#1f1f1f",
        plot_bgcolor: "#1f1f1f",
        font: { color: "#ffffff" },
        margin: { l: 70, r: 25, t: 55, b: 70 },
        xaxis: {
          title: { text: "Lap Time (seconds)", standoff: 10 },
          automargin: true,
          showgrid: true,
          gridcolor: "rgba(255,255,255,0.08)",
        },
        yaxis: {
          title: { text: "Frequency (number of laps)", standoff: 10 },
          automargin: true,
          showgrid: true,
          gridcolor: "rgba(255,255,255,0.08)",
        },
      }}
      config={{ responsive: true, displaylogo: false }}
      useResizeHandler
      style={{ width: "100%", height: "420px" }}
    />
  );
}
