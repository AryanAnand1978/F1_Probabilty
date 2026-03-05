import Plot from "react-plotly.js";

export default function ScatterRegressionChart({ scatter, line, title = "Correlation & Regression" }) {
  return (
    <Plot
      className="w-full"
      data={[
        {
          x: scatter?.x || [],
          y: scatter?.y || [],
          mode: "markers",
          type: "scatter",
          name: "Observed Laps",
          marker: { color: "#ffffff", size: 7, opacity: 0.75 },
        },
        {
          x: line?.x || [],
          y: line?.y || [],
          mode: "lines",
          type: "scatter",
          name: "Regression Line",
          line: { color: "#e10600", width: 3 },
        },
      ]}
      layout={{
        title,
        paper_bgcolor: "#1f1f1f",
        plot_bgcolor: "#1f1f1f",
        font: { color: "#ffffff" },
        margin: { l: 70, r: 25, t: 55, b: 70 },
        xaxis: {
          title: { text: "Lap Number", standoff: 10 },
          automargin: true,
          showgrid: true,
          gridcolor: "rgba(255,255,255,0.08)",
        },
        yaxis: {
          title: { text: "Lap Time (seconds)", standoff: 10 },
          automargin: true,
          showgrid: true,
          gridcolor: "rgba(255,255,255,0.08)",
        },
      }}
      config={{ responsive: true, displaylogo: false }}
      useResizeHandler
      style={{ width: "100%", height: "430px" }}
    />
  );
}
