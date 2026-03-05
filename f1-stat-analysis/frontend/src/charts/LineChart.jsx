import Plot from "react-plotly.js";

export default function LineChart({ x, y, title = "Line Chart", xTitle = "X", yTitle = "Y" }) {
  return (
    <Plot
      className="w-full"
      data={[
        {
          x,
          y,
          mode: "lines+markers",
          type: "scatter",
          line: { color: "#e10600", width: 2 },
          marker: { color: "#ffffff", size: 5 },
        },
      ]}
      layout={{
        title,
        paper_bgcolor: "#1f1f1f",
        plot_bgcolor: "#1f1f1f",
        font: { color: "#ffffff" },
        margin: { l: 70, r: 25, t: 55, b: 70 },
        xaxis: {
          title: { text: xTitle, standoff: 10 },
          automargin: true,
          showgrid: true,
          gridcolor: "rgba(255,255,255,0.08)",
        },
        yaxis: {
          title: { text: yTitle, standoff: 10 },
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
