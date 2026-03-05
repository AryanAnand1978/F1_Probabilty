import Plot from "react-plotly.js";

export default function BoxPlotChart({ values, title = "Lap Time Box Plot" }) {
  return (
    <Plot
      className="w-full"
      data={[
        {
          x: values.map(() => "Selected Driver"),
          y: values,
          type: "box",
          name: "Lap Times",
          marker: { color: "#e10600" },
          boxpoints: "outliers",
        },
      ]}
      layout={{
        title,
        paper_bgcolor: "#1f1f1f",
        plot_bgcolor: "#1f1f1f",
        font: { color: "#ffffff" },
        margin: { l: 70, r: 25, t: 55, b: 70 },
        xaxis: {
          title: { text: "Driver Group", standoff: 10 },
          automargin: true,
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
      style={{ width: "100%", height: "420px" }}
    />
  );
}
