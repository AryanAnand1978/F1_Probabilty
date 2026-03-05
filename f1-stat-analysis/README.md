# Formula 1 Statistical Analysis Platform

A full-stack web application for demonstrating Probability and Statistics concepts using **real Formula 1 race data** via the **FastF1 API**.

## Tech Stack

- Frontend: React + Tailwind CSS + Plotly.js (Vite)
- Backend: FastAPI
- Data/Stats: fastf1, pandas, numpy, scipy, scikit-learn

## Folder Structure

```text
f1-stat-analysis/
  backend/
    main.py
    analysis_engine.py
    statistics_module.py
    f1_data_loader.py
    requirements.txt
  frontend/
    index.html
    package.json
    vite.config.js
    tailwind.config.js
    postcss.config.js
    src/
      App.jsx
      main.jsx
      index.css
      components/
        FormulaBlock.jsx
        PageHeader.jsx
        StatCard.jsx
      pages/
        LandingPage.jsx
        AnalysisPage.jsx
        ResultsPage.jsx
      charts/
        HistogramChart.jsx
        ScatterRegressionChart.jsx
        BoxPlotChart.jsx
        LineChart.jsx
        DistributionCurveChart.jsx
      services/
        api.js
```

## Backend Setup

```bash
cd f1-stat-analysis/backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## Frontend Setup

```bash
cd f1-stat-analysis/frontend
npm install
npm run dev
```

Optional API base URL:

```bash
# frontend/.env
VITE_API_BASE=http://localhost:8000
```

## API Endpoints

- `GET /seasons`
- `GET /races/{year}`
- `GET /session/{year}/{race}`
- `POST /analyze`
- `GET /results/{job_id}`

## Notes

- Data is loaded from FastF1 and cached in `backend/cache/`.
- Analysis includes descriptive stats, random variable analysis, correlation/regression, distributions, and hypothesis testing.
- Decision rule used in testing: reject H0 if `p-value < 0.05`.
