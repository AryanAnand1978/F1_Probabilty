from __future__ import annotations

import traceback
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from analysis_engine import AnalysisEngine
from f1_data_loader import F1DataLoader

app = FastAPI(title="Formula 1 Statistical Analysis Platform", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

loader = F1DataLoader(cache_dir="./cache")
engine = AnalysisEngine(loader)
results_store: dict[str, dict[str, Any]] = {}


class AnalyzeRequest(BaseModel):
    year: int = Field(ge=2018, le=2025)
    race: str | int
    driver: str
    comparison_driver: str | None = None
    team1: str
    team2: str


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Formula 1 Statistical Analysis Platform API"}


@app.get("/seasons")
def get_seasons() -> dict[str, list[int]]:
    return {"seasons": loader.seasons()}


@app.get("/races/{year}")
def get_races(year: int) -> dict[str, Any]:
    if year < 2018 or year > 2025:
        raise HTTPException(status_code=400, detail="Year must be between 2018 and 2025")
    try:
        return {"year": year, "races": loader.races_for_year(year)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/session/{year}/{race}")
def get_session(year: int, race: str) -> dict[str, Any]:
    try:
        return loader.session_metadata(year, race)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/analyze")
def analyze(payload: AnalyzeRequest) -> dict[str, str]:
    job_id = str(uuid.uuid4())
    results_store[job_id] = {"job_id": job_id, "status": "running"}

    try:
        analysis_result = engine.run(
            year=payload.year,
            race=payload.race,
            driver=payload.driver,
            comparison_driver=payload.comparison_driver,
            team1=payload.team1,
            team2=payload.team2,
        )
        results_store[job_id] = {
            "job_id": job_id,
            "status": "completed",
            "result": analysis_result,
        }
    except Exception as exc:
        results_store[job_id] = {
            "job_id": job_id,
            "status": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

    return {"job_id": job_id, "status": results_store[job_id]["status"]}


@app.get("/results/{job_id}")
def get_results(job_id: str) -> dict[str, Any]:
    result = results_store.get(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Job not found")
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
