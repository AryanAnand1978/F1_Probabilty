from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fastf1
import numpy as np
import pandas as pd
from fastf1.ergast import Ergast


@dataclass
class SessionSelection:
    year: int
    race: str
    session_type: str = "R"


class F1DataLoader:
    def __init__(self, cache_dir: str = "./cache") -> None:
        self.cache_path = Path(cache_dir).resolve()
        self.cache_path.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(self.cache_path))
        self.ergast = Ergast()

    @staticmethod
    def seasons() -> list[int]:
        return list(range(2018, 2026))

    def races_for_year(self, year: int) -> list[dict[str, Any]]:
        schedule = fastf1.get_event_schedule(year)
        races: list[dict[str, Any]] = []
        for _, row in schedule.iterrows():
            if str(row.get("EventFormat", "")).lower() == "testing":
                continue
            races.append(
                {
                    "round": int(row["RoundNumber"]),
                    "race_name": str(row["EventName"]),
                    "country": str(row.get("Country", "")),
                    "location": str(row.get("Location", "")),
                    "date": str(row.get("EventDate", "")),
                }
            )
        return races

    def load_race_session(self, year: int, race: str | int):
        session = fastf1.get_session(year, race, "R")
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        return session

    def session_metadata(self, year: int, race: str | int) -> dict[str, Any]:
        session = self.load_race_session(year, race)
        laps = session.laps.copy()
        laps = laps[laps["LapTime"].notna()]

        drivers: list[dict[str, str]] = []
        for driver_code in sorted(laps["Driver"].dropna().unique().tolist()):
            driver_laps = laps[laps["Driver"] == driver_code]
            team_name = str(driver_laps["Team"].mode().iloc[0]) if not driver_laps.empty else "Unknown"
            drivers.append({"code": str(driver_code), "team": team_name})

        teams = sorted(laps["Team"].dropna().astype(str).unique().tolist())
        return {
            "year": year,
            "race": race,
            "drivers": drivers,
            "teams": teams,
        }

    @staticmethod
    def _lap_seconds(laps: pd.DataFrame) -> list[float]:
        clean = laps[laps["LapTime"].notna()].copy()
        return clean["LapTime"].dt.total_seconds().tolist()

    @staticmethod
    def _ergast_content_to_df(response: Any) -> pd.DataFrame:
        content = getattr(response, "content", None)
        if content is None:
            return pd.DataFrame()

        if isinstance(content, pd.DataFrame):
            return content.copy()

        if isinstance(content, list):
            if not content:
                return pd.DataFrame()
            if all(isinstance(item, pd.DataFrame) for item in content):
                return pd.concat(content, ignore_index=True)
            try:
                return pd.DataFrame(content)
            except Exception:
                return pd.DataFrame()

        try:
            return pd.DataFrame(content)
        except Exception:
            return pd.DataFrame()

    def driver_laps(self, session, driver_code: str) -> pd.DataFrame:
        laps = session.laps.copy()
        driver_df = laps[(laps["Driver"] == driver_code) & laps["LapTime"].notna()].copy()
        return driver_df

    def team_laps(self, session, team_name: str) -> pd.DataFrame:
        laps = session.laps.copy()
        return laps[(laps["Team"] == team_name) & laps["LapTime"].notna()].copy()

    def team_pit_stop_times(self, session, team_name: str) -> list[float]:
        team_laps = self.team_laps(session, team_name)
        has_pit = team_laps[team_laps["PitInTime"].notna() & team_laps["PitOutTime"].notna()].copy()
        if has_pit.empty:
            return []
        durations = (has_pit["PitOutTime"] - has_pit["PitInTime"]).dt.total_seconds()
        durations = durations[(durations > 0) & np.isfinite(durations)]
        return durations.tolist()

    def pit_stops_per_driver(self, session) -> dict[str, int]:
        laps = session.laps.copy()
        out_laps = laps[laps["PitOutTime"].notna()]
        return out_laps.groupby("Driver").size().astype(int).to_dict()

    def dnf_data(self, session) -> list[dict[str, str]]:
        results = session.results.copy()
        if results is None or results.empty:
            return []

        dnf_rows: list[dict[str, str]] = []
        for _, row in results.iterrows():
            status = str(row.get("Status", ""))
            if "Finished" in status or status.startswith("+"):
                continue
            dnf_rows.append(
                {
                    "driver": str(row.get("Abbreviation", row.get("DriverNumber", ""))),
                    "team": str(row.get("TeamName", "Unknown")),
                    "status": status,
                }
            )
        return dnf_rows

    def season_podium_stats(self, year: int, driver_code: str) -> dict[str, int]:
        race_results = self.ergast.get_race_results(season=year, limit=1000)
        content = self._ergast_content_to_df(race_results)
        if content.empty:
            return {"races": 0, "podiums": 0}

        races = int(content["round"].nunique()) if "round" in content.columns else 0
        if not {"driverCode", "position"}.issubset(content.columns):
            return {"races": races, "podiums": 0}

        positions = pd.to_numeric(content["position"], errors="coerce")
        podiums = content[(content["driverCode"] == driver_code) & (positions <= 3)].shape[0]
        return {"races": races, "podiums": int(podiums)}

    def mechanical_failure_intervals(self, year: int, driver_code: str) -> list[int]:
        race_results = self.ergast.get_race_results(season=year, limit=1000)
        content = self._ergast_content_to_df(race_results)
        if content.empty or not {"status", "driverCode", "round"}.issubset(content.columns):
            return []

        status_col = content["status"].astype(str).str.lower()
        mechanical_mask = status_col.str.contains(
            "engine|gearbox|brake|hydraulic|electrical|power unit|oil|water|suspension|mechanical",
            regex=True,
        )

        dnf_rounds = (
            content[(content["driverCode"] == driver_code) & mechanical_mask]["round"]
            .astype(int)
            .sort_values()
            .tolist()
        )
        if len(dnf_rounds) < 2:
            return []

        return [dnf_rounds[i] - dnf_rounds[i - 1] for i in range(1, len(dnf_rounds))]

    @staticmethod
    def lap_times_from_df(df: pd.DataFrame) -> list[float]:
        return df["LapTime"].dt.total_seconds().tolist() if not df.empty else []
