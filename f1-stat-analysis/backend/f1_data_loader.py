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

    @staticmethod
    def _normalize_nested_df(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        try:
            normalized = pd.json_normalize(df.to_dict(orient="records"), sep=".")
            return normalized if not normalized.empty else df.copy()
        except Exception:
            return df.copy()

    @staticmethod
    def _flatten_race_result_content(response: Any) -> pd.DataFrame:
        content = getattr(response, "content", None)
        description = getattr(response, "description", None)
        if content is None:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        content_items = content if isinstance(content, list) else [content]
        description_records: list[dict[str, Any]] = []

        raw_description_items = description if isinstance(description, list) else [description]
        for desc_item in raw_description_items:
            desc_df = F1DataLoader._normalize_nested_df(
                F1DataLoader._ergast_content_to_df(type("Response", (), {"content": desc_item})())
            )
            if not desc_df.empty:
                description_records.extend(desc_df.to_dict(orient="records"))
            elif isinstance(desc_item, dict):
                description_records.append(desc_item)

        # FastF1 Ergast responses often split race metadata into `description`
        # and the result rows into `content`. Stitch them back together so
        # each result row carries the round and race info.
        if description_records and len(description_records) == len(content_items):
            for desc_row, content_item in zip(description_records, content_items):
                content_df = F1DataLoader._normalize_nested_df(
                    F1DataLoader._ergast_content_to_df(type("Response", (), {"content": content_item})())
                )
                if not content_df.empty:
                    merged_df = content_df.copy()
                    for key, value in desc_row.items():
                        if key not in merged_df.columns:
                            merged_df[key] = value
                    rows.extend(merged_df.to_dict(orient="records"))
                    continue

                if isinstance(content_item, dict):
                    nested_results = content_item.get("Results") or content_item.get("results") or []
                    if isinstance(nested_results, list) and nested_results:
                        for result in nested_results:
                            if isinstance(result, dict):
                                rows.append({**desc_row, **result})
                    else:
                        rows.append({**desc_row, **content_item})
                    continue

                if isinstance(content_item, pd.DataFrame):
                    merged_df = F1DataLoader._normalize_nested_df(content_item)
                    for key, value in desc_row.items():
                        if key not in merged_df.columns:
                            merged_df[key] = value
                    rows.extend(merged_df.to_dict(orient="records"))
            if rows:
                return pd.DataFrame(rows)

        direct_df = F1DataLoader._normalize_nested_df(F1DataLoader._ergast_content_to_df(response))
        if not direct_df.empty and {"driverCode", "round", "points", "position"}.issubset(direct_df.columns):
            return direct_df

        if isinstance(content, pd.DataFrame) and {"Results", "results"}.intersection(set(content.columns)):
            content_items = content.to_dict(orient="records")

        for item in content_items:
            if isinstance(item, pd.DataFrame):
                rows.extend(F1DataLoader._normalize_nested_df(item).to_dict(orient="records"))
                continue

            if not isinstance(item, dict):
                continue

            nested_results = item.get("Results") or item.get("results") or []
            base_fields = {
                key: value
                for key, value in item.items()
                if not isinstance(value, (list, dict, pd.DataFrame))
            }

            if isinstance(nested_results, list) and nested_results:
                for result in nested_results:
                    if not isinstance(result, dict):
                        continue

                    flat_row = base_fields.copy()
                    driver = result.get("Driver") if isinstance(result.get("Driver"), dict) else {}
                    flat_row.update(
                        {
                            "driverCode": result.get("driverCode") or driver.get("code"),
                            "position": result.get("position") or result.get("positionText"),
                            "points": result.get("points", 0),
                            "status": result.get("status") or result.get("Status"),
                        }
                    )
                    rows.append(flat_row)
            else:
                rows.append(base_fields)

        return pd.DataFrame(rows)

    @staticmethod
    def _resolve_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
        lower_to_actual = {str(column).lower(): str(column) for column in df.columns}
        for candidate in candidates:
            match = lower_to_actual.get(candidate.lower())
            if match:
                return match
        return None

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
        content = self._flatten_race_result_content(race_results)
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
        content = self._flatten_race_result_content(race_results)
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

    def season_driver_comparison(self, year: int, driver_codes: list[str]) -> dict[str, Any]:
        race_results = self.ergast.get_race_results(season=year, limit=1000)
        content = self._flatten_race_result_content(race_results)
        if content.empty:
            return {"rounds": [], "drivers": {}, "interpretation": "No season comparison data available."}

        driver_col = self._resolve_column(content, ["driverCode", "driver.code", "Driver.code", "Driver.driverCode"])
        round_col = self._resolve_column(content, ["round", "RoundNumber"])
        points_col = self._resolve_column(content, ["points", "Points"])
        position_col = self._resolve_column(content, ["position", "positionText", "Position"])

        if not all([driver_col, round_col, points_col, position_col]):
            return {"rounds": [], "drivers": {}, "interpretation": "Season comparison data is incomplete for this season."}

        normalized = content.rename(
            columns={
                driver_col: "driverCode",
                round_col: "round",
                points_col: "points",
                position_col: "position",
            }
        ).copy()

        normalized = normalized[normalized["driverCode"].notna() & normalized["round"].notna()].copy()
        if normalized.empty:
            return {"rounds": [], "drivers": {}, "interpretation": "No season comparison data available."}

        rounds = sorted(normalized["round"].dropna().astype(int).unique().tolist())
        schedule = {race["round"]: race["race_name"] for race in self.races_for_year(year)}

        drivers: dict[str, Any] = {}
        for driver_code in driver_codes:
            driver_rows = normalized[normalized["driverCode"].astype(str) == str(driver_code)].copy()
            if driver_rows.empty:
                drivers[driver_code] = {
                    "race_names": [],
                    "rounds": [],
                    "points_by_round": [],
                    "cumulative_points": [],
                    "finishing_positions": [],
                    "total_points": 0.0,
                    "average_finish": None,
                }
                continue

            driver_rows["round"] = pd.to_numeric(driver_rows["round"], errors="coerce")
            driver_rows["points"] = pd.to_numeric(driver_rows["points"], errors="coerce").fillna(0.0)
            positions = pd.to_numeric(driver_rows["position"], errors="coerce")
            driver_rows["position_numeric"] = positions
            driver_rows = driver_rows.sort_values("round")

            drivers[driver_code] = {
                "race_names": [schedule.get(int(r), f"Round {int(r)}") for r in driver_rows["round"].tolist()],
                "rounds": [int(r) for r in driver_rows["round"].tolist()],
                "points_by_round": driver_rows["points"].astype(float).round(2).tolist(),
                "cumulative_points": driver_rows["points"].cumsum().astype(float).round(2).tolist(),
                "finishing_positions": driver_rows["position_numeric"].astype(float).tolist(),
                "total_points": round(float(driver_rows["points"].sum()), 2),
                "average_finish": round(float(driver_rows["position_numeric"].mean()), 2)
                if driver_rows["position_numeric"].notna().any()
                else None,
            }

        interpretation = "Season comparison uses cumulative points and finishing positions across all rounds."
        return {"rounds": rounds, "drivers": drivers, "interpretation": interpretation}
