from __future__ import annotations

from typing import Any

from f1_data_loader import F1DataLoader
from statistics_module import (
    correlation_regression,
    covariance_between_drivers,
    descriptive_statistics,
    driver_comparison_summary,
    lap_trend_analysis,
    probability_distributions,
    random_variable_analysis,
    team_comparison_summary,
)


class AnalysisEngine:
    def __init__(self, data_loader: F1DataLoader) -> None:
        self.data_loader = data_loader

    def run(
        self,
        year: int,
        race: str | int,
        driver: str,
        comparison_driver: str | None,
        team1: str,
        team2: str,
    ) -> dict[str, Any]:
        session = self.data_loader.load_race_session(year=year, race=race)
        event_name = str(getattr(session.event, "EventName", race))

        driver_df = self.data_loader.driver_laps(session, driver)
        selected_driver_laps = self.data_loader.lap_times_from_df(driver_df)
        if len(selected_driver_laps) < 3:
            raise ValueError(f"Not enough valid lap-time data for driver {driver} in {event_name}.")

        team1_df = self.data_loader.team_laps(session, team1)
        team2_df = self.data_loader.team_laps(session, team2)

        available_drivers = set(session.laps["Driver"].dropna().astype(str).unique().tolist())
        if comparison_driver is None:
            comparison_driver = self._pick_comparison_driver(session, driver, team1, team2)

        if not comparison_driver or comparison_driver == driver:
            raise ValueError("Please choose a comparison driver different from the selected driver.")
        if comparison_driver not in available_drivers:
            raise ValueError(f"Comparison driver {comparison_driver} not available in this race session.")

        comparison_df = self.data_loader.driver_laps(session, comparison_driver)
        comparison_laps = self.data_loader.lap_times_from_df(comparison_df)
        if len(comparison_laps) < 3:
            raise ValueError(f"Not enough lap data for comparison driver {comparison_driver}.")

        lap_numbers = driver_df["LapNumber"].astype(int).tolist()
        comparison_lap_numbers = comparison_df["LapNumber"].astype(int).tolist()

        pit_counts = list(self.data_loader.pit_stops_per_driver(session).values())
        team1_pit_times = self.data_loader.team_pit_stop_times(session, team1)
        team2_pit_times = self.data_loader.team_pit_stop_times(session, team2)

        dnf_records = self.data_loader.dnf_data(session)
        podium_stats = self.data_loader.season_podium_stats(year, driver)
        failure_intervals = self.data_loader.mechanical_failure_intervals(year, driver)
        season_comparison = self.data_loader.season_driver_comparison(year, [driver, comparison_driver])

        selected_trend = lap_trend_analysis(lap_numbers, selected_driver_laps)
        comparison_trend = lap_trend_analysis(comparison_lap_numbers, comparison_laps)
        selected_clean = selected_trend["cleaned_scatter"]["y"]
        comparison_clean = comparison_trend["cleaned_scatter"]["y"]

        team1_laps = self._clean_laps(self.data_loader.lap_times_from_df(team1_df))
        team2_laps = self._clean_laps(self.data_loader.lap_times_from_df(team2_df))

        covariance = covariance_between_drivers(selected_clean, comparison_clean)

        descriptive = descriptive_statistics(selected_driver_laps)
        random_variable = random_variable_analysis(
            selected_clean,
            covariance=covariance,
            comparison_driver=comparison_driver,
        )
        corr_reg = correlation_regression(
            selected_trend["smoothed_line"]["x"],
            selected_trend["smoothed_line"]["y"],
        )
        distributions = probability_distributions(
            lap_times=selected_clean,
            pit_stop_counts=pit_counts,
            podium_count=podium_stats["podiums"],
            total_races=podium_stats["races"],
            failure_intervals=failure_intervals,
        )
        driver_comparison = driver_comparison_summary(
            driver,
            comparison_driver,
            selected_clean,
            comparison_clean,
        )
        team_comparison = team_comparison_summary(
            team1,
            team2,
            team1_laps,
            team2_laps,
            team1_pit_times,
            team2_pit_times,
        )

        return {
            "meta": {
                "season": year,
                "race": event_name,
                "driver": driver,
                "comparison_driver": comparison_driver,
                "team1": team1,
                "team2": team2,
                "laps_analyzed": len(selected_driver_laps),
            },
            "descriptive_statistics": descriptive,
            "random_variables": random_variable,
            "correlation_regression": corr_reg,
            "driver_trends": {
                "selected_driver": selected_trend,
                "comparison_driver": comparison_trend,
                "comparison_summary": driver_comparison,
            },
            "team_comparison": team_comparison,
            "season_comparison": season_comparison,
            "probability_distributions": distributions,
            "race_context": {
                "dnf_data": dnf_records,
                "team1_pit_stop_times": team1_pit_times,
                "team2_pit_stop_times": team2_pit_times,
                "pit_stops_per_driver": pit_counts,
                "podium_stats": podium_stats,
            },
        }

    def _pick_comparison_driver(self, session, selected_driver: str, team1: str, team2: str) -> str | None:
        laps = session.laps.copy()
        candidates = laps[(laps["Driver"] != selected_driver) & (laps["Team"].isin([team1, team2]))]["Driver"]
        if not candidates.empty:
            return str(candidates.mode().iloc[0])

        other = laps[laps["Driver"] != selected_driver]["Driver"]
        if not other.empty:
            return str(other.mode().iloc[0])

        return None

    @staticmethod
    def _clean_laps(lap_values: list[float], threshold: float = 120.0) -> list[float]:
        return [lap for lap in lap_values if lap <= threshold]
