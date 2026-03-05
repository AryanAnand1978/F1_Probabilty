from __future__ import annotations

from typing import Any

from f1_data_loader import F1DataLoader
from statistics_module import (
    correlation_regression,
    covariance_between_drivers,
    descriptive_statistics,
    hypothesis_testing,
    probability_distributions,
    random_variable_analysis,
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

        pit_counts = list(self.data_loader.pit_stops_per_driver(session).values())
        team1_pit_times = self.data_loader.team_pit_stop_times(session, team1)
        team2_pit_times = self.data_loader.team_pit_stop_times(session, team2)

        dnf_records = self.data_loader.dnf_data(session)
        podium_stats = self.data_loader.season_podium_stats(year, driver)
        failure_intervals = self.data_loader.mechanical_failure_intervals(year, driver)

        anova_groups = self._anova_groups(session, limit=6)

        covariance = covariance_between_drivers(selected_driver_laps, comparison_laps)

        descriptive = descriptive_statistics(selected_driver_laps)
        random_variable = random_variable_analysis(
            selected_driver_laps,
            covariance=covariance,
            comparison_driver=comparison_driver,
        )
        corr_reg = correlation_regression(lap_numbers, selected_driver_laps)
        distributions = probability_distributions(
            lap_times=selected_driver_laps,
            pit_stop_counts=pit_counts,
            podium_count=podium_stats["podiums"],
            total_races=podium_stats["races"],
            failure_intervals=failure_intervals,
        )
        hypothesis = hypothesis_testing(
            selected_driver_laps=selected_driver_laps,
            comparison_driver_laps=comparison_laps,
            team1_pit_times=team1_pit_times,
            team2_pit_times=team2_pit_times,
            anova_groups=anova_groups,
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
            "probability_distributions": distributions,
            "hypothesis_testing": hypothesis,
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

    def _anova_groups(self, session, limit: int = 6) -> dict[str, list[float]]:
        laps = session.laps.copy()
        groups: dict[str, list[float]] = {}
        for driver_code in laps["Driver"].dropna().unique().tolist()[:limit]:
            driver_df = laps[(laps["Driver"] == driver_code) & laps["LapTime"].notna()].copy()
            lap_values = driver_df["LapTime"].dt.total_seconds().tolist()
            if len(lap_values) >= 3:
                groups[str(driver_code)] = lap_values
        return groups
