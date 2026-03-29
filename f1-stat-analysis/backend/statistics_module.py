from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde, linregress, norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures


def descriptive_statistics(lap_times: list[float]) -> dict[str, Any]:
    arr = np.array(lap_times, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        raise ValueError("Not enough lap time points for descriptive statistics.")

    mean = float(np.mean(arr))
    variance = float(np.var(arr, ddof=0))
    std_dev = float(np.std(arr, ddof=0))
    skewness = float(stats.skew(arr, bias=False))
    kurtosis = float(stats.kurtosis(arr, fisher=True, bias=False))

    hist_counts, hist_bins = np.histogram(arr, bins="auto")

    return {
        "summary": {
            "count": int(arr.size),
            "mean": round(mean, 4),
            "variance": round(variance, 4),
            "std_dev": round(std_dev, 4),
            "skewness": round(skewness, 4),
            "kurtosis": round(kurtosis, 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
        },
        "histogram": {
            "counts": hist_counts.tolist(),
            "bins": hist_bins.tolist(),
            "values": arr.tolist(),
        },
        "boxplot": {
            "values": arr.tolist(),
        },
        "formulas": {
            "mean": "mu = sum(x_i) / n",
            "variance": "sigma^2 = sum((x_i - mu)^2) / n",
            "std_dev": "sigma = sqrt(sigma^2)",
        },
        "interpretation": (
            "The lap time distribution is approximately normal with low variance, indicating consistent performance."
            if abs(skewness) < 0.7 and std_dev < 2.0
            else "The distribution shows wider spread or asymmetry, suggesting variation across stints or race events."
        ),
    }


def random_variable_analysis(
    lap_times: list[float], covariance: float | None = None, comparison_driver: str | None = None
) -> dict[str, Any]:
    arr = np.array(lap_times, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        raise ValueError("Not enough lap data for random variable analysis.")

    kde = gaussian_kde(arr)
    x = np.linspace(np.min(arr) - 1.5, np.max(arr) + 1.5, 250)
    y = kde(x)

    return {
        "definition": "X = Lap time random variable (seconds)",
        "expectation": round(float(np.mean(arr)), 4),
        "variance": round(float(np.var(arr)), 4),
        "pdf": {"x": x.tolist(), "y": y.tolist()},
        "covariance_with_comparison_driver": None if covariance is None else round(float(covariance), 4),
        "comparison_driver": comparison_driver,
    }


def correlation_regression(lap_numbers: list[int], lap_times: list[float]) -> dict[str, Any]:
    x = np.array(lap_numbers, dtype=float)
    y = np.array(lap_times, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < 3:
        raise ValueError("Not enough points for correlation/regression.")

    corr, corr_p = stats.pearsonr(x, y)
    reg = linregress(x, y)
    x_line = np.linspace(np.min(x), np.max(x), 200)
    y_line = reg.intercept + reg.slope * x_line

    return {
        "correlation_coefficient": round(float(corr), 5),
        "correlation_p_value": round(float(corr_p), 6),
        "equation": f"LapTime = {reg.intercept:.4f} + {reg.slope:.5f} * LapNumber",
        "slope": round(float(reg.slope), 6),
        "intercept": round(float(reg.intercept), 6),
        "scatter": {"x": x.tolist(), "y": y.tolist()},
        "regression_line": {"x": x_line.tolist(), "y": y_line.tolist()},
        "interpretation": (
            "Positive slope indicates tire degradation impact across laps."
            if reg.slope > 0
            else "Negative slope indicates pace improvement through the stint."
        ),
        "formulas": {
            "correlation": "r = Cov(X,Y) / (sigma_x sigma_y)",
            "regression": "LapTime = a + b * LapNumber",
        },
    }


def lap_trend_analysis(
    lap_numbers: list[int],
    lap_times: list[float],
    outlier_threshold: float = 120.0,
    smoothing_window: int = 5,
    polynomial_degree: int = 3,
) -> dict[str, Any]:
    x = np.array(lap_numbers, dtype=float)
    y = np.array(lap_times, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < 3:
        raise ValueError("Not enough points for trend analysis.")

    original = {"x": x.tolist(), "y": y.tolist()}

    clean_mask = y <= outlier_threshold
    x_clean = x[clean_mask]
    y_clean = y[clean_mask]
    removed = int(x.size - x_clean.size)

    if x_clean.size < 3:
        raise ValueError("Not enough cleaned lap data after outlier removal.")

    if x_clean.size >= smoothing_window:
        kernel = np.ones(smoothing_window, dtype=float) / smoothing_window
        y_smooth = np.convolve(y_clean, kernel, mode="valid")
        x_smooth = x_clean[smoothing_window - 1 :]
    else:
        y_smooth = y_clean.copy()
        x_smooth = x_clean.copy()

    if x_smooth.size < 3:
        raise ValueError("Not enough smoothed data points for regression.")

    x_model = x_smooth.reshape(-1, 1)

    linear_model = LinearRegression()
    linear_model.fit(x_model, y_smooth)
    linear_pred = linear_model.predict(x_model)

    x_curve = np.linspace(float(np.min(x_clean)), float(np.max(x_clean)), 200).reshape(-1, 1)
    linear_curve = linear_model.predict(x_curve)
    piecewise_fit = _piecewise_polynomial_fit(x_smooth, y_smooth, polynomial_degree)

    corr, corr_p = stats.pearsonr(x_smooth, y_smooth)

    return {
        "parameters": {
            "outlier_threshold": outlier_threshold,
            "smoothing_window": smoothing_window,
            "polynomial_degree": polynomial_degree,
            "removed_outliers": removed,
            "stint_segments": len(piecewise_fit["segments"]),
        },
        "original_scatter": original,
        "cleaned_scatter": {"x": x_clean.tolist(), "y": y_clean.tolist()},
        "smoothed_line": {"x": x_smooth.tolist(), "y": y_smooth.tolist()},
        "linear_fit": {
            "x": x_curve.ravel().tolist(),
            "y": linear_curve.tolist(),
            "equation": f"LapTime = {linear_model.intercept_:.4f} + {linear_model.coef_[0]:.5f} * LapNumber",
            "r_squared": round(float(r2_score(y_smooth, linear_pred)), 5),
        },
        "polynomial_fit": {
            "x": piecewise_fit["curve"]["x"],
            "y": piecewise_fit["curve"]["y"],
            "equation": "Piecewise polynomial fit by race stint",
            "r_squared": piecewise_fit["r_squared"],
            "degree": piecewise_fit["max_degree"],
            "segment_equations": piecewise_fit["segment_equations"],
            "segments": piecewise_fit["segments"],
            "model_label": "Piecewise polynomial (stint-aware)",
        },
        "correlation": {
            "pearson_r": round(float(corr), 5),
            "p_value": round(float(corr_p), 6),
        },
        "interpretation": (
            "Outlier laps were removed, a moving average was applied, and a piecewise polynomial model was fitted separately to each stint so the pace trend follows realistic race phases better than a single global curve."
        ),
    }


def driver_comparison_summary(
    driver_a: str,
    driver_b: str,
    driver_a_laps: list[float],
    driver_b_laps: list[float],
) -> dict[str, Any]:
    a = np.array(driver_a_laps, dtype=float)
    b = np.array(driver_b_laps, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if a.size < 3 or b.size < 3:
        raise ValueError("Not enough cleaned lap data for driver comparison.")

    a_mean = float(np.mean(a))
    b_mean = float(np.mean(b))
    a_std = float(np.std(a, ddof=0))
    b_std = float(np.std(b, ddof=0))

    return {
        "driver_a": driver_a,
        "driver_b": driver_b,
        "mean_delta_seconds": round(a_mean - b_mean, 4),
        "summary": {
            driver_a: {"mean": round(a_mean, 4), "std_dev": round(a_std, 4), "best_lap": round(float(np.min(a)), 4)},
            driver_b: {"mean": round(b_mean, 4), "std_dev": round(b_std, 4), "best_lap": round(float(np.min(b)), 4)},
        },
        "interpretation": (
            f"{driver_a} had the lower average race-pace lap time."
            if a_mean < b_mean
            else f"{driver_b} had the lower average race-pace lap time."
        ),
    }


def team_comparison_summary(
    team1: str,
    team2: str,
    team1_laps: list[float],
    team2_laps: list[float],
    team1_pit_times: list[float],
    team2_pit_times: list[float],
) -> dict[str, Any]:
    t1 = np.array(team1_laps, dtype=float)
    t2 = np.array(team2_laps, dtype=float)
    t1 = t1[np.isfinite(t1)]
    t2 = t2[np.isfinite(t2)]

    if t1.size < 3 or t2.size < 3:
        raise ValueError("Not enough team lap data for comparison.")

    team1_mean = float(np.mean(t1))
    team2_mean = float(np.mean(t2))
    team1_pit_mean = float(np.mean(team1_pit_times)) if team1_pit_times else None
    team2_pit_mean = float(np.mean(team2_pit_times)) if team2_pit_times else None

    return {
        "lap_time_boxplot": {
            "team1": {"name": team1, "values": t1.tolist()},
            "team2": {"name": team2, "values": t2.tolist()},
        },
        "lap_time_histogram": {
            "team1": {"name": team1, "values": t1.tolist()},
            "team2": {"name": team2, "values": t2.tolist()},
        },
        "summary": {
            team1: {
                "mean_lap": round(team1_mean, 4),
                "std_dev": round(float(np.std(t1, ddof=0)), 4),
                "avg_pit_time": None if team1_pit_mean is None else round(team1_pit_mean, 4),
            },
            team2: {
                "mean_lap": round(team2_mean, 4),
                "std_dev": round(float(np.std(t2, ddof=0)), 4),
                "avg_pit_time": None if team2_pit_mean is None else round(team2_pit_mean, 4),
            },
        },
        "interpretation": (
            f"{team1} had the lower average lap time in this race."
            if team1_mean < team2_mean
            else f"{team2} had the lower average lap time in this race."
        ),
    }


def probability_distributions(
    lap_times: list[float],
    pit_stop_counts: list[int],
    podium_count: int,
    total_races: int,
    failure_intervals: list[int],
) -> dict[str, Any]:
    arr = np.array(lap_times, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        raise ValueError("Not enough lap times for distribution fitting.")

    mu, sigma = norm.fit(arr)
    nx = np.linspace(np.min(arr) - 1.5, np.max(arr) + 1.5, 250)
    npdf = norm.pdf(nx, mu, sigma)

    n = max(total_races, 1)
    p = float(podium_count) / n
    bx = np.arange(0, n + 1)
    bpmf = stats.binom.pmf(bx, n, p)

    lam = float(np.mean(pit_stop_counts)) if pit_stop_counts else 1.0
    px = np.arange(0, 12)
    ppmf = stats.poisson.pmf(px, lam)

    if failure_intervals:
        exp_scale = float(np.mean(failure_intervals))
    else:
        lap_deltas = np.diff(np.sort(arr))
        lap_deltas = lap_deltas[lap_deltas > 0]
        exp_scale = float(np.mean(lap_deltas)) if lap_deltas.size > 0 else 1.0
    ex = np.linspace(0, max(10.0, exp_scale * 6), 250)
    epdf = stats.expon.pdf(ex, scale=exp_scale)

    return {
        "normal": {
            "parameters": {"mu": round(float(mu), 4), "sigma": round(float(sigma), 4)},
            "curve": {"x": nx.tolist(), "y": npdf.tolist()},
            "formula": "f(x) = 1/(sigma*sqrt(2*pi)) * exp(-(x-mu)^2/(2*sigma^2))",
        },
        "binomial": {
            "parameters": {"n": int(n), "p": round(float(p), 4)},
            "curve": {"x": bx.tolist(), "y": bpmf.tolist()},
            "formula": "P(X=k) = C(n,k) p^k (1-p)^(n-k)",
        },
        "poisson": {
            "parameters": {"lambda": round(float(lam), 4)},
            "curve": {"x": px.tolist(), "y": ppmf.tolist()},
            "formula": "P(X=k) = (lambda^k * e^-lambda) / k!",
        },
        "exponential": {
            "parameters": {"lambda": round(float(1 / exp_scale), 4), "scale": round(float(exp_scale), 4)},
            "curve": {"x": ex.tolist(), "y": epdf.tolist()},
            "formula": "f(x) = lambda * e^(-lambda x)",
        },
    }


def covariance_between_drivers(driver_a_laps: list[float], driver_b_laps: list[float]) -> float | None:
    a = np.array(driver_a_laps, dtype=float)
    b = np.array(driver_b_laps, dtype=float)
    n = min(a.size, b.size)
    if n < 2:
        return None
    return float(np.cov(a[:n], b[:n], ddof=0)[0, 1])


def _polynomial_equation(model: LinearRegression, degree: int) -> str:
    terms = [f"{model.intercept_:.4f}"]
    for power, coefficient in enumerate(model.coef_, start=1):
        if power > degree:
            break
        terms.append(f"{coefficient:.5f} * LapNumber^{power}")
    return "LapTime = " + " + ".join(terms)


def _piecewise_polynomial_fit(x_values: np.ndarray, y_values: np.ndarray, polynomial_degree: int) -> dict[str, Any]:
    split_points = np.where(np.diff(x_values) > 1.5)[0] + 1
    x_segments = np.split(x_values, split_points)
    y_segments = np.split(y_values, split_points)

    curve_x: list[float | None] = []
    curve_y: list[float | None] = []
    segment_equations: list[str] = []
    segments: list[dict[str, Any]] = []
    weighted_r2_total = 0.0
    total_points = 0
    max_degree = 1

    for x_segment, y_segment in zip(x_segments, y_segments):
        if x_segment.size < 2:
            continue

        degree = min(polynomial_degree, max(1, x_segment.size - 1))
        max_degree = max(max_degree, degree)

        x_model = x_segment.reshape(-1, 1)
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        x_poly = poly_features.fit_transform(x_model)
        poly_model = LinearRegression()
        poly_model.fit(x_poly, y_segment)
        y_pred = poly_model.predict(x_poly)
        segment_r2 = float(r2_score(y_segment, y_pred)) if x_segment.size > 2 else 1.0

        x_curve = np.linspace(float(np.min(x_segment)), float(np.max(x_segment)), 60).reshape(-1, 1)
        y_curve = poly_model.predict(poly_features.transform(x_curve))

        curve_x.extend(x_curve.ravel().tolist())
        curve_y.extend(y_curve.tolist())
        curve_x.append(None)
        curve_y.append(None)

        equation = _polynomial_equation(poly_model, degree)
        segment_equations.append(f"Laps {int(x_segment[0])}-{int(x_segment[-1])}: {equation}")
        segments.append(
            {
                "start_lap": int(x_segment[0]),
                "end_lap": int(x_segment[-1]),
                "degree": degree,
                "r_squared": round(segment_r2, 5),
                "equation": equation,
            }
        )

        weighted_r2_total += segment_r2 * x_segment.size
        total_points += int(x_segment.size)

    if curve_x and curve_x[-1] is None:
        curve_x.pop()
        curve_y.pop()

    overall_r2 = round(weighted_r2_total / total_points, 5) if total_points else None
    return {
        "curve": {"x": curve_x, "y": curve_y},
        "segment_equations": segment_equations,
        "segments": segments,
        "r_squared": overall_r2,
        "max_degree": max_degree,
    }
