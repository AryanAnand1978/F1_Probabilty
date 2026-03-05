from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats
from scipy.stats import f_oneway, gaussian_kde, linregress, norm, ttest_ind


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


def hypothesis_testing(
    selected_driver_laps: list[float],
    comparison_driver_laps: list[float],
    team1_pit_times: list[float],
    team2_pit_times: list[float],
    anova_groups: dict[str, list[float]],
) -> dict[str, Any]:
    alpha = 0.05
    result: dict[str, Any] = {}

    d1 = np.array(selected_driver_laps, dtype=float)
    d2 = np.array(comparison_driver_laps, dtype=float)
    d1 = d1[np.isfinite(d1)]
    d2 = d2[np.isfinite(d2)]

    if d1.size > 2 and d2.size > 2:
        pop_std = float(np.std(d2, ddof=0)) or 1e-12
        z_stat = (float(np.mean(d1)) - float(np.mean(d2))) / (pop_std / np.sqrt(d1.size))
        z_p = float(2 * (1 - norm.cdf(abs(z_stat))))
        result["z_test"] = {
            "statistic": round(z_stat, 5),
            "p_value": round(z_p, 6),
            "decision": "Reject H0" if z_p < alpha else "Fail to Reject H0",
            "hypothesis": "H0: mean(selected_driver) = mean(comparison_driver)",
        }
    else:
        result["z_test"] = {"error": "Insufficient data for Z-test."}

    t1 = np.array(team1_pit_times, dtype=float)
    t2 = np.array(team2_pit_times, dtype=float)
    t1 = t1[np.isfinite(t1)]
    t2 = t2[np.isfinite(t2)]
    if t1.size > 1 and t2.size > 1:
        t_stat, t_p = ttest_ind(t1, t2, equal_var=False)
        result["t_test"] = {
            "statistic": round(float(t_stat), 5),
            "p_value": round(float(t_p), 6),
            "decision": "Reject H0" if t_p < alpha else "Fail to Reject H0",
            "hypothesis": "H0: mean pit-stop time team1 = team2",
        }
    else:
        result["t_test"] = {"error": "Insufficient pit stop data for t-test."}

    if d1.size >= 12:
        observed, bins = np.histogram(d1, bins=min(10, max(4, d1.size // 5)))
        mu = float(np.mean(d1))
        sigma = float(np.std(d1, ddof=0)) or 1e-12

        cdf_vals = norm.cdf(bins, mu, sigma)
        probs = np.diff(cdf_vals)
        probs = np.maximum(probs, 1e-6)
        expected = probs / probs.sum() * observed.sum()
        chi_stat = float(np.sum((observed - expected) ** 2 / expected))
        df = max(1, len(observed) - 1 - 2)
        chi_p = float(1 - stats.chi2.cdf(chi_stat, df=df))

        result["chi_square"] = {
            "statistic": round(chi_stat, 5),
            "p_value": round(chi_p, 6),
            "decision": "Reject H0" if chi_p < alpha else "Fail to Reject H0",
            "hypothesis": "H0: lap times follow normal distribution",
        }
    else:
        result["chi_square"] = {"error": "Need at least 12 laps for chi-square normality test."}

    cleaned_groups: dict[str, np.ndarray] = {}
    for name, values in anova_groups.items():
        arr = np.array(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size >= 2:
            cleaned_groups[name] = arr

    if len(cleaned_groups) >= 2:
        f_stat, f_p = f_oneway(*cleaned_groups.values())
        result["anova"] = {
            "statistic": round(float(f_stat), 5),
            "p_value": round(float(f_p), 6),
            "decision": "Reject H0" if f_p < alpha else "Fail to Reject H0",
            "hypothesis": "H0: all compared drivers have equal mean lap time",
            "group_means": {k: round(float(np.mean(v)), 4) for k, v in cleaned_groups.items()},
        }
    else:
        result["anova"] = {"error": "Insufficient groups for ANOVA."}

    result["decision_rule"] = "Reject H0 if p-value < 0.05"
    return result


def covariance_between_drivers(driver_a_laps: list[float], driver_b_laps: list[float]) -> float | None:
    a = np.array(driver_a_laps, dtype=float)
    b = np.array(driver_b_laps, dtype=float)
    n = min(a.size, b.size)
    if n < 2:
        return None
    return float(np.cov(a[:n], b[:n], ddof=0)[0, 1])
