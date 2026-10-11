"""
Weather Pattern Analyzer
IEEE LGU AI/ML Cohort One - Week 3: Data Analysis with NumPy & Pandas
AI/ML Leads: Abdullah Faisal & Alina Irshad

Loads daily weather readings from data/daily_weather.csv, cleans sensor errors,
runs NumPy calculations, groups the data by month, merges it with
data/monthly_targets.csv and prints a comparison report.
Charts are saved as PNG files in the charts/ folder.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no GUI needed, we only save PNG files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
WEATHER_FILE = BASE_DIR / "data" / "daily_weather.csv"
TARGETS_FILE = BASE_DIR / "data" / "monthly_targets.csv"
CHART_DIR = BASE_DIR / "charts"

# Thresholds used for "extreme" days and for judging a reading as a sensor fault
HOT_DAY_C = 35.0
HEAVY_RAIN_MM = 10.0
VALID_TEMP_C = (-10.0, 55.0)
VALID_HUMIDITY = (0.0, 100.0)

LINE = "=" * 62


def banner(title: str) -> None:
    print(f"\n{LINE}\n{title}\n{LINE}")


# --------------------------------------------------------------------------
# 1. LOAD & CLEAN
# --------------------------------------------------------------------------
def load_data(path: Path) -> pd.DataFrame:
    """Read the raw CSV as text so bad entries like 'abc' survive loading."""
    return pd.read_csv(path, dtype=str)


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean daily weather readings and return (clean_df, log of changes)."""
    log = {"rows_loaded": len(raw)}
    df = raw.copy()

    # Parse dates; impossible dates (e.g. 2026-13-45) become NaT and are dropped
    df["date"] = pd.to_datetime(df["date"].str.strip(), format="%Y-%m-%d", errors="coerce")
    before = len(df)
    df = df.dropna(subset=["date"])
    log["bad_dates_dropped"] = before - len(df)

    # One reading per day: keep the first row for any repeated date
    before = len(df)
    df = df.drop_duplicates(subset="date", keep="first")
    log["duplicates_removed"] = before - len(df)

    # Convert to numbers; text such as "abc" or "N/A" becomes NaN
    for col in ["temperature_c", "humidity_pct", "rainfall_mm"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Physically impossible values are sensor faults -> treat as missing
    bad_temp = ~df["temperature_c"].between(*VALID_TEMP_C) & df["temperature_c"].notna()
    bad_hum = ~df["humidity_pct"].between(*VALID_HUMIDITY) & df["humidity_pct"].notna()
    bad_rain = (df["rainfall_mm"] < 0) & df["rainfall_mm"].notna()
    log["impossible_values_blanked"] = int(bad_temp.sum() + bad_hum.sum() + bad_rain.sum())
    df.loc[bad_temp, "temperature_c"] = np.nan
    df.loc[bad_hum, "humidity_pct"] = np.nan
    df.loc[bad_rain, "rainfall_mm"] = np.nan

    # Sort by date so interpolation uses the neighbouring days
    df = df.sort_values("date").reset_index(drop=True)

    # Temperature and humidity change smoothly day to day -> linear interpolation
    log["temp_filled"] = int(df["temperature_c"].isna().sum())
    log["humidity_filled"] = int(df["humidity_pct"].isna().sum())
    df["temperature_c"] = df["temperature_c"].interpolate(limit_direction="both")
    df["humidity_pct"] = df["humidity_pct"].interpolate(limit_direction="both")

    # Rainfall is sporadic, so interpolating would invent rain.
    # A missing rainfall reading is assumed to mean no rain was recorded.
    log["rainfall_filled"] = int(df["rainfall_mm"].isna().sum())
    df["rainfall_mm"] = df["rainfall_mm"].fillna(0.0)

    df["month"] = df["date"].dt.to_period("M").astype(str)
    log["rows_clean"] = len(df)
    return df, log


def print_cleaning_report(log: dict) -> None:
    banner("DATA CLEANING REPORT")
    print(f"Rows loaded from CSV              : {log['rows_loaded']}")
    print(f"Invalid dates dropped             : {log['bad_dates_dropped']}")
    print(f"Duplicate dates removed           : {log['duplicates_removed']}")
    print(f"Impossible values set to missing  : {log['impossible_values_blanked']}")
    print(f"Temperature gaps interpolated     : {log['temp_filled']}")
    print(f"Humidity gaps interpolated        : {log['humidity_filled']}")
    print(f"Rainfall gaps filled with 0.0 mm  : {log['rainfall_filled']}")
    print(f"Rows remaining after cleaning     : {log['rows_clean']}")


# --------------------------------------------------------------------------
# 2. NUMPY CALCULATIONS
# --------------------------------------------------------------------------
def overall_stats(df: pd.DataFrame) -> dict:
    """Whole-period statistics computed on NumPy arrays."""
    temp = df["temperature_c"].to_numpy()
    hum = df["humidity_pct"].to_numpy()
    rain = df["rainfall_mm"].to_numpy()
    daily_change = np.diff(temp)  # today's temperature minus yesterday's
    return {
        "avg_temp": np.mean(temp),
        "avg_humidity": np.mean(hum),
        "total_rain": np.sum(rain),
        "temp_range": np.max(temp) - np.min(temp),
        "hottest": np.max(temp),
        "coolest": np.min(temp),
        "temp_std": np.std(temp),
        "biggest_swing": np.max(np.abs(daily_change)),
        "avg_swing": np.mean(np.abs(daily_change)),
    }


def flag_extreme_days(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean columns marking hot days and heavy-rain days."""
    df = df.copy()
    temp = df["temperature_c"].to_numpy()
    rain = df["rainfall_mm"].to_numpy()
    df["hot_day"] = temp > HOT_DAY_C
    df["heavy_rain_day"] = rain > HEAVY_RAIN_MM
    df["extreme_day"] = df["hot_day"] | df["heavy_rain_day"]
    return df


# --------------------------------------------------------------------------
# 3. GROUPING & MERGING
# --------------------------------------------------------------------------
def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per month: averages, totals, temperature range, extreme-day counts."""
    summary = df.groupby("month").agg(
        avg_temp=("temperature_c", "mean"),
        max_temp=("temperature_c", "max"),
        min_temp=("temperature_c", "min"),
        avg_humidity=("humidity_pct", "mean"),
        total_rain=("rainfall_mm", "sum"),
        hot_days=("hot_day", "sum"),
        heavy_rain_days=("heavy_rain_day", "sum"),
        extreme_days=("extreme_day", "sum"),
    )
    summary["temp_range"] = summary["max_temp"] - summary["min_temp"]
    return summary.reset_index()


def merge_targets(summary: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Join the monthly summary with the historical baseline on the month key."""
    merged = summary.merge(targets, on="month", how="left")
    merged["temp_diff"] = merged["avg_temp"] - merged["historical_avg_temp"]
    merged["rain_pct_of_target"] = merged["total_rain"] / merged["target_rainfall_mm"] * 100
    return merged


# --------------------------------------------------------------------------
# 4. CONSOLE REPORT
# --------------------------------------------------------------------------
def print_summary(df: pd.DataFrame, stats: dict, merged: pd.DataFrame) -> None:
    first, last = df["date"].min().date(), df["date"].max().date()
    banner(f"WEATHER SUMMARY REPORT  ({first} to {last})")
    print(f"Days analysed            : {len(df)}")
    print(f"Average temperature      : {stats['avg_temp']:.1f} C")
    print(f"Average humidity         : {stats['avg_humidity']:.1f} %")
    print(f"Total rainfall           : {stats['total_rain']:.1f} mm")
    print(f"Hottest / coolest day    : {stats['hottest']:.1f} C / {stats['coolest']:.1f} C")
    print(f"Overall temperature range: {stats['temp_range']:.1f} C")
    print(f"Temperature std. dev.    : {stats['temp_std']:.2f} C")
    print(f"Avg day-to-day swing     : {stats['avg_swing']:.2f} C")
    print(f"Biggest day-to-day swing : {stats['biggest_swing']:.1f} C")

    banner(f"EXTREME DAYS (temp > {HOT_DAY_C:.0f} C or rain > {HEAVY_RAIN_MM:.0f} mm)")
    print(f"Hot days        : {int(df['hot_day'].sum())}")
    print(f"Heavy-rain days : {int(df['heavy_rain_day'].sum())}")
    print(f"Total extreme   : {int(df['extreme_day'].sum())} of {len(df)} days")
    print("\nFive hottest days:")
    for _, row in df.nlargest(5, "temperature_c").iterrows():
        print(f"  {row['date'].date()}  {row['temperature_c']:.1f} C")
    print("\nFive wettest days:")
    for _, row in df.nlargest(5, "rainfall_mm").iterrows():
        print(f"  {row['date'].date()}  {row['rainfall_mm']:.1f} mm")

    banner("MONTHLY AVERAGES")
    for _, r in merged.iterrows():
        print(f"{r['month']}:  avg temp {r['avg_temp']:.1f} C | "
              f"range {r['temp_range']:.1f} C | humidity {r['avg_humidity']:.1f} % | "
              f"rain {r['total_rain']:.1f} mm | extreme days {int(r['extreme_days'])}")

    banner("COMPARISON WITH HISTORICAL BASELINE")
    for _, r in merged.iterrows():
        direction = "warmer" if r["temp_diff"] > 0 else "cooler"
        print(f"{r['month']}:")
        print(f"  Temperature : {r['avg_temp']:.1f} C vs {r['historical_avg_temp']:.1f} C "
              f"historical  ({abs(r['temp_diff']):.1f} C {direction})")
        print(f"  Rainfall    : {r['total_rain']:.1f} mm vs {r['target_rainfall_mm']:.1f} mm "
              f"target  ({r['rain_pct_of_target']:.0f}% of target)")


# --------------------------------------------------------------------------
# 5. CHARTS (PNG)
# --------------------------------------------------------------------------
def save_charts(df: pd.DataFrame, merged: pd.DataFrame) -> None:
    CHART_DIR.mkdir(exist_ok=True)
    blue, red, grey = "#2a6fb0", "#c0392b", "#7f8c8d"

    # Daily temperature with the hot-day threshold and hot days highlighted
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(df["date"], df["temperature_c"], color=blue, label="Daily temperature")
    ax.axhline(HOT_DAY_C, color=grey, linestyle="--", label=f"{HOT_DAY_C:.0f} C threshold")
    hot = df[df["hot_day"]]
    ax.scatter(hot["date"], hot["temperature_c"], color=red, zorder=3, label="Hot day")
    ax.set_title("Daily Temperature")
    ax.set_ylabel("Temperature (C)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(CHART_DIR / "daily_temperature.png", dpi=150)
    plt.close(fig)

    # Daily rainfall with the heavy-rain threshold
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = np.where(df["heavy_rain_day"], red, blue)
    ax.bar(df["date"], df["rainfall_mm"], color=colors)
    ax.axhline(HEAVY_RAIN_MM, color=grey, linestyle="--")
    ax.set_title(f"Daily Rainfall (red = over {HEAVY_RAIN_MM:.0f} mm)")
    ax.set_ylabel("Rainfall (mm)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(CHART_DIR / "daily_rainfall.png", dpi=150)
    plt.close(fig)

    # Actual vs historical, side by side per month
    x = np.arange(len(merged))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(x - w / 2, merged["avg_temp"], w, color=blue, label="Recorded")
    axes[0].bar(x + w / 2, merged["historical_avg_temp"], w, color=grey, label="Historical")
    axes[0].set_xticks(x, merged["month"])
    axes[0].set_title("Average Temperature (C)")
    axes[0].legend()
    axes[1].bar(x - w / 2, merged["total_rain"], w, color=blue, label="Recorded")
    axes[1].bar(x + w / 2, merged["target_rainfall_mm"], w, color=grey, label="Target")
    axes[1].set_xticks(x, merged["month"])
    axes[1].set_title("Total Rainfall (mm)")
    axes[1].legend()
    # Leave headroom so the legends never cover the bars
    axes[0].set_ylim(0, merged[["avg_temp", "historical_avg_temp"]].to_numpy().max() * 1.3)
    axes[1].set_ylim(0, merged[["total_rain", "target_rainfall_mm"]].to_numpy().max() * 1.3)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "monthly_comparison.png", dpi=150)
    plt.close(fig)

    # Humidity vs temperature
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(df["temperature_c"], df["humidity_pct"], color=blue, alpha=0.7)
    ax.set_title("Humidity vs Temperature")
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("Humidity (%)")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "humidity_vs_temperature.png", dpi=150)
    plt.close(fig)

    print(f"\nCharts saved to: {CHART_DIR.name}/ (4 PNG files)")


def main() -> None:
    raw = load_data(WEATHER_FILE)
    df, log = clean_data(raw)
    print_cleaning_report(log)

    df = flag_extreme_days(df)
    stats = overall_stats(df)

    targets = pd.read_csv(TARGETS_FILE)
    merged = merge_targets(monthly_summary(df), targets)

    print_summary(df, stats, merged)
    save_charts(df, merged)


if __name__ == "__main__":
    main()
