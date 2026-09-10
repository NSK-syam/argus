"""
NASA C-MAPSS FD001 loader, RUL labeling, and feature engineering.

FD001 column layout (26 columns, whitespace-delimited, no header):
    unit_number, time_cycles, op_setting_1..3, sensor_1..21

Reference: A. Saxena, K. Goebel (2008), "Turbofan Engine Degradation
Simulation Data Set", NASA Ames Prognostics Data Repository.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

RUL_CAP = 125  # standard C-MAPSS RUL clipping used in most published baselines
WARNING_THRESHOLD = 30  # cycles; used for the F1(RUL<=30) classification metric

INDEX_COLS = ["unit_number", "time_cycles"]
OP_SETTING_COLS = [f"op_setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + OP_SETTING_COLS + SENSOR_COLS

ROLLING_WINDOWS = (5, 10, 20)
LAG_STEPS = (1, 5)


def _read_raw(path: Path) -> pd.DataFrame:
    """Read a whitespace-delimited C-MAPSS file with no header."""
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    # Some published copies carry two trailing all-NaN columns from a
    # trailing delimiter in the original NASA export; drop anything past
    # the 26 documented columns rather than assuming a fixed shape.
    df = df.iloc[:, : len(ALL_COLS)]
    df.columns = ALL_COLS
    return df


def load_fd001_train(data_dir: Path) -> pd.DataFrame:
    df = _read_raw(Path(data_dir) / "train_FD001.txt")
    df["RUL"] = _compute_train_rul(df)
    return df


def load_fd001_test(data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Return the test feature frame plus the *true* end-of-trace RUL per
    engine (from RUL_FD001.txt) — the official FD001 held-out evaluation.
    """
    df = _read_raw(Path(data_dir) / "test_FD001.txt")
    rul_path = Path(data_dir) / "RUL_FD001.txt"
    true_rul = pd.read_csv(rul_path, sep=r"\s+", header=None, engine="python")[0]
    true_rul.index = true_rul.index + 1  # unit_number is 1-indexed
    true_rul.name = "true_RUL"
    return df, true_rul


def _compute_train_rul(df: pd.DataFrame) -> pd.Series:
    """RUL(t) = max_cycle_for_engine - t, capped at RUL_CAP.

    The cap reflects the modeling assumption (standard in C-MAPSS work)
    that degradation is not meaningfully distinguishable from "healthy"
    far from end-of-life, so we don't ask the model to regress a
    near-constant target across the flat part of the engine's life.
    """
    max_cycle = df.groupby("unit_number")["time_cycles"].transform("max")
    rul = (max_cycle - df["time_cycles"]).clip(upper=RUL_CAP)
    return rul


@dataclass
class ProfileResult:
    n_engines: int
    n_rows: int
    constant_sensors: list[str]
    near_constant_sensors: list[str]
    missing_by_column: dict
    engine_life_stats: dict
    leakage_risk_notes: list[str] = field(default_factory=list)


def profile_dataset(df: pd.DataFrame, std_threshold: float = 1e-4) -> ProfileResult:
    """Deterministic data-quality profile — no LLM involved.

    This is step 1 of the pipeline in the build plan: schema, missingness,
    constant sensors, per-engine life-length stats, and leakage risk notes
    (e.g. if a candidate feature is monotonic with cycle count in a way
    that trivially encodes the label).
    """
    sensor_std = df[SENSOR_COLS].std()
    constant = sensor_std[sensor_std == 0].index.tolist()
    near_constant = sensor_std[(sensor_std > 0) & (sensor_std < std_threshold)].index.tolist()

    life_lengths = df.groupby("unit_number")["time_cycles"].max()

    notes = []
    if "RUL" in df.columns:
        notes.append(
            "RUL is derived directly from time_cycles and per-engine max cycle; "
            "time_cycles itself must never be used as a raw feature or it trivially "
            "reconstructs the label."
        )

    return ProfileResult(
        n_engines=df["unit_number"].nunique(),
        n_rows=len(df),
        constant_sensors=constant,
        near_constant_sensors=near_constant,
        missing_by_column=df.isna().sum().to_dict(),
        engine_life_stats={
            "min_cycles": int(life_lengths.min()),
            "max_cycles": int(life_lengths.max()),
            "mean_cycles": float(life_lengths.mean()),
        },
        leakage_risk_notes=notes,
    )


def usable_sensor_columns(profile: ProfileResult) -> list[str]:
    """Sensors worth engineering features from: drop constant and
    near-constant channels identified during profiling."""
    drop = set(profile.constant_sensors) | set(profile.near_constant_sensors)
    return [c for c in SENSOR_COLS if c not in drop]


def engineer_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
    windows: tuple[int, ...] = ROLLING_WINDOWS,
    lags: tuple[int, ...] = LAG_STEPS,
) -> pd.DataFrame:
    """Add rolling mean/std and lag features, computed per engine so no
    information crosses an engine boundary.

    time_cycles and unit_number are index/grouping columns only — neither
    is ever passed to the model as a feature (that would leak the label).
    """
    out = df.copy()
    grouped = out.groupby("unit_number", group_keys=False)

    for w in windows:
        roll_mean = grouped[sensor_cols].rolling(window=w, min_periods=1).mean()
        roll_std = grouped[sensor_cols].rolling(window=w, min_periods=1).std().fillna(0.0)
        roll_mean = roll_mean.reset_index(level=0, drop=True)
        roll_std = roll_std.reset_index(level=0, drop=True)
        out = out.join(roll_mean.add_suffix(f"_roll_mean_{w}"))
        out = out.join(roll_std.add_suffix(f"_roll_std_{w}"))

    for lag in lags:
        lagged = grouped[sensor_cols].shift(lag)
        lagged = lagged.bfill()
        out = out.join(lagged.add_suffix(f"_lag_{lag}"))

    return out


def feature_columns(engineered: pd.DataFrame, sensor_cols: list[str]) -> list[str]:
    """The full feature set: raw sensors + op settings + every engineered
    column, explicitly excluding unit_number, time_cycles, and RUL."""
    excluded = set(INDEX_COLS) | {"RUL", "true_RUL"}
    engineered_cols = [c for c in engineered.columns if c not in excluded]
    # keep a stable, de-duplicated order
    ordered = list(OP_SETTING_COLS) + list(sensor_cols)
    ordered += [c for c in engineered_cols if c not in ordered]
    return [c for c in ordered if c in engineered.columns]


def last_cycle_per_engine(df: pd.DataFrame) -> pd.DataFrame:
    """For the test set, the official FD001 protocol scores only the
    *last observed cycle* of each engine against RUL_FD001.txt."""
    idx = df.groupby("unit_number")["time_cycles"].idxmax()
    return df.loc[idx].reset_index(drop=True)
