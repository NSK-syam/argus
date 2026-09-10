import numpy as np
import pandas as pd
import pytest

from app.ml.data import cmapss


def _toy_engine(unit: int, n_cycles: int, sensor_start: float = 0.0) -> pd.DataFrame:
    """A synthetic single-engine trace: sensor_1 increases linearly from
    ``sensor_start``, everything else constant — enough to test labeling
    and rolling/lag mechanics without needing the real dataset."""
    cycles = np.arange(1, n_cycles + 1)
    df = pd.DataFrame({"unit_number": unit, "time_cycles": cycles})
    for c in cmapss.OP_SETTING_COLS:
        df[c] = 0.0
    for i, c in enumerate(cmapss.SENSOR_COLS):
        if c == "sensor_1":
            df[c] = sensor_start + cycles.astype(float)
        elif c == "sensor_2":
            df[c] = 100.0  # constant -> should be flagged by profiling
        else:
            df[c] = float(i)  # constant, distinct per sensor
    return df


def test_rul_capped_and_zero_at_end_of_life():
    df = _toy_engine(1, n_cycles=200)
    rul = cmapss._compute_train_rul(df)
    assert rul.iloc[-1] == 0  # last cycle of the engine's life -> RUL 0
    assert rul.max() == cmapss.RUL_CAP  # early cycles are capped, not left to grow unbounded
    assert rul.iloc[0] == cmapss.RUL_CAP


def test_rul_uncapped_case():
    df = _toy_engine(1, n_cycles=50)  # life shorter than the cap
    rul = cmapss._compute_train_rul(df)
    assert rul.iloc[0] == 49  # 50 - 1, no capping needed
    assert rul.iloc[-1] == 0


def test_rul_is_per_engine_not_global():
    long_engine = _toy_engine(1, n_cycles=50)
    short_engine = _toy_engine(2, n_cycles=10)
    df = pd.concat([long_engine, short_engine], ignore_index=True)
    rul = cmapss._compute_train_rul(df)
    # engine 2's RUL must be computed from its OWN max cycle (10), not
    # engine 1's max cycle (50) -- this is the classic leakage bug for
    # this kind of grouped label.
    engine2_first_rul = rul[df["unit_number"] == 2].iloc[0]
    assert engine2_first_rul == 9


def test_profile_flags_constant_sensor():
    df = _toy_engine(1, n_cycles=30)
    profile = cmapss.profile_dataset(df)
    assert "sensor_2" in profile.constant_sensors
    assert "sensor_1" not in profile.constant_sensors


def test_usable_sensor_columns_excludes_constants():
    df = _toy_engine(1, n_cycles=30)
    profile = cmapss.profile_dataset(df)
    usable = cmapss.usable_sensor_columns(profile)
    assert "sensor_2" not in usable
    assert "sensor_1" in usable


def test_feature_columns_never_include_index_or_label():
    df = _toy_engine(1, n_cycles=30)
    df["RUL"] = cmapss._compute_train_rul(df)
    engineered = cmapss.engineer_features(df, ["sensor_1"], windows=(5,), lags=(1,))
    feat_cols = cmapss.feature_columns(engineered, ["sensor_1"])
    assert "unit_number" not in feat_cols
    assert "time_cycles" not in feat_cols
    assert "RUL" not in feat_cols


def test_engineer_features_no_cross_engine_leakage():
    # engine 1: sensor_1 = 1000 + cycle (very different scale from engine 2)
    engine1 = _toy_engine(1, n_cycles=10, sensor_start=1000.0)
    # engine 2: sensor_1 = 0 + cycle
    engine2 = _toy_engine(2, n_cycles=10, sensor_start=0.0)
    df = pd.concat([engine1, engine2], ignore_index=True)

    engineered = cmapss.engineer_features(df, ["sensor_1"], windows=(5,), lags=(1,))
    engine2_first_row = engineered[engineered["unit_number"] == 2].iloc[0]

    # if engine 1's tail leaked into engine 2's first rolling window, the
    # rolling mean would be pulled toward ~1000; it must instead equal
    # engine 2's own first value.
    assert engine2_first_row["sensor_1_roll_mean_5"] == pytest.approx(1.0)
    # same check for the lag feature: engine 2's first lagged value must
    # back-fill from its OWN first observation, not engine 1's last one.
    assert engine2_first_row["sensor_1_lag_1"] == pytest.approx(1.0)


def test_last_cycle_per_engine_returns_one_row_each():
    engine1 = _toy_engine(1, n_cycles=10)
    engine2 = _toy_engine(2, n_cycles=25)
    df = pd.concat([engine1, engine2], ignore_index=True)
    last = cmapss.last_cycle_per_engine(df)
    assert len(last) == 2
    assert set(last["unit_number"]) == {1, 2}
    assert last.loc[last["unit_number"] == 1, "time_cycles"].item() == 10
    assert last.loc[last["unit_number"] == 2, "time_cycles"].item() == 25
