import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import polars as pl
    import os
    os.environ["POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR"] = "load_as_storage"


    SOURCE = "s3://aind-scratch-data/vr-foraging/replenishment-temp"

    def read_table(table_name: str) -> pl.DataFrame:
        source = SOURCE.rstrip("/")
        return pl.read_parquet(
            f"{source}/{table_name}.parquet",
            storage_options={"aws_skip_signature": "true", "aws_region": "us-west-2"},
        )

    SITES = read_table("sites")
    LICKS = read_table("licks")
    POSITION_VELOCITY = read_table("position_velocity")
    SESSION_METADATA = read_table("session")
    SOFTWARE_EVENTS = read_table("software_events")
    return LICKS, POSITION_VELOCITY, SITES, SOFTWARE_EVENTS, pl


@app.cell
def _(SITES, mo, pl):
    mo.ui.table(SITES.filter(pl.col("site_label") == "RewardSite"))
    return


@app.cell
def _(SITES):
    import marimo as mo

    from plotting import build_patch_ethogram

    # Widget to pick which window of patches (by patch_index) to render.
    _patch_indices = sorted(SITES["patch_index"].unique().to_list())
    _lo, _hi = (min(_patch_indices), max(_patch_indices)) if _patch_indices else (0, 0)

    patch_range = mo.ui.range_slider(
        start=_lo,
        stop=_hi,
        step=1,
        value=(_lo, min(_lo + 9, _hi)),
        label="Patch range",
        full_width=True,
        show_value=True,
    )
    return build_patch_ethogram, mo, patch_range


@app.cell
def _(LICKS, POSITION_VELOCITY, SITES, build_patch_ethogram, mo, patch_range):

    _lo, _hi = patch_range.value
    _fig = build_patch_ethogram(SITES, LICKS, POSITION_VELOCITY, _lo, _hi)
    mo.vstack([patch_range, _fig])
    return


@app.cell
def _(POSITION_VELOCITY, SITES, mo):
    from plotting import build_ethogram, session_bounds

    # Widgets to pick which time window to render: a center position + a window size,
    # both in seconds from session start.
    _t0, _t1 = session_bounds(SITES, POSITION_VELOCITY)
    _duration = _t1 - _t0
    _step = max(1.0, _duration / 200)

    window_center = mo.ui.slider(
        start=0,
        stop=_duration,
        step=_step,
        value=min(15.0, _duration),
        label="Window center (s from session start)",
        full_width=True,
        show_value=True,
    )
    window_size = mo.ui.slider(
        start=_step,
        stop=_duration,
        step=_step,
        value=min(30.0, _duration),
        label="Window size (s)",
        full_width=True,
        show_value=True,
    )
    return build_ethogram, session_bounds, window_center, window_size


@app.cell
def _(
    LICKS,
    POSITION_VELOCITY,
    SITES,
    build_ethogram,
    mo,
    session_bounds,
    window_center,
    window_size,
):

    _t0, _ = session_bounds(SITES, POSITION_VELOCITY)
    _half = window_size.value / 2
    _lo, _hi = window_center.value - _half, window_center.value + _half
    _fig = build_ethogram(SITES, LICKS, POSITION_VELOCITY, _t0 + _lo, _t0 + _hi)
    mo.vstack([window_center, window_size, _fig])
    return


@app.cell
def _(SOFTWARE_EVENTS, mo, pl):
    # --- Example: opening SOFTWARE_EVENTS and parsing the global PatchState event ---
    # SOFTWARE_EVENTS is a generic (event_name, data, timestamp) log: `data` is a JSON
    # string whose shape depends on `event_name`. Filter to the event you care about,
    # then decode `data`.
    #
    # `PatchState` is the "global" patch state: it's emitted for *every* patch (arm),
    # on every environment tick, e.g.:
    #     {"PatchId": 0, "Amount": 4.0, "Probability": 0.7, "Available": 999999.0}
    # `PatchId` is the arm's stable identity (0, 1, 2, ...) -- not the sequential
    # per-visit `patch_index` from SITES.
    _patch_state = SOFTWARE_EVENTS.filter(pl.col("event_name") == "PatchState")

    _patch_state_schema = pl.Struct(
        {"PatchId": pl.Int64, "Amount": pl.Float64, "Probability": pl.Float64, "Available": pl.Float64}
    )
    _parsed_example = (
        _patch_state.select("timestamp", "data")
        .with_columns(pl.col("data").str.json_decode(_patch_state_schema).alias("_d"))
        .unnest("_d")
    )
    mo.vstack(
        [
            mo.md("`PatchState` events decoded (reward probability per patch, per tick):"),
            _parsed_example.head(10),
        ]
    )
    return


@app.cell
def _(SOFTWARE_EVENTS, mo):
    from plotting import build_running_probability

    mo.vstack(
        [
            mo.md("### Running reward probability of all patches"),
            build_running_probability(SOFTWARE_EVENTS),
        ]
    )
    return


@app.cell
def _(SITES, SOFTWARE_EVENTS, mo):
    from plotting import build_patch_offset_probability

    mo.vstack(
        [
            mo.md("### Reward probability aligned to patch exit (-5s to +30s)"),
            build_patch_offset_probability(SITES, SOFTWARE_EVENTS, t_start=-5, t_end=30),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
