"""Plot builders for the replenishment dashboard.

``build_patch_ethogram`` is a per-patch stacked ethogram: each row is one
patch, x is time relative to patch entry. ``build_ethogram`` is a single
continuous session-wide timeline. Both are ported (matplotlib instead of
plotly) from AllenNeuralDynamics/Aind.Behavior.VrForaging.Dashboard's
``viz/patch_ethogram.py`` and ``viz/ethogram.py``, adapted to the SITES /
LICKS / POSITION_VELOCITY tables produced by this project.

``parse_patch_state`` / ``build_running_probability`` read the SOFTWARE_EVENTS
table's ``PatchState`` stream to track each patch's reward probability over
the course of the session. ``build_patch_offset_probability`` re-aligns that
same stream to each patch *exit*, one subplot per patch label.
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from utils import (
    BLOCK_SEPARATOR_COLOR,
    CHOICE_COLOR,
    FORCE_REWARD_COLOR,
    LICK_COLOR,
    REWARD_COLOR,
    VELOCITY_COLOR,
    get_color_from_site,
    patch_color,
)

# {"PatchId": 0, "Amount": 4.0, "Probability": 0.7, "Available": 999999.0}
_PATCH_STATE_SCHEMA = pl.Struct(
    {"PatchId": pl.Int64, "Amount": pl.Float64, "Probability": pl.Float64, "Available": pl.Float64}
)

_ROW_HEIGHT = 0.8  # fraction of 1.0 each row occupies


def build_patch_ethogram(
    sites: pl.DataFrame,
    licks: pl.DataFrame,
    velocity: pl.DataFrame,
    patch_lo: int,
    patch_hi: int,
) -> Figure:
    """Stacked ethogram: one horizontal band per patch, time relative to patch entry."""
    if sites.is_empty():
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.set_title("Patch ethogram (no data)")
        ax.axis("off")
        return fig

    patch_indices = sorted(
        sites.filter(
            (pl.col("patch_index") >= patch_lo) & (pl.col("patch_index") <= patch_hi)
        )["patch_index"]
        .unique()
        .to_list()
    )
    if not patch_indices:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.set_title("Patch ethogram (no patches in range)")
        ax.axis("off")
        return fig

    # Consistent color per patch *label* (type), regardless of which window is shown.
    patch_labels = sorted(sites["patch_label"].unique().to_list())
    patch_order = {label: i for i, label in enumerate(patch_labels)}

    lick_times = (
        licks.filter(pl.col("is_lick_onset"))["Time"].to_numpy()
        if not licks.is_empty()
        else np.array([], dtype=float)
    )
    vel_time = velocity["Time"].to_numpy()
    vel_val = velocity["velocity"].to_numpy()

    n_rows = len(patch_indices)
    fig, ax = plt.subplots(figsize=(11, max(3, 0.7 * n_rows + 1)))

    y_tickvals: list[float] = []
    y_ticktext: list[str] = []
    legend_handles: dict[str, object] = {}
    patch_to_block: dict[int, int] = {}

    for row_idx, patch_idx in enumerate(patch_indices):
        display_row = n_rows - 1 - row_idx  # earliest patch on top
        patch = sites.filter(pl.col("patch_index") == patch_idx).sort("start_time")
        if patch.is_empty():
            continue
        patch_to_block[patch_idx] = int(patch["block_index"][0])

        # --- align anchor: first non-interpatch/intersite site in the patch ---
        odor_mask = ~patch["site_label"].str.to_lowercase().str.starts_with("inter")
        odor_sites = patch.filter(odor_mask)
        t_anchor = float((odor_sites if odor_sites.height else patch)["start_time"][0])

        y0 = display_row
        y_center = display_row + _ROW_HEIGHT / 2
        patch_label_val = str(patch["patch_label"][0])
        y_tickvals.append(y_center)
        y_ticktext.append(f"{patch_label_val} / {patch_idx}")

        # --- site colored bands ---
        for site_row in patch.iter_rows(named=True):
            x0 = site_row["start_time"] - t_anchor
            width = site_row["stop_time"] - site_row["start_time"]
            color = get_color_from_site(
                site_row["site_label"], patch_order.get(site_row["patch_label"], 0)
            )
            ax.add_patch(
                Rectangle((x0, y0), width, _ROW_HEIGHT, facecolor=color, alpha=0.45, linewidth=0)
            )

        # --- velocity overlay (normalised to row height) ---
        t0_p = float(patch["start_time"][0])
        t1_p = float(patch["stop_time"][-1])
        vel_mask = (vel_time >= t0_p) & (vel_time <= t1_p)
        if vel_mask.any():
            v_win_t = vel_time[vel_mask]
            v_win_v = vel_val[vel_mask]
            v_max = float(v_win_v.max()) or 1.0
            v_norm = v_win_v / v_max * _ROW_HEIGHT + y0
            (line,) = ax.plot(v_win_t - t_anchor, v_norm, color=VELOCITY_COLOR, linewidth=1)
            legend_handles.setdefault("Velocity", line)

        # --- event markers ---
        def _col_times(col: str) -> np.ndarray:
            return patch[col].drop_nulls().to_numpy() - t_anchor

        def _scatter(name: str, times: np.ndarray, color: str, marker: str) -> None:
            handle = ax.scatter(
                times,
                np.full(len(times), y_center),
                marker=marker,
                color=color,
                s=64,
                edgecolors="black",
                linewidths=0.8,
                zorder=3,
            )
            legend_handles.setdefault(name, handle)

        choice_t = _col_times("choice_cue_time")
        reward_t = _col_times("reward_onset_time")
        forced_t = (
            patch.filter(pl.col("has_forced_rewards"))["reward_onset_time"].drop_nulls().to_numpy()
            - t_anchor
        )
        patch_licks = lick_times[(lick_times >= t0_p) & (lick_times <= t1_p)] - t_anchor

        if len(choice_t):
            _scatter("Choices", choice_t, CHOICE_COLOR, "s")
        if len(reward_t):
            _scatter("Rewards", reward_t, REWARD_COLOR, "o")
        if len(forced_t):
            _scatter("Force rewards", forced_t, FORCE_REWARD_COLOR, "o")
        if len(patch_licks):
            _scatter("Licks", patch_licks, LICK_COLOR, "|")

    # --- block separators: vertical dotted line at x=0 for the first patch of each block ---
    seen_blocks: set[int] = set()
    for row_idx, patch_idx in enumerate(patch_indices):
        display_row = n_rows - 1 - row_idx
        blk = patch_to_block.get(patch_idx)
        if blk is None or blk in seen_blocks:
            continue
        seen_blocks.add(blk)
        line = ax.vlines(
            0,
            display_row,
            display_row + _ROW_HEIGHT,
            colors=BLOCK_SEPARATOR_COLOR,
            linestyles="dotted",
            linewidth=2,
        )
        legend_handles.setdefault("Block boundary", line)

    ax.axvline(0, color="#888888", linewidth=0.8, zorder=0)
    ax.set_yticks(y_tickvals)
    ax.set_yticklabels(y_ticktext)
    ax.set_ylim(-0.1, n_rows + 0.1)
    ax.set_xlabel("Time from patch entry (s)")
    ax.spines[["top", "right"]].set_visible(False)
    if legend_handles:
        ax.legend(
            legend_handles.values(),
            legend_handles.keys(),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(legend_handles),
            frameon=False,
        )
    fig.tight_layout()
    return fig


def session_bounds(sites: pl.DataFrame, velocity: pl.DataFrame) -> tuple[float, float]:
    """First and last timestamps covered by the session."""
    if not sites.is_empty():
        sites_sorted = sites.sort("start_time")
        return float(sites_sorted["start_time"][0]), float(sites_sorted["stop_time"][-1])
    if not velocity.is_empty():
        return float(velocity["Time"].min()), float(velocity["Time"].max())
    return 0.0, 1.0


def _fmt_hms(seconds: float) -> str:
    """Format seconds as h:mm:ss (no leading zeros on hours if zero)."""
    total = int(abs(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    sign = "-" if seconds < 0 else ""
    if h:
        return f"{sign}{h}:{m:02d}:{s:02d}"
    return f"{sign}{m:02d}:{s:02d}"


def build_ethogram(
    sites: pl.DataFrame,
    licks: pl.DataFrame,
    velocity: pl.DataFrame,
    t_start: float | None = None,
    t_end: float | None = None,
) -> Figure:
    """Session-wide ethogram: one continuous timeline with site bands, velocity, and events."""
    t0, t1 = session_bounds(sites, velocity)
    win_start = t_start if t_start is not None else t0
    win_end = t_end if t_end is not None else t1
    origin = t0

    fig, ax = plt.subplots(figsize=(12, 4))
    ax2 = ax.twinx()  # 0-1 "paper" axis for event markers and full-height site bands
    legend_handles: dict[str, object] = {}

    if not sites.is_empty():
        patch_labels = sorted(sites["patch_label"].unique().to_list())
        patch_order = {label: i for i, label in enumerate(patch_labels)}

        visible = sites.filter(
            (pl.col("stop_time") >= win_start) & (pl.col("start_time") <= win_end)
        )
        for row in visible.iter_rows(named=True):
            x0 = row["start_time"] - origin
            width = row["stop_time"] - row["start_time"]
            color = get_color_from_site(row["site_label"], patch_order.get(row["patch_label"], 0))
            ax.axvspan(x0, x0 + width, color=color, alpha=0.4, linewidth=0, zorder=0)

        # --- block separators: vertical line at the start of the first site of each new block ---
        if "block_index" in sites.columns:
            prev_block = None
            for row in sites.sort("start_time").iter_rows(named=True):
                blk = row["block_index"]
                if blk is None:
                    continue
                if prev_block is None:
                    prev_block = blk
                    continue
                if blk != prev_block:
                    x = row["start_time"] - origin
                    if win_start - origin <= x <= win_end - origin:
                        line = ax.axvline(
                            x, color=BLOCK_SEPARATOR_COLOR, linestyle=":", linewidth=1.5, zorder=2
                        )
                        legend_handles.setdefault("Block boundary", line)
                    prev_block = blk

    vel_time = velocity["Time"].to_numpy()
    vel_val = velocity["velocity"].to_numpy()
    vel_mask = (vel_time >= win_start) & (vel_time <= win_end)
    if vel_mask.any():
        (line,) = ax.plot(
            vel_time[vel_mask] - origin, vel_val[vel_mask], color=VELOCITY_COLOR, linewidth=1.2, zorder=3
        )
        legend_handles.setdefault("Velocity", line)

    def _column_in_window(col: str) -> np.ndarray:
        if sites.is_empty() or col not in sites.columns:
            return np.array([], dtype=float)
        values = sites[col].drop_nulls().to_numpy()
        in_win = values[(values >= win_start) & (values <= win_end)]
        return in_win - origin

    def _marker(name: str, times: np.ndarray, y: float, color: str, symbol: str, size: int = 70) -> None:
        if len(times) == 0:
            return
        handle = ax2.scatter(
            times,
            np.full(len(times), y),
            marker=symbol,
            color=color,
            s=size,
            edgecolors="black",
            linewidths=1.0,
            zorder=4,
        )
        legend_handles.setdefault(name, handle)

    _marker("Choices", _column_in_window("choice_cue_time"), 0.40, CHOICE_COLOR, "s")
    _marker("Rewards", _column_in_window("reward_onset_time"), 0.60, REWARD_COLOR, "o")
    lick_times = (
        licks.filter(pl.col("is_lick_onset"))["Time"].to_numpy()
        if not licks.is_empty()
        else np.array([], dtype=float)
    )
    licks_in_win = lick_times[(lick_times >= win_start) & (lick_times <= win_end)] - origin
    _marker("Licks", licks_in_win, 0.80, LICK_COLOR, "|", size=90)

    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.spines[["top", "right"]].set_visible(False)

    n_ticks = 5
    tick_abs = np.linspace(win_start, win_end, n_ticks)
    ax.set_xticks(tick_abs - origin)
    ax.set_xticklabels([_fmt_hms(v - t0) for v in tick_abs])
    ax.set_xlim(win_start - origin, win_end - origin)
    ax.set_xlabel("Time (hh:mm:ss)")
    ax.set_ylabel("Velocity (cm/s)")
    ax.axhline(0, color="#bbbbbb", linewidth=0.8, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)

    if legend_handles:
        ax.legend(
            legend_handles.values(),
            legend_handles.keys(),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.12),
            ncol=len(legend_handles),
            frameon=False,
        )
    fig.tight_layout()
    return fig


def parse_patch_state(software_events: pl.DataFrame) -> pl.DataFrame:
    """Parse the ``PatchState`` global patch-state event stream into a tidy table.

    ``SOFTWARE_EVENTS`` is a generic (event_name, data, timestamp) log where ``data``
    is a JSON string whose shape depends on ``event_name``. ``PatchState`` is emitted
    for every patch (arm) on every environment tick, e.g.::

        {"PatchId": 0, "Amount": 4.0, "Probability": 0.7, "Available": 999999.0}

    ``PatchId`` is a small stable arm identity (0, 1, 2, ...), not the sequential
    per-visit ``patch_index`` used in SITES. It's resolved to a human-readable label
    (e.g. "High") via the ``ActivePatch`` event, which records
    ``{"label": ..., "state_index": ...}`` the first time each patch becomes active
    (``state_index`` == ``PatchId``).
    """
    patch_state = software_events.filter(pl.col("event_name") == "PatchState")
    if patch_state.is_empty():
        return pl.DataFrame(
            schema={
                "timestamp": pl.Float64,
                "patch_id": pl.Int64,
                "patch_label": pl.Utf8,
                "probability": pl.Float64,
                "amount": pl.Float64,
                "available": pl.Float64,
            }
        )

    parsed = (
        patch_state.select("timestamp", "data")
        .with_columns(pl.col("data").str.json_decode(_PATCH_STATE_SCHEMA).alias("_d"))
        .unnest("_d")
        .rename(
            {
                "PatchId": "patch_id",
                "Amount": "amount",
                "Probability": "probability",
                "Available": "available",
            }
        )
    )

    active_patch = software_events.filter(pl.col("event_name") == "ActivePatch")
    id_to_label: dict[int, str] = {}
    if not active_patch.is_empty():
        labels = active_patch.select(
            pl.col("data").str.json_path_match("$.state_index").cast(pl.Int64).alias("patch_id"),
            pl.col("data").str.json_path_match("$.label").alias("patch_label"),
        ).unique(subset="patch_id")
        id_to_label = dict(zip(labels["patch_id"].to_list(), labels["patch_label"].to_list()))

    return parsed.with_columns(
        pl.col("patch_id").replace_strict(id_to_label, default=None).alias("patch_label")
    ).select("timestamp", "patch_id", "patch_label", "probability", "amount", "available")


def build_running_probability(software_events: pl.DataFrame) -> Figure:
    """Running reward probability of every patch (arm) across the whole session."""
    parsed = parse_patch_state(software_events)
    if parsed.is_empty():
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.set_title("Running patch probability (no PatchState events)")
        ax.axis("off")
        return fig

    patch_labels_sorted = sorted(
        label for label in parsed["patch_label"].unique().to_list() if label is not None
    )
    patch_order = {label: i for i, label in enumerate(patch_labels_sorted)}
    t0 = float(parsed["timestamp"].min())

    fig, ax = plt.subplots(figsize=(11, 4))
    for patch_id in sorted(parsed["patch_id"].unique().to_list()):
        sub = parsed.filter(pl.col("patch_id") == patch_id).sort("timestamp")
        label = sub["patch_label"][0] or f"Patch {patch_id}"
        color = patch_color(patch_order.get(label, patch_id))
        ax.step(
            sub["timestamp"].to_numpy() - t0,
            sub["probability"].to_numpy(),
            where="post",
            color=color,
            linewidth=1.5,
            label=label,
        )

    ax.set_xlabel("Time (s from session start)")
    ax.set_ylabel("Reward probability")
    ax.set_ylim(0, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=max(1, len(patch_order)),
        frameon=False,
    )
    fig.tight_layout()
    return fig


# Reading order for subplots (a nicer-to-scan left-to-right layout); color assignment
# still comes from the alphabetical `patch_order` shared with the other plot builders.
_LABEL_READING_ORDER = ["High", "Medium", "Low"]


def build_patch_offset_probability(
    sites: pl.DataFrame,
    software_events: pl.DataFrame,
    t_start: float = -1.0,
    t_end: float = 5.0,
) -> Figure:
    """Reward probability aligned to patch *exit* ("offset"), one subplot per patch label.

    For every patch visit, "offset" is the moment the patch is left (the ``stop_time``
    of the last site in that visit). Each visit contributes one line: the visited
    patch's ``PatchState`` probability trace over ``[offset + t_start, offset + t_end]``
    (``t_start`` is typically negative, e.g. -1 for "1s before exit"), re-centered so
    the exit sits at x=0. All visits of a given patch label are overlaid on that
    label's subplot.

    ``PatchState`` only ticks when the probability actually changes -- on reward while
    a patch is active, then on a delayed replenishment schedule once it isn't -- so raw
    ticks can be sparse right around a given exit. Each line therefore carries the last
    known value forward into the window (the true value at any instant is simply
    whatever the most recent tick said), rather than only plotting ticks that happen to
    land inside the window.
    """
    parsed = parse_patch_state(software_events)
    if sites.is_empty() or parsed.is_empty():
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.set_title("Patch-offset probability (no data)")
        ax.axis("off")
        return fig

    # One row per patch visit: its label and its exit ("offset") time.
    offsets = (
        sites.sort("start_time")
        .group_by("patch_index", maintain_order=True)
        .agg(pl.col("patch_label").first(), pl.col("stop_time").last().alias("offset_time"))
    )

    labels_present = set(offsets["patch_label"].unique().to_list())
    patch_order = {label: i for i, label in enumerate(sorted(labels_present))}
    ordered_labels = [lbl for lbl in _LABEL_READING_ORDER if lbl in labels_present] + sorted(
        labels_present - set(_LABEL_READING_ORDER)
    )

    fig, axes = plt.subplots(
        1, len(ordered_labels), figsize=(4.5 * len(ordered_labels), 4), sharey=True
    )
    axes = np.atleast_1d(axes)

    for ax, label in zip(axes, ordered_labels):
        color = patch_color(patch_order[label])
        label_probs = parsed.filter(pl.col("patch_label") == label)
        prob_t = label_probs["timestamp"].to_numpy()
        prob_v = label_probs["probability"].to_numpy()

        visits = offsets.filter(pl.col("patch_label") == label).sort("offset_time")
        for offset_time in visits["offset_time"].to_list():
            window_start = offset_time + t_start
            window_end = offset_time + t_end

            in_window = (prob_t > window_start) & (prob_t <= window_end)
            xs = prob_t[in_window]
            ys = prob_v[in_window]

            # Carry the last known value (from before the window) forward to its start.
            before = prob_t <= window_start
            if before.any():
                xs = np.concatenate(([window_start], xs))
                ys = np.concatenate(([prob_v[before][-1]], ys))
            if len(xs) == 0:
                continue
            # ...and hold the final value flat out to the window's right edge.
            xs = np.concatenate((xs, [window_end]))
            ys = np.concatenate((ys, [ys[-1]]))

            ax.step(
                xs - offset_time,
                ys,
                where="post",
                color=color,
                alpha=0.5,
                linewidth=1,
            )

        ax.axvline(0, color="#888888", linewidth=0.8, zorder=0)
        ax.set_title(f"{label} (n={visits.height})")
        ax.set_xlabel("Time from patch exit (s)")
        ax.set_xlim(t_start, t_end)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Reward probability")
    axes[0].set_ylim(0, 1.02)
    fig.tight_layout()
    return fig
