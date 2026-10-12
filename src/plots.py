"""Figure generation for the Engagement Profitability report.

Each function here takes an already-computed frame from :mod:`src.analysis` and
writes one PNG at 300 DPI. No function in this module performs an analysis of its
own: if a number appears on a chart it was calculated in :mod:`src.analysis`,
which keeps the charts and the tables telling the same story.

The visual grammar is deliberately plain. Bars are sorted, values are labelled
directly rather than left to a gridline, the palette carries meaning (red for
value lost, teal for value kept), and every figure carries a subtitle stating the
period and the unit so that a chart screenshotted out of the report still says
what it shows.

Usage::

    from src.plots import build_all_figures

    paths = build_all_figures(headline_facts, clients, utilization, facts)
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from loguru import logger

from src.analysis import WATERFALL_LABELS
from src.config import FIGURE_DPI, FIGURE_SIZE, HEADLINE_YEAR, Settings, get_settings

#: Colour palette. Value lost is red, value kept is teal, neutral structure is
#: grey-blue. Every pairing used on a chart separates by lightness as well as
#: hue, so the figures survive greyscale printing.
PALETTE: dict[str, str] = {
    "ink": "#1f2d3d",
    "muted": "#8a94a6",
    "grid": "#dfe3e8",
    "lost": "#b3453a",
    "kept": "#2e7d6f",
    "neutral": "#5b6b7f",
    "accent": "#3b6ea5",
    "amber": "#c98a2e",
    "violet": "#7a5ea8",
}

#: Colour used for each contract type, kept consistent across every figure.
CONTRACT_COLOURS: dict[str, str] = {
    "time-and-materials": PALETTE["accent"],
    "fixed-fee": PALETTE["lost"],
    "retainer": PALETTE["amber"],
}

#: Colour used for each client tier, kept consistent across every figure.
TIER_COLOURS: dict[str, str] = {
    "A": PALETTE["kept"],
    "B": PALETTE["accent"],
    "C": PALETTE["muted"],
}


def _money_millions(value: float, _position: int | None = None) -> str:
    """Format a currency value in millions for an axis tick.

    Args:
        value: Axis value in currency units.
        _position: Tick position, supplied by matplotlib and unused.

    Returns:
        A string such as ``$45M`` or ``-$2.5M``.
    """
    if value == 0:
        return "$0"
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value) / 1e6:,.1f}M".replace(".0M", "M")


def _percent(value: float, _position: int | None = None) -> str:
    """Format a fraction as a whole-percentage axis tick.

    Args:
        value: Axis value as a fraction.
        _position: Tick position, supplied by matplotlib and unused.

    Returns:
        A string such as ``85%``.
    """
    return f"{value:.0%}"


def _blend(start: str, end: str, fraction: float) -> tuple[float, float, float]:
    """Interpolate between two hex colours.

    Args:
        start: Hex colour at ``fraction`` 0.
        end: Hex colour at ``fraction`` 1.
        fraction: Position between the two colours.

    Returns:
        An RGB triple with components in the range 0 to 1.
    """
    low = np.array(matplotlib.colors.to_rgb(start))
    high = np.array(matplotlib.colors.to_rgb(end))
    return tuple(low + (high - low) * float(np.clip(fraction, 0.0, 1.0)))  # type: ignore[return-value]


def _style_axes(ax: plt.Axes) -> None:
    """Apply the house axis style: no top or right spine, faint horizontal grid.

    Args:
        ax: Axes to restyle in place.

    Returns:
        None.
    """
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PALETTE["grid"])
    ax.tick_params(colors=PALETTE["neutral"], labelsize=10, length=0)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def _titles(ax: plt.Axes, title: str, subtitle: str) -> None:
    """Write a left-aligned title and a smaller subtitle above the axes.

    Args:
        ax: Axes to annotate.
        title: Main title text.
        subtitle: Supporting line stating the period and the unit.

    Returns:
        None.
    """
    ax.set_title(
        title,
        loc="left",
        fontsize=15,
        fontweight="bold",
        color=PALETTE["ink"],
        pad=26,
    )
    ax.annotate(
        subtitle,
        xy=(0.0, 1.015),
        xycoords="axes fraction",
        fontsize=10.5,
        color=PALETTE["neutral"],
        ha="left",
        va="bottom",
    )


def _save(fig: plt.Figure, path: Path) -> Path:
    """Write a figure to disk at the configured resolution and close it.

    Args:
        fig: Figure to write.
        path: Destination PNG path.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.debug("Wrote figure {}", path.name)
    return path


def plot_margin_waterfall(
    waterfall: pd.DataFrame,
    path: Path,
    period_label: str,
) -> Path:
    """Draw the margin-leakage waterfall from rate-card value to actual margin.

    Args:
        waterfall: Frame produced by :func:`src.analysis.waterfall`.
        path: Destination PNG path.
        period_label: Period shown in the subtitle, such as ``FY2025``.

    Returns:
        The path written.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    _style_axes(ax)

    labels = [WATERFALL_LABELS.get(step, step) for step in waterfall["step"]]
    amounts = waterfall["amount"].to_numpy(dtype=float)
    running = waterfall["running_total"].to_numpy(dtype=float)
    opening = waterfall["is_opening"].to_numpy(dtype=bool)

    # Floating bars: a deduction hangs from the running total above it, an
    # opening or closing bar sits on the baseline.
    bottoms = np.where(opening, 0.0, running - amounts)
    # Red is reserved for value that was earned and then lost commercially.
    # Delivery and consultant cost are the real cost of doing the work, so they
    # are drawn neutral; colouring them as leakage would overstate the problem.
    leakage_steps = {"discount_amount", "unbilled_value", "write_off_amount"}
    colours = []
    for step, is_open in zip(waterfall["step"], opening, strict=True):
        if is_open:
            colours.append(PALETTE["ink"])
        elif step in leakage_steps:
            colours.append(PALETTE["lost"])
        else:
            colours.append(PALETTE["neutral"])
    colours[-1] = PALETTE["kept"]

    positions = np.arange(len(amounts))
    ax.bar(positions, amounts, bottom=bottoms, color=colours, width=0.62, zorder=3)

    # Connector lines between consecutive steps.
    for index in range(len(amounts) - 1):
        ax.plot(
            [index + 0.31, index + 0.69],
            [running[index], running[index]],
            color=PALETTE["muted"],
            linewidth=0.9,
            linestyle=(0, (3, 3)),
            zorder=2,
        )

    span = float(np.nanmax(running)) or 1.0
    for index, (amount, is_open) in enumerate(zip(amounts, opening, strict=True)):
        value = amount if is_open else -amount
        label = _money_millions(value)
        if not is_open:
            label = f"-{_money_millions(abs(value))}" if value else "$0"
        top = bottoms[index] + amount
        ax.annotate(
            label,
            xy=(index, max(top, bottoms[index])),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=9.5,
            fontweight="bold",
            color=PALETTE["ink"],
        )
        if not is_open and span:
            share = abs(value) / span
            ax.annotate(
                f"{share:.1%}",
                xy=(index, min(top, bottoms[index])),
                xytext=(0, -14),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color=PALETTE["neutral"],
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=10.5, color=PALETTE["ink"])
    ax.yaxis.set_major_formatter(FuncFormatter(_money_millions))
    ax.set_ylim(0, span * 1.16)

    leakage_total = -float(
        waterfall.loc[waterfall["step"].isin(leakage_steps), "amount"].sum()
    )
    _titles(
        ax,
        "Where the rate card goes",
        f"{period_label} · rate-card value delivered, less commercial leakage of "
        f"{_money_millions(leakage_total)} and the cost of delivering the work",
    )
    ax.annotate(
        "Red is value earned and then lost commercially; grey is the real cost of "
        "delivery. Percentage under each deduction is its share of rate-card value.",
        xy=(0.0, -0.20),
        xycoords="axes fraction",
        fontsize=9,
        color=PALETTE["neutral"],
    )
    fig.tight_layout()
    return _save(fig, path)


def plot_client_pareto(clients: pd.DataFrame, path: Path) -> Path:
    """Draw the client margin Pareto curve with tier bands.

    Args:
        clients: Frame produced by :func:`src.analysis.client_profitability`,
            ordered by margin descending.
        path: Destination PNG path.

    Returns:
        The path written.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    _style_axes(ax)

    positions = np.arange(len(clients))
    tier_colours = [TIER_COLOURS.get(tier, PALETTE["muted"]) for tier in clients["tier"]]
    ax.bar(positions, clients["margin"], color=tier_colours, width=0.72, zorder=3)
    ax.yaxis.set_major_formatter(FuncFormatter(_money_millions))
    ax.set_ylabel("Margin by client", fontsize=10.5, color=PALETTE["neutral"])
    ax.set_xlabel("Clients, ranked by lifetime margin", fontsize=10.5, color=PALETTE["neutral"])
    ax.set_xlim(-1, len(clients))

    cumulative = ax.twinx()
    cumulative.plot(
        positions,
        clients["cumulative_share"],
        color=PALETTE["ink"],
        linewidth=2.0,
        marker="o",
        markersize=3.0,
        zorder=4,
    )
    cumulative.set_ylim(0, 1.05)
    cumulative.yaxis.set_major_formatter(FuncFormatter(_percent))
    cumulative.set_ylabel("Cumulative share of firm margin", fontsize=10.5, color=PALETTE["neutral"])
    cumulative.tick_params(colors=PALETTE["neutral"], labelsize=10, length=0)
    for side in ("top", "left", "right"):
        cumulative.spines[side].set_visible(False)

    crossing = int(np.argmax(clients["cumulative_share"].to_numpy() >= 0.80)) + 1
    ax.axvline(crossing - 0.5, color=PALETTE["muted"], linewidth=1.0, linestyle=(0, (4, 4)), zorder=2)
    cumulative.axhline(0.80, color=PALETTE["muted"], linewidth=1.0, linestyle=(0, (4, 4)), zorder=2)
    # Sits just under the 80% guide, to the right of the crossing, where the
    # curve has already risen clear of the text.
    cumulative.annotate(
        f"{crossing} of {len(clients)} clients carry 80% of firm margin",
        xy=(crossing + 1.0, 0.735),
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color=PALETTE["ink"],
        zorder=5,
    )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=TIER_COLOURS[tier])
        for tier in ("A", "B", "C")
    ]
    ax.legend(
        handles,
        ["Tier A", "Tier B", "Tier C"],
        loc="center left",
        bbox_to_anchor=(0.52, 0.30),
        frameon=False,
        fontsize=10,
        labelcolor=PALETTE["neutral"],
    )

    _titles(
        ax,
        "A minority of clients carry the firm",
        "Three-year portfolio · bars are margin per client, line is cumulative share of firm margin",
    )
    fig.tight_layout()
    return _save(fig, path)


def plot_utilization_by_level(
    utilization: pd.DataFrame,
    path: Path,
    firm_utilization: float,
) -> Path:
    """Draw billable hours as a share of available hours, by grade.

    Args:
        utilization: Frame produced by
            :func:`src.analysis.utilization_by_level`.
        path: Destination PNG path.
        firm_utilization: Firm-wide utilisation, drawn as a reference line.

    Returns:
        The path written.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    _style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.8, alpha=0.9)

    ordered = utilization.sort_values("utilization")
    positions = np.arange(len(ordered))
    ax.barh(
        positions,
        ordered["utilization"],
        color=PALETTE["accent"],
        height=0.58,
        zorder=3,
    )
    ax.axvline(
        firm_utilization,
        color=PALETTE["lost"],
        linewidth=1.6,
        linestyle=(0, (5, 3)),
        zorder=4,
    )
    # Placed below the lowest bar, where nothing else is drawn.
    ax.annotate(
        f"Firm average {firm_utilization:.1%}",
        xy=(firm_utilization, -0.78),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        color=PALETTE["lost"],
        va="center",
    )

    for position, row in zip(positions, ordered.to_dict("records"), strict=True):
        ax.annotate(
            f"{row['utilization']:.1%}   {row['billable_hours']:,.0f} of {row['available_hours']:,.0f} h   ({int(row['consultants'])} people)",
            xy=(row["utilization"], position),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=9.5,
            color=PALETTE["ink"],
            va="center",
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(ordered["level"], fontsize=11, color=PALETTE["ink"])
    ax.xaxis.set_major_formatter(FuncFormatter(_percent))
    ax.set_xlim(0, max(1.0, float(ordered["utilization"].max()) * 1.55))
    ax.set_ylim(-1.05, len(ordered) - 0.45)
    ax.set_xlabel("Billable hours as a share of available hours", fontsize=10.5, color=PALETTE["neutral"])

    _titles(
        ax,
        "Seniority, not effort, drives the utilisation gap",
        "Three-year portfolio · billable hours over available hours, by grade",
    )
    fig.tight_layout()
    return _save(fig, path)


def plot_realization_vs_margin(
    facts: pd.DataFrame,
    deciles: pd.DataFrame,
    path: Path,
    floor: float = -0.35,
) -> Path:
    """Draw the relationship between realisation and margin rate.

    A handful of severely loss-making engagements sit far below the rest, and
    plotting them at full scale would squash the ninety-five percent of the
    portfolio that lies within a normal band. The y-axis is therefore clipped at
    ``floor``, clipped engagements are counted and disclosed on the figure, and
    the shape of the relationship is drawn from decile medians rather than a
    fitted straight line, which the data does not support.

    Args:
        facts: Engagement-level fact table.
        deciles: Frame produced by
            :func:`src.analysis.realization_decile_profile`.
        path: Destination PNG path.
        floor: Lowest margin rate shown before clipping, as a fraction.

    Returns:
        The path written.
    """
    usable = facts.dropna(subset=["realization_rate", "margin_rate"])
    visible = usable[usable["margin_rate"] >= floor]
    clipped = len(usable) - len(visible)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    _style_axes(ax)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.8, alpha=0.9)

    sizes = 18.0 + 260.0 * (
        visible["standard_value"] / max(float(usable["standard_value"].max()), 1.0)
    )
    for contract_type, group in visible.groupby("contract_type"):
        ax.scatter(
            group["realization_rate"],
            group["margin_rate"],
            s=sizes.loc[group.index],
            color=CONTRACT_COLOURS.get(contract_type, PALETTE["muted"]),
            alpha=0.55,
            edgecolors="white",
            linewidths=0.5,
            label=contract_type,
            zorder=3,
        )

    ax.plot(
        deciles["median_realization"],
        deciles["median_margin_rate"],
        color=PALETTE["ink"],
        linewidth=2.4,
        marker="o",
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=2.0,
        label="decile median",
        zorder=5,
    )

    ax.axhline(0.0, color=PALETTE["lost"], linewidth=1.1, zorder=2)
    ax.annotate(
        "break-even",
        xy=(usable["realization_rate"].min(), 0.0),
        xytext=(3, 4),
        textcoords="offset points",
        fontsize=9,
        color=PALETTE["lost"],
    )

    top = deciles["median_margin_rate"].iloc[-1]
    bottom = deciles["median_margin_rate"].iloc[0]
    ax.annotate(
        f"median margin rate is {(top - bottom) * 100:.1f} points higher\n"
        "in the strongest decile than in the weakest",
        xy=(0.025, 0.90),
        xycoords="axes fraction",
        fontsize=10.5,
        fontweight="bold",
        color=PALETTE["ink"],
        va="top",
    )

    ax.xaxis.set_major_formatter(FuncFormatter(_percent))
    ax.yaxis.set_major_formatter(FuncFormatter(_percent))
    ax.set_ylim(floor, max(0.45, float(visible["margin_rate"].max()) * 1.12))
    ax.set_xlabel("Realisation rate (billed over rate-card value)", fontsize=10.5, color=PALETTE["neutral"])
    ax.set_ylabel("Margin rate (margin over collections)", fontsize=10.5, color=PALETTE["neutral"])
    legend = ax.legend(loc="lower right", frameon=False, fontsize=10, ncol=2)
    for text in legend.get_texts():
        text.set_color(PALETTE["neutral"])

    note = "Bubble area is engagement size at rate card."
    if clipped:
        note += (
            f" {clipped} engagement{'s' if clipped != 1 else ''} below "
            f"{floor:.0%} margin omitted from the plot."
        )
    ax.annotate(
        note,
        xy=(0.0, -0.145),
        xycoords="axes fraction",
        fontsize=9,
        color=PALETTE["neutral"],
    )

    _titles(
        ax,
        "Realisation is the lever on margin",
        "Three-year portfolio · one point per engagement, with the decile median overlaid",
    )
    fig.tight_layout()
    return _save(fig, path)


def plot_overrun_bands(bands: pd.DataFrame, path: Path) -> Path:
    """Draw margin rate against delivery overrun, by overrun quintile.

    Args:
        bands: ``bands`` entry from :func:`src.analysis.overrun_vs_margin`.
        path: Destination PNG path.

    Returns:
        The path written.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    _style_axes(ax)

    positions = np.arange(len(bands))
    span = max(len(bands) - 1, 1)
    colours = [
        _blend(PALETTE["kept"], PALETTE["lost"], index / span)
        for index in range(len(bands))
    ]
    ax.bar(positions, bands["margin_rate"], color=colours, width=0.6, zorder=3)

    for position, row in zip(positions, bands.to_dict("records"), strict=True):
        ax.annotate(
            f"{row['margin_rate']:.1%}",
            xy=(position, row["margin_rate"]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=11,
            fontweight="bold",
            color=PALETTE["ink"],
        )
        ax.annotate(
            f"{row['mean_overrun']:.2f}x budget\n{int(row['engagements'])} engagements",
            xy=(position, 0),
            xytext=(0, -34),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=PALETTE["neutral"],
        )

    drop = float(bands["margin_rate"].iloc[0] - bands["margin_rate"].iloc[-1])
    ax.annotate(
        f"{drop * 100:.1f} points of margin lost\nbetween the tightest and loosest quintile",
        xy=(0.5, 0.93),
        xycoords="axes fraction",
        ha="center",
        fontsize=10.5,
        fontweight="bold",
        color=PALETTE["lost"],
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"Q{index + 1}" for index in positions], fontsize=11, color=PALETTE["ink"]
    )
    ax.yaxis.set_major_formatter(FuncFormatter(_percent))
    ax.set_ylim(0, float(bands["margin_rate"].max()) * 1.28)
    ax.set_xlabel(
        "Engagements grouped into quintiles by delivery overrun",
        fontsize=10.5,
        color=PALETTE["neutral"],
        labelpad=46,
    )
    ax.set_ylabel("Margin rate", fontsize=10.5, color=PALETTE["neutral"])

    _titles(
        ax,
        "Every extra hour of overrun costs margin",
        "Three-year portfolio · engagements in quintiles by hours delivered over hours budgeted",
    )
    fig.tight_layout()
    return _save(fig, path)


def build_all_figures(
    headline_facts: pd.DataFrame,
    clients: pd.DataFrame,
    utilization: pd.DataFrame,
    facts: pd.DataFrame,
    overrun_bands: pd.DataFrame,
    deciles: pd.DataFrame,
    settings: Settings | None = None,
    period_label: str | None = None,
) -> dict[str, Path]:
    """Render every report figure into ``reports/figures``.

    Args:
        headline_facts: Fact table for the headline period.
        clients: Frame produced by :func:`src.analysis.client_profitability`.
        utilization: Frame produced by
            :func:`src.analysis.utilization_by_level`.
        facts: Engagement-level fact table covering the whole portfolio.
        overrun_bands: ``bands`` entry from
            :func:`src.analysis.overrun_vs_margin`.
        deciles: Frame produced by
            :func:`src.analysis.realization_decile_profile`.
        settings: Optional settings override; defaults to the process singleton.
        period_label: Period shown on the waterfall. Defaults to ``FY<headline>``.

    Returns:
        Mapping of figure key to the path written.
    """
    from src.analysis import firm_utilization, waterfall

    resolved = settings or get_settings()
    label = period_label or f"FY{HEADLINE_YEAR}"
    figures = resolved.figures_dir

    paths = {
        "waterfall": plot_margin_waterfall(
            waterfall(headline_facts), figures / "margin_leakage_waterfall.png", label
        ),
        "pareto": plot_client_pareto(clients, figures / "client_margin_pareto.png"),
        "utilization": plot_utilization_by_level(
            utilization,
            figures / "utilization_by_level.png",
            firm_utilization(utilization),
        ),
        "scatter": plot_realization_vs_margin(
            facts, deciles, figures / "realization_vs_margin.png"
        ),
        "overrun": plot_overrun_bands(
            overrun_bands, figures / "margin_decay_by_overrun.png"
        ),
    }
    logger.info("Rendered {} figures to {}", len(paths), figures)
    return paths
