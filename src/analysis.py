"""Engagement profitability calculations.

This module is the single source of truth for every number the report and the
application display. Neither of them performs arithmetic of its own: they call
the functions here and format the result. If a definition changes, it changes
once.

The central object is the *fact table*: one row per engagement carrying its
hours, its money and its derived rates. A second, year-split fact table allocates
each engagement's money across calendar years in proportion to the hours worked
in each, which is how the headline fiscal year is isolated from an engagement
portfolio that spans three years.

Terminology used throughout:

``standard value``
    Delivered hours valued at the rate card. What the work was worth before any
    commercial conversation.
``billed amount``
    What was actually invoiced, after discounts and after time that could not be
    invoiced at all.
``collected amount``
    What the client actually paid. This is net revenue.
``realization rate``
    ``billed / standard value``. Measures how much of the rate card survived
    pricing and delivery.
``collection rate``
    ``collected / billed``. Measures how much of the invoice survived the
    client's accounts payable department.
``margin rate``
    ``margin / collected amount``. Margin is always expressed against net
    revenue, never against the rate card.

Usage::

    from src.analysis import build_facts, firm_summary, waterfall

    facts = build_facts()
    print(firm_summary(facts)["margin_rate"])
    print(waterfall(facts))
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.config import (
    HEADLINE_YEAR,
    SENSITIVITY_BOTTOM_QUANTILE,
    SENSITIVITY_POINTS,
    TIER_A_CUMULATIVE_CUT,
    TIER_B_CUMULATIVE_CUT,
    Settings,
    get_settings,
)

#: Money columns carried through the fact table and scaled when money is split
#: across years. Every one of them is additive across engagements.
MONEY_COLUMNS: tuple[str, ...] = (
    "standard_value",
    "discount_amount",
    "unbilled_value",
    "billed_amount",
    "write_off_amount",
    "collected_amount",
    "direct_delivery_cost",
    "consultant_cost",
    "margin",
)

#: The waterfall steps after the opening standard value. Each is subtracted.
WATERFALL_DEDUCTIONS: tuple[str, ...] = (
    "discount_amount",
    "unbilled_value",
    "write_off_amount",
    "direct_delivery_cost",
    "consultant_cost",
)

#: Human-readable label for each waterfall step.
WATERFALL_LABELS: dict[str, str] = {
    "standard_value": "Standard value",
    "discount_amount": "Discounts",
    "unbilled_value": "Unbilled time",
    "write_off_amount": "Write-offs",
    "direct_delivery_cost": "Direct delivery cost",
    "consultant_cost": "Consultant cost",
    "margin": "Actual margin",
}


@lru_cache(maxsize=1)
def load_raw(settings: Settings | None = None) -> dict[str, pd.DataFrame]:
    """Read every generated CSV from ``data/raw``.

    The result is cached for the process, which matters for the Streamlit app:
    every widget interaction reruns the script, and re-reading the CSVs each time
    would be wasteful. The returned frames are cached, so callers must not mutate
    them in place.

    Args:
        settings: Optional settings override; defaults to the process singleton.

    Returns:
        Mapping of table name to DataFrame.

    Raises:
        FileNotFoundError: If the dataset has not been generated yet.
    """
    resolved = settings or get_settings()
    names = ("consultants", "clients", "engagements", "timesheets", "billing")
    frames: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    for name in names:
        path = resolved.raw_dir / f"{name}.csv"
        if not path.exists():
            missing.append(str(path))
            continue
        frames[name] = pd.read_csv(path)

    if missing:
        raise FileNotFoundError(
            "Missing generated data: "
            + ", ".join(missing)
            + ". Run 'python cli.py generate-data' first."
        )
    logger.debug("Loaded {} tables from {}", len(frames), resolved.raw_dir)
    return frames


@lru_cache(maxsize=1)
def build_facts(settings: Settings | None = None) -> pd.DataFrame:
    """Build the engagement-level fact table.

    One row per engagement, joining its commercial terms from ``engagements.csv``
    to its money from ``billing.csv`` and its delivered hours from
    ``timesheets.csv``. Cached for the process; callers must not mutate the
    result in place.

    Args:
        settings: Optional settings override; defaults to the process singleton.

    Returns:
        DataFrame with one row per engagement and the derived rate columns
        ``realization_rate``, ``collection_rate``, ``margin_rate``,
        ``overrun_ratio`` and ``delivered_hours``.
    """
    raw = load_raw(settings)
    hours = (
        raw["timesheets"]
        .groupby("engagement_id", as_index=False)
        .agg(delivered_hours=("hours", "sum"))
    )
    facts = (
        raw["billing"]
        .merge(
            raw["engagements"],
            on="engagement_id",
            how="left",
            validate="one_to_one",
        )
        .merge(hours, on="engagement_id", how="left", validate="one_to_one")
    )
    return _add_rates(facts)


def _add_rates(facts: pd.DataFrame) -> pd.DataFrame:
    """Attach the derived rate and ratio columns to a fact table.

    Every division is guarded so that a zero denominator yields ``NaN`` rather
    than an exception or a silent infinity.

    Args:
        facts: Fact table carrying the money and hour columns.

    Returns:
        A copy of ``facts`` with the derived columns appended.
    """
    out = facts.copy()
    out["realization_rate"] = _safe_divide(
        out["billed_amount"], out["standard_value"]
    )
    out["collection_rate"] = _safe_divide(
        out["collected_amount"], out["billed_amount"]
    )
    out["margin_rate"] = _safe_divide(out["margin"], out["collected_amount"])
    out["overrun_ratio"] = _safe_divide(out["delivered_hours"], out["budgeted_hours"])
    out["discount_rate"] = _safe_divide(out["discount_amount"], out["standard_value"])
    out["write_off_rate"] = _safe_divide(
        out["write_off_amount"], out["billed_amount"]
    )
    return out


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series, returning ``NaN`` wherever the denominator is zero.

    Args:
        numerator: Values to divide.
        denominator: Values to divide by.

    Returns:
        The element-wise quotient as floats.
    """
    numerator = pd.to_numeric(numerator, errors="coerce").astype(float)
    denominator = pd.to_numeric(denominator, errors="coerce").astype(float)
    result = numerator.div(denominator.where(denominator != 0))
    return result.replace([np.inf, -np.inf], np.nan)


def allocate_money_by_year(
    facts: pd.DataFrame,
    timesheets: pd.DataFrame,
) -> pd.DataFrame:
    """Split each engagement's money across calendar years by hours worked.

    This is percentage-of-completion revenue recognition. An engagement's
    standard value, discount, billing, cost and margin are all allocated by the
    share of that engagement's hours that fall in each year, so the year rows for
    an engagement sum back to its engagement-level row.

    Args:
        facts: Engagement-level fact table.
        timesheets: Timesheet rows carrying ``engagement_id``, ``year`` and
            ``hours``.

    Returns:
        DataFrame with one row per engagement and year, carrying every money
        column plus the allocated ``delivered_hours`` and ``budgeted_hours``.
    """
    by_year = (
        timesheets.groupby(["engagement_id", "year"], as_index=False)
        .agg(delivered_hours=("hours", "sum"))
    )
    totals = by_year.groupby("engagement_id")["delivered_hours"].transform("sum")
    by_year = by_year.assign(
        hours_share=(by_year["delivered_hours"] / totals).fillna(0.0)
    )

    base_columns = ["engagement_id", "client_id", "client_name", "industry"]
    base_columns += ["service_line", "contract_type", "size_tier", "status"]
    allocated = by_year.merge(
        facts[base_columns + list(MONEY_COLUMNS) + ["budgeted_hours"]],
        on="engagement_id",
        how="left",
        validate="many_to_one",
    )

    share = allocated["hours_share"]
    for column in MONEY_COLUMNS:
        allocated[column] = allocated[column] * share
    allocated["budgeted_hours"] = allocated["budgeted_hours"] * share

    logger.debug(
        "Allocated {} engagement-years across {} years",
        len(allocated),
        allocated["year"].nunique(),
    )
    return _add_rates(allocated)


def firm_summary(facts: pd.DataFrame) -> dict[str, Any]:
    """Reduce a fact table to the firm-wide headline figures.

    Args:
        facts: Fact table at engagement or engagement-year grain.

    Returns:
        Mapping of metric name to value. Rates are expressed as fractions, not
        percentages.
    """
    total = {column: float(facts[column].sum()) for column in MONEY_COLUMNS}
    standard_value = total["standard_value"] or np.nan
    billed = total["billed_amount"] or np.nan
    collected = total["collected_amount"] or np.nan

    delivered_hours = float(facts.get("delivered_hours", pd.Series(dtype=float)).sum())
    budgeted_hours = float(facts.get("budgeted_hours", pd.Series(dtype=float)).sum())
    loss_making = facts[facts["margin"] < 0]

    summary: dict[str, Any] = {
        **total,
        "engagements": int(facts["engagement_id"].nunique()),
        "clients": int(facts["client_id"].nunique()),
        "delivered_hours": delivered_hours,
        "budgeted_hours": budgeted_hours,
        "realization_rate": total["billed_amount"] / standard_value,
        "collection_rate": total["collected_amount"] / billed,
        "margin_rate": total["margin"] / collected,
        "discount_rate": total["discount_amount"] / standard_value,
        "unbilled_rate": total["unbilled_value"] / standard_value,
        "write_off_rate": total["write_off_amount"] / billed,
        "overrun_ratio": delivered_hours / budgeted_hours if budgeted_hours else np.nan,
        "loss_making_engagements": int(len(loss_making)),
        "loss_making_margin": float(loss_making["margin"].sum()),
        "loss_making_share": len(loss_making) / len(facts) if len(facts) else np.nan,
    }
    return summary


def waterfall(facts: pd.DataFrame) -> pd.DataFrame:
    """Build the margin-leakage waterfall for a fact table.

    The steps are additive by construction: the opening standard value minus each
    deduction equals the closing margin. :func:`waterfall_reconciles` asserts
    that identity.

    Args:
        facts: Fact table at engagement or engagement-year grain.

    Returns:
        DataFrame with columns ``step``, ``label``, ``amount``, ``is_opening``
        and ``running_total``. ``amount`` is positive for the opening value and
        for the closing margin, and negative for each deduction.
    """
    opening = float(facts["standard_value"].sum())
    rows: list[dict[str, Any]] = [
        {
            "step": "standard_value",
            "label": WATERFALL_LABELS["standard_value"],
            "amount": opening,
            "is_opening": True,
            "running_total": opening,
        }
    ]

    running = opening
    for column in WATERFALL_DEDUCTIONS:
        amount = -float(facts[column].sum())
        running += amount
        rows.append(
            {
                "step": column,
                "label": WATERFALL_LABELS[column],
                "amount": amount,
                "is_opening": False,
                "running_total": running,
            }
        )

    closing = float(facts["margin"].sum())
    rows.append(
        {
            "step": "margin",
            "label": WATERFALL_LABELS["margin"],
            "amount": closing,
            "is_opening": True,
            "running_total": closing,
        }
    )
    return pd.DataFrame.from_records(rows)


def waterfall_reconciles(facts: pd.DataFrame, tolerance: float = 1e-6) -> bool:
    """Test whether the waterfall's steps sum to the closing margin.

    Args:
        facts: Fact table at engagement or engagement-year grain.
        tolerance: Absolute tolerance in currency units.

    Returns:
        True when the opening value less every deduction equals the margin.
    """
    opening = float(facts["standard_value"].sum())
    deductions = float(sum(facts[column].sum() for column in WATERFALL_DEDUCTIONS))
    expected = opening - deductions
    actual = float(facts["margin"].sum())
    return bool(np.isclose(expected, actual, atol=tolerance, rtol=0.0))


def utilization_by_consultant(
    timesheets: pd.DataFrame,
    consultants: pd.DataFrame,
) -> pd.DataFrame:
    """Measure billable hours as a share of available hours per consultant.

    Args:
        timesheets: Timesheet rows carrying ``consultant_id`` and ``hours``.
        consultants: Consultant roster carrying ``available_hours``.

    Returns:
        DataFrame with one row per consultant and columns ``billable_hours``,
        ``available_hours``, ``utilization`` and ``unused_hours``.
    """
    billable = (
        timesheets.groupby("consultant_id", as_index=False)
        .agg(billable_hours=("hours", "sum"))
    )
    frame = consultants.merge(
        billable, on="consultant_id", how="left", validate="one_to_one"
    )
    frame["billable_hours"] = frame["billable_hours"].fillna(0.0)
    frame["utilization"] = _safe_divide(frame["billable_hours"], frame["available_hours"])
    frame["unused_hours"] = frame["available_hours"] - frame["billable_hours"]
    return frame


def utilization_by_level(utilization: pd.DataFrame) -> pd.DataFrame:
    """Aggregate utilisation to grade.

    Utilisation is recomputed from summed hours rather than averaged across
    consultants, so a part-year joiner does not carry the same weight as a full
    year of availability.

    Args:
        utilization: Frame produced by :func:`utilization_by_consultant`.

    Returns:
        DataFrame indexed by level with ``consultants``, ``billable_hours``,
        ``available_hours`` and ``utilization``.
    """
    grouped = utilization.groupby("level", as_index=False).agg(
        consultants=("consultant_id", "nunique"),
        billable_hours=("billable_hours", "sum"),
        available_hours=("available_hours", "sum"),
    )
    grouped["utilization"] = _safe_divide(
        grouped["billable_hours"], grouped["available_hours"]
    )
    return grouped.sort_values("utilization", ascending=False).reset_index(drop=True)


def firm_utilization(utilization: pd.DataFrame) -> float:
    """Return firm-wide utilisation from a per-consultant utilisation frame.

    Args:
        utilization: Frame produced by :func:`utilization_by_consultant`.

    Returns:
        Billable hours divided by available hours, firm wide.
    """
    available = float(utilization["available_hours"].sum())
    if available == 0:
        return float("nan")
    return float(utilization["billable_hours"].sum()) / available


def group_by(facts: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Aggregate a fact table along one dimension.

    Args:
        facts: Fact table at engagement or engagement-year grain.
        dimension: Column to group by, such as ``service_line`` or
            ``contract_type``.

    Returns:
        DataFrame with one row per member of ``dimension`` and the summed money
        columns, together with realisation, collection and margin rates.

    Raises:
        KeyError: If ``dimension`` is not a column of ``facts``.
    """
    if dimension not in facts.columns:
        raise KeyError(f"{dimension!r} is not a column of the fact table")

    aggregations: dict[str, tuple[str, str]] = {
        column: (column, "sum") for column in MONEY_COLUMNS
    }
    aggregations["engagements"] = ("engagement_id", "nunique")
    if "delivered_hours" in facts.columns:
        aggregations["delivered_hours"] = ("delivered_hours", "sum")
    if "budgeted_hours" in facts.columns:
        aggregations["budgeted_hours"] = ("budgeted_hours", "sum")

    grouped = facts.groupby(dimension, as_index=False).agg(**aggregations)
    grouped["realization_rate"] = _safe_divide(
        grouped["billed_amount"], grouped["standard_value"]
    )
    grouped["collection_rate"] = _safe_divide(
        grouped["collected_amount"], grouped["billed_amount"]
    )
    grouped["margin_rate"] = _safe_divide(grouped["margin"], grouped["collected_amount"])
    if "delivered_hours" in grouped.columns:
        grouped["overrun_ratio"] = _safe_divide(
            grouped["delivered_hours"], grouped["budgeted_hours"]
        )
    return grouped.sort_values("margin", ascending=False).reset_index(drop=True)


def realization_by_level(
    facts: pd.DataFrame,
    timesheets: pd.DataFrame,
) -> pd.DataFrame:
    """Attribute realisation to the grade that delivered the work.

    Billing is recorded against an engagement, not a grade, so each engagement's
    billed and collected amounts are apportioned across grades in proportion to
    the standard value each grade contributed. The allocation is an assumption,
    stated plainly: it answers "what would realisation look like if every grade's
    time were priced and collected on the same terms as its engagement".

    Args:
        facts: Engagement-level fact table.
        timesheets: Timesheet rows carrying ``level``, ``hours`` and
            ``standard_rate``.

    Returns:
        DataFrame with one row per grade and its allocated realisation,
        collection and margin rates.
    """
    enriched = timesheets.assign(
        standard_value=timesheets["hours"] * timesheets["standard_rate"]
    )
    by_level = enriched.groupby(["engagement_id", "level"], as_index=False).agg(
        level_standard_value=("standard_value", "sum"),
        hours=("hours", "sum"),
    )
    engagement_total = by_level.groupby("engagement_id")["level_standard_value"].transform(
        "sum"
    )
    by_level["value_share"] = _safe_divide(
        by_level["level_standard_value"], engagement_total
    )

    merged = by_level.merge(
        facts[["engagement_id", "standard_value", "billed_amount", "collected_amount", "margin"]],
        on="engagement_id",
        how="left",
        validate="many_to_one",
    )
    for column in ("standard_value", "billed_amount", "collected_amount", "margin"):
        merged[column] = merged[column] * merged["value_share"]

    grouped = merged.groupby("level", as_index=False).agg(
        hours=("hours", "sum"),
        standard_value=("standard_value", "sum"),
        billed_amount=("billed_amount", "sum"),
        collected_amount=("collected_amount", "sum"),
        margin=("margin", "sum"),
    )
    grouped["realization_rate"] = _safe_divide(
        grouped["billed_amount"], grouped["standard_value"]
    )
    grouped["collection_rate"] = _safe_divide(
        grouped["collected_amount"], grouped["billed_amount"]
    )
    grouped["margin_rate"] = _safe_divide(grouped["margin"], grouped["collected_amount"])
    return grouped.sort_values("standard_value", ascending=False).reset_index(drop=True)


def overrun_vs_margin(facts: pd.DataFrame, bins: int = 5) -> dict[str, Any]:
    """Quantify the relationship between delivery overrun and margin decay.

    Args:
        facts: Engagement-level fact table.
        bins: Number of overrun bands to summarise.

    Returns:
        Mapping with the Pearson ``correlation`` between the overrun ratio and
        the margin rate, the fitted ``slope`` of margin rate per unit of overrun,
        and ``bands``: a DataFrame of overrun quintiles with their engagement
        count, total margin and mean margin rate.
    """
    usable = facts.dropna(subset=["overrun_ratio", "margin_rate"])
    correlation = float(usable["overrun_ratio"].corr(usable["margin_rate"]))
    slope = float(
        np.polyfit(usable["overrun_ratio"], usable["margin_rate"], 1)[0]
    )

    banded = usable.copy()
    banded["overrun_band"] = pd.qcut(
        banded["overrun_ratio"], q=bins, duplicates="drop"
    )
    grouped = banded.groupby("overrun_band", as_index=False, observed=True).agg(
        engagements=("engagement_id", "nunique"),
        mean_overrun=("overrun_ratio", "mean"),
        standard_value=("standard_value", "sum"),
        margin=("margin", "sum"),
        collected_amount=("collected_amount", "sum"),
    )
    grouped["margin_rate"] = _safe_divide(
        grouped["margin"], grouped["collected_amount"]
    )
    return {"correlation": correlation, "slope": slope, "bands": grouped}


def realization_decile_profile(facts: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Profile margin rate across the realisation distribution.

    The relationship between realisation and margin is monotone but not linear,
    and it is heteroscedastic: the low-realisation tail holds a handful of
    severely loss-making engagements whose spread would dominate any straight
    line fitted through it. Grouping into deciles and taking the median within
    each shows the shape of the relationship without letting those few points
    dictate it.

    Args:
        facts: Engagement-level fact table.
        bins: Number of realisation groups to form.

    Returns:
        DataFrame with one row per decile, ordered from worst realisation
        upward, carrying ``median_realization``, ``median_margin_rate``,
        ``engagements``, ``collected_amount`` and ``margin``.
    """
    usable = facts.dropna(subset=["realization_rate", "margin_rate"]).copy()
    usable["realization_decile"] = pd.qcut(
        usable["realization_rate"], q=bins, labels=False, duplicates="drop"
    )
    grouped = usable.groupby("realization_decile", as_index=False).agg(
        median_realization=("realization_rate", "median"),
        median_margin_rate=("margin_rate", "median"),
        engagements=("engagement_id", "nunique"),
        collected_amount=("collected_amount", "sum"),
        margin=("margin", "sum"),
    )
    return grouped.sort_values("median_realization").reset_index(drop=True)


def assign_client_tiers(
    client_margin: pd.Series,
    a_cut: float = TIER_A_CUMULATIVE_CUT,
    b_cut: float = TIER_B_CUMULATIVE_CUT,
) -> pd.DataFrame:
    """Tier clients by how much of the firm's margin they carry.

    Clients are ranked by margin, largest first, and assigned a tier by where
    their cumulative margin lands: tier A up to ``a_cut`` of firm margin, tier B
    up to ``b_cut``, tier C for the rest. A client whose own cumulative share
    exceeds the A cut is still placed in A if it is the largest, so tier A is
    never empty.

    Args:
        client_margin: Margin by client. The index is the client identifier.
        a_cut: Cumulative margin share that closes tier A. Inclusive.
        b_cut: Cumulative margin share that closes tier B. Inclusive.

    Returns:
        DataFrame indexed by client, sorted by margin descending, with columns
        ``margin``, ``cumulative_margin``, ``cumulative_share``, ``rank`` and
        ``tier``.

    Raises:
        ValueError: If total margin across the supplied clients is not positive,
            in which case a cumulative share is undefined.
    """
    ordered = client_margin.sort_values(ascending=False)
    total = float(ordered.sum())
    if total <= 0:
        raise ValueError("Total client margin must be positive to assign tiers")

    cumulative = ordered.cumsum()
    share = cumulative / total
    tier = np.where(share <= a_cut, "A", np.where(share <= b_cut, "B", "C"))
    if len(tier):
        tier[0] = "A"

    return pd.DataFrame(
        {
            "margin": ordered,
            "cumulative_margin": cumulative,
            "cumulative_share": share,
            "rank": np.arange(1, len(ordered) + 1),
            "tier": tier,
        }
    )


def client_profitability(facts: pd.DataFrame) -> pd.DataFrame:
    """Build the client profitability table with A/B/C tiers attached.

    Args:
        facts: Fact table at engagement or engagement-year grain.

    Returns:
        DataFrame with one row per client, ordered by margin descending, carrying
        revenue, margin, rates, engagement count, tier and cumulative share.
    """
    grouped = facts.groupby(["client_id", "client_name"], as_index=False).agg(
        industry=("industry", "first"),
        engagements=("engagement_id", "nunique"),
        standard_value=("standard_value", "sum"),
        billed_amount=("billed_amount", "sum"),
        collected_amount=("collected_amount", "sum"),
        direct_delivery_cost=("direct_delivery_cost", "sum"),
        consultant_cost=("consultant_cost", "sum"),
        margin=("margin", "sum"),
    )
    grouped["realization_rate"] = _safe_divide(
        grouped["billed_amount"], grouped["standard_value"]
    )
    grouped["margin_rate"] = _safe_divide(
        grouped["margin"], grouped["collected_amount"]
    )

    tiers = assign_client_tiers(grouped.set_index("client_id")["margin"])
    grouped = grouped.merge(
        tiers.reset_index()[["client_id", "rank", "cumulative_share", "tier"]],
        on="client_id",
        how="left",
        validate="one_to_one",
    )
    return grouped.sort_values("rank").reset_index(drop=True)


def tier_summary(tiers: pd.DataFrame) -> pd.DataFrame:
    """Summarise client tiers by revenue, margin and count.

    Args:
        tiers: Frame produced by :func:`client_profitability`.

    Returns:
        DataFrame with one row per tier carrying client count, revenue, margin,
        margin rate and each tier's share of revenue and of margin.
    """
    grouped = tiers.groupby("tier", as_index=False).agg(
        clients=("client_id", "nunique"),
        revenues=("collected_amount", "sum"),
        margin=("margin", "sum"),
    )
    grouped["margin_rate"] = _safe_divide(grouped["margin"], grouped["revenues"])
    grouped["revenue_share"] = grouped["revenues"] / grouped["revenues"].sum()
    grouped["margin_share"] = grouped["margin"] / grouped["margin"].sum()
    return grouped.sort_values("tier").reset_index(drop=True)


def bottom_quartile_engagements(
    facts: pd.DataFrame,
    quantile: float = SENSITIVITY_BOTTOM_QUANTILE,
) -> pd.DataFrame:
    """Select the weakest engagements by realisation rate.

    Args:
        facts: Engagement-level fact table.
        quantile: Share of the portfolio treated as the bottom group.

    Returns:
        The subset of ``facts`` at or below the realisation-rate quantile, ranked
        from worst realisation upward.
    """
    threshold = float(facts["realization_rate"].quantile(quantile))
    selected = facts[facts["realization_rate"] <= threshold]
    return selected.sort_values("realization_rate").reset_index(drop=True)


def uplift_value(subset: pd.DataFrame, points: float) -> float:
    """Value the margin from lifting realisation on a subset of engagements.

    Raising realisation by ``points`` lifts billing by that share of rate-card
    value. The extra billing is collected at the subset's existing collection
    rate and carries no incremental delivery or consultant cost, because the
    hours have already been worked. This is the single definition of the
    recovery arithmetic; every sensitivity and every recommendation uses it.

    Args:
        subset: Fact table rows the uplift applies to.
        points: Realisation points to recover, as a percentage.

    Returns:
        Incremental margin in currency units.
    """
    standard_value = float(subset["standard_value"].sum())
    billed = float(subset["billed_amount"].sum())
    collected = float(subset["collected_amount"].sum())
    collection_rate = collected / billed if billed else 0.0
    return standard_value * points / 100.0 * collection_rate


def sensitivity_realization(
    facts: pd.DataFrame,
    points: Sequence[float] = SENSITIVITY_POINTS,
    quantile: float = SENSITIVITY_BOTTOM_QUANTILE,
) -> pd.DataFrame:
    """Value the margin recovered by lifting realisation on the weakest work.

    Applies :func:`uplift_value` to the engagements in the bottom realisation
    quartile.

    Args:
        facts: Engagement-level fact table.
        points: Realisation points to recover, as percentages.
        quantile: Share of the portfolio treated as the bottom group.

    Returns:
        DataFrame with one row per value in ``points``, carrying the group's
        engagement count and standard value, its collection rate, the incremental
        billing and the incremental margin.
    """
    selected = bottom_quartile_engagements(facts, quantile)
    standard_value = float(selected["standard_value"].sum())
    billed = float(selected["billed_amount"].sum())
    collected = float(selected["collected_amount"].sum())
    collection_rate = collected / billed if billed else float("nan")

    records: list[dict[str, Any]] = []
    for point in points:
        records.append(
            {
                "points": float(point),
                "engagements": int(len(selected)),
                "standard_value": standard_value,
                "collection_rate": collection_rate,
                "incremental_billing": standard_value * point / 100.0,
                "incremental_margin": uplift_value(selected, point),
            }
        )
    return pd.DataFrame.from_records(records)


def sensitivity_at(
    facts: pd.DataFrame,
    points: float,
    quantile: float = SENSITIVITY_BOTTOM_QUANTILE,
) -> float:
    """Return the incremental margin from recovering a single point level.

    Args:
        facts: Engagement-level fact table.
        points: Realisation points to recover, as a percentage.
        quantile: Share of the portfolio treated as the bottom group.

    Returns:
        Incremental firm margin in currency units.
    """
    table = sensitivity_realization(facts, [points], quantile)
    return float(table.loc[0, "incremental_margin"])


def annual_summary(year_facts: pd.DataFrame) -> pd.DataFrame:
    """Summarise every fiscal year present in a year-split fact table.

    Args:
        year_facts: Frame produced by :func:`allocate_money_by_year`.

    Returns:
        DataFrame with one row per year and the headline rates for that year.
    """
    records: list[dict[str, Any]] = []
    for year, group in year_facts.groupby("year"):
        summary = firm_summary(group)
        summary["year"] = int(year)
        records.append(summary)
    columns = ["year", "engagements", "standard_value", "billed_amount", "collected_amount", "margin", "realization_rate", "collection_rate", "margin_rate"]
    return pd.DataFrame.from_records(records)[columns].sort_values("year").reset_index(drop=True)


def headline_year_facts(year_facts: pd.DataFrame, year: int = HEADLINE_YEAR) -> pd.DataFrame:
    """Select one fiscal year from a year-split fact table.

    Args:
        year_facts: Frame produced by :func:`allocate_money_by_year`.
        year: Fiscal year to select.

    Returns:
        The rows of ``year_facts`` belonging to ``year``.

    Raises:
        ValueError: If the requested year is absent from the data.
    """
    selected = year_facts[year_facts["year"] == year]
    if selected.empty:
        available = sorted(int(value) for value in year_facts["year"].unique())
        raise ValueError(f"Year {year} is not in the data. Available: {available}")
    return selected.reset_index(drop=True)
