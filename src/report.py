"""Analysis bundle, tabular exports and the self-contained HTML report.

This module ties the pipeline together: it calls :mod:`src.analysis` for every
number, :mod:`src.plots` for every figure, and assembles the results into one
HTML file with the images inlined as base64 data URIs. The report opens from the
filesystem with no server, no stylesheet fetch and no network access of any kind,
which is what makes it safe to attach to an email or drop into a repository.

Usage::

    from src.report import run_analysis

    report_path = run_analysis()
"""

from __future__ import annotations

import base64
import html
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from src.analysis import (
    MONEY_COLUMNS,
    allocate_money_by_year,
    annual_summary,
    build_facts,
    bottom_quartile_engagements,
    client_profitability,
    firm_summary,
    firm_utilization,
    group_by,
    headline_year_facts,
    load_raw,
    overrun_vs_margin,
    realization_by_level,
    realization_decile_profile,
    sensitivity_realization,
    tier_summary,
    uplift_value,
    utilization_by_consultant,
    utilization_by_level,
    waterfall,
)
from src.config import (
    HEADLINE_YEAR,
    SENSITIVITY_POINTS,
    Settings,
    get_settings,
)
from src.plots import build_all_figures


@dataclass(frozen=True)
class AnalysisBundle:
    """Everything the report, the tables and the application need.

    Attributes:
        settings: Settings the bundle was built with.
        facts: Engagement-level fact table for the whole portfolio.
        year_facts: Fact table split by engagement and calendar year.
        headline: Fact rows for the headline fiscal year.
        portfolio: Firm-wide summary across the whole portfolio.
        headline_summary: Firm-wide summary for the headline year.
        annual: One summary row per fiscal year.
        waterfall_headline: Margin-leakage waterfall for the headline year.
        by_service_line: Metrics grouped by service line.
        by_contract: Metrics grouped by contract type.
        by_level: Realisation attributed to each grade.
        utilization: Utilisation by grade.
        utilization_firm: Firm-wide utilisation.
        clients: Client profitability table with tiers.
        tiers: Summary by client tier.
        overrun: Overrun-to-margin relationship, including its bands.
        deciles: Margin rate across realisation deciles.
        sensitivity: Realisation-recovery sensitivity at each point level.
        figures: Mapping of figure key to the PNG written.
    """

    settings: Settings
    facts: pd.DataFrame
    year_facts: pd.DataFrame
    headline: pd.DataFrame
    portfolio: dict[str, Any]
    headline_summary: dict[str, Any]
    annual: pd.DataFrame
    waterfall_headline: pd.DataFrame
    by_service_line: pd.DataFrame
    by_contract: pd.DataFrame
    by_level: pd.DataFrame
    utilization: pd.DataFrame
    utilization_firm: float
    clients: pd.DataFrame
    tiers: pd.DataFrame
    overrun: dict[str, Any]
    deciles: pd.DataFrame
    sensitivity: pd.DataFrame
    figures: dict[str, Path]


def build_bundle(settings: Settings | None = None) -> AnalysisBundle:
    """Compute every metric, table and figure the deliverables need.

    Args:
        settings: Optional settings override; defaults to the process singleton.

    Returns:
        A fully populated :class:`AnalysisBundle`.
    """
    resolved = settings or get_settings()
    raw = load_raw(resolved)
    facts = build_facts(resolved)
    year_facts = allocate_money_by_year(facts, raw["timesheets"])
    headline = headline_year_facts(year_facts, resolved.headline_year)

    utilization = utilization_by_consultant(raw["timesheets"], raw["consultants"])
    by_level = utilization_by_level(utilization)
    clients = client_profitability(facts)
    overrun = overrun_vs_margin(facts)
    deciles = realization_decile_profile(facts)

    figures = build_all_figures(
        headline_facts=headline,
        clients=clients,
        utilization=by_level,
        facts=facts,
        overrun_bands=overrun["bands"],
        deciles=deciles,
        settings=resolved,
    )

    return AnalysisBundle(
        settings=resolved,
        facts=facts,
        year_facts=year_facts,
        headline=headline,
        portfolio=firm_summary(facts),
        headline_summary=firm_summary(headline),
        annual=annual_summary(year_facts),
        waterfall_headline=waterfall(headline),
        by_service_line=group_by(facts, "service_line"),
        by_contract=group_by(facts, "contract_type"),
        by_level=realization_by_level(facts, raw["timesheets"]),
        utilization=by_level,
        utilization_firm=firm_utilization(utilization),
        clients=clients,
        tiers=tier_summary(clients),
        overrun=overrun,
        deciles=deciles,
        sensitivity=sensitivity_realization(facts),
        figures=figures,
    )


def recommendations(bundle: AnalysisBundle) -> list[dict[str, Any]]:
    """Derive the three prioritised actions from the computed metrics.

    Each recommendation carries a value in currency units produced by
    :func:`src.analysis.uplift_value`, so the memo cannot drift away from the
    numbers. The three levers act on overlapping populations - a small fixed-fee
    engagement with poor realisation is counted by all three - so the values are
    individually defensible but are not additive.

    Args:
        bundle: Populated analysis bundle.

    Returns:
        List of recommendation dictionaries, largest value first.
    """
    facts = bundle.facts
    fixed_fee = facts[facts["contract_type"] == "fixed-fee"]
    small = facts[facts["size_tier"] == "small"]
    low_realization = bottom_quartile_engagements(facts)

    def _rate(contract_type: str) -> float:
        """Return realisation for one contract type.

        Args:
            contract_type: Contract type to look up.

        Returns:
            Realisation rate as a fraction.
        """
        return float(
            bundle.by_contract.loc[
                bundle.by_contract["contract_type"] == contract_type, "realization_rate"
            ].iloc[0]
        )

    def _overrun(contract_type: str) -> float:
        """Return the mean overrun ratio for one contract type.

        Args:
            contract_type: Contract type to look up.

        Returns:
            Delivered hours over budgeted hours.
        """
        return float(
            bundle.by_contract.loc[
                bundle.by_contract["contract_type"] == contract_type, "overrun_ratio"
            ].iloc[0]
        )

    # Expressed in percentage points, which is the unit uplift_value expects.
    fixed_fee_gap_points = (_rate("time-and-materials") - _rate("fixed-fee")) * 100.0
    loss_makers = facts[facts["margin"] < 0]
    small_loss_makers = loss_makers[loss_makers["size_tier"] == "small"]

    items = [
        {
            "title": "Recover realisation on the bottom quartile of engagements",
            "value": float(bundle.sensitivity.loc[
                bundle.sensitivity["points"] == 5.0, "incremental_margin"
            ].iloc[0]),
            "evidence": (
                f"{len(low_realization)} engagements carrying "
                f"${low_realization['standard_value'].sum() / 1e6:,.1f}M of rate-card value "
                f"realise at or below {low_realization['realization_rate'].max():.0%}. "
                "Every point recovered is billing that was already earned."
            ),
            "action": (
                "Put these engagements through a commercial review before the next "
                "invoice run: confirm scope against delivered hours, re-paper "
                "out-of-scope work as a change order, and submit the unbilled time "
                "that is still recoverable."
            ),
        },
        {
            "title": "Close the fixed-fee realisation gap",
            "value": uplift_value(fixed_fee, fixed_fee_gap_points),
            "assumption": (
                f"Assumes fixed-fee work realises at the "
                f"{_rate('time-and-materials'):.1%} the firm already achieves on "
                "time-and-materials, a "
                f"{fixed_fee_gap_points:.1f} point improvement."
            ),
            "evidence": (
                f"Fixed-fee work realises at {_rate('fixed-fee'):.1%} against "
                f"{_rate('time-and-materials'):.1%} for time-and-materials, and "
                f"overruns budget by {_overrun('fixed-fee'):.2f}x against "
                f"{_overrun('time-and-materials'):.2f}x. Unbilled time on fixed-fee "
                f"work is "
                f"{fixed_fee['unbilled_value'].sum() / fixed_fee['standard_value'].sum():.1%} "
                "of rate-card value, close to three times the time-and-materials rate."
            ),
            "action": (
                "Convert the overrun into a priced change-control step. A fixed fee "
                "should fix the scope, not the hours: require a signed change order "
                "once delivered hours pass budget, and price the mobilisation "
                "separately instead of absorbing it."
            ),
        },
        {
            "title": "Put a floor under small engagements",
            "value": float(-loss_makers["margin"].sum()),
            "evidence": (
                f"{len(loss_makers)} engagements destroyed "
                f"${-loss_makers['margin'].sum() / 1e3:,.0f}k of margin, and "
                f"{len(small_loss_makers)} of them are small. The median loss-maker "
                f"carries ${loss_makers['standard_value'].median() / 1e3:,.0f}k of "
                f"rate-card value against ${facts[facts['margin'] >= 0]['standard_value'].median() / 1e3:,.0f}k "
                "for engagements that make money: the fixed cost of mobilising a "
                "team does not scale down with the engagement."
            ),
            "action": (
                "Set a minimum engagement size by service line, or price a standing "
                "mobilisation fee that covers the first phase. Where a small "
                "engagement is strategic, book it as business development rather "
                "than as delivery so it stops distorting service-line margin."
            ),
        },
    ]
    return sorted(items, key=lambda item: item["value"], reverse=True)


def _usd(value: float, precision: int = 1) -> str:
    """Format a currency value compactly, in millions or thousands.

    Values below a million are shown in thousands, so a $480k figure does not
    round away to ``$0.5M``.

    Args:
        value: Amount in currency units.
        precision: Decimal places to show.

    Returns:
        A string such as ``$6.8M`` or ``$480k``.
    """
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    if magnitude and magnitude < 1e6:
        return f"{sign}${magnitude / 1e3:,.0f}k"
    return f"{sign}${magnitude / 1e6:,.{precision}f}M"


def _usd_plain(value: float) -> str:
    """Format a currency value with thousands separators and no decimals.

    Args:
        value: Amount in currency units.

    Returns:
        A string such as ``$6,807,463``.
    """
    return f"${value:,.0f}"


def _pct(value: float, precision: int = 1) -> str:
    """Format a fraction as a percentage.

    Args:
        value: Fraction to format.
        precision: Decimal places to show.

    Returns:
        A string such as ``84.4%``.
    """
    return f"{value:.{precision}%}"


def _data_uri(path: Path) -> str:
    """Encode a PNG as a base64 data URI so it renders without a network fetch.

    Args:
        path: Path to the PNG.

    Returns:
        A ``data:`` URI holding the encoded image.
    """
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _kpi(label: str, value: str, note: str = "") -> str:
    """Render one headline figure as markup.

    Args:
        label: Short metric name.
        value: Formatted metric value.
        note: Optional supporting line.

    Returns:
        An HTML fragment.
    """
    note_markup = f'<div class="kpi-note">{html.escape(note)}</div>' if note else ""
    return (
        '<div class="kpi">'
        f'<div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="kpi-value">{html.escape(value)}</div>'
        f"{note_markup}"
        "</div>"
    )


def _table(headers: list[str], rows: list[list[str]], align: list[str]) -> str:
    """Render a data table.

    Args:
        headers: Column headings.
        rows: Pre-formatted cell values.
        align: Per-column CSS text alignment.

    Returns:
        An HTML fragment.
    """
    head = "".join(
        f'<th style="text-align:{align[index]}">{html.escape(name)}</th>'
        for index, name in enumerate(headers)
    )
    body = "".join(
        "<tr>"
        + "".join(
            f'<td style="text-align:{align[index]}">{html.escape(cell)}</td>'
            for index, cell in enumerate(row)
        )
        + "</tr>"
        for row in rows
    )
    return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def _waterfall_table(frame: pd.DataFrame, rate_denominator: float) -> str:
    """Render the margin-leakage waterfall as a table.

    Args:
        frame: Frame produced by :func:`src.analysis.waterfall`.
        rate_denominator: Value the percentage column is measured against.

    Returns:
        An HTML fragment.
    """
    rows: list[list[str]] = []
    for record in frame.to_dict("records"):
        amount = float(record["amount"])
        is_opening = bool(record["is_opening"])
        rows.append(
            [
                str(record["label"]),
                _usd_plain(abs(amount)) if is_opening else f"({_usd_plain(abs(amount))})",
                "-" if is_opening else _pct(abs(amount) / rate_denominator),
                _usd_plain(float(record["running_total"])),
            ]
        )
    return _table(
        ["Waterfall step", "Amount", "Share of rate card", "Running total"],
        rows,
        ["left", "right", "right", "right"],
    )


def _annual_table(frame: pd.DataFrame) -> str:
    """Render the fiscal-year summary as a table.

    Args:
        frame: Frame produced by :func:`src.analysis.annual_summary`.

    Returns:
        An HTML fragment.
    """
    rows = [
        [
            str(int(record["year"])),
            f"{int(record['engagements']):,}",
            _usd(record["standard_value"]),
            _usd(record["billed_amount"]),
            _usd(record["collected_amount"]),
            _usd(record["margin"]),
            _pct(record["realization_rate"]),
            _pct(record["margin_rate"]),
        ]
        for record in frame.to_dict("records")
    ]
    return _table(
        ["Year", "Engagements", "Rate-card value", "Billed", "Collected", "Margin", "Realisation", "Margin rate"],
        rows,
        ["left"] + ["right"] * 7,
    )


def _client_table(frame: pd.DataFrame, limit: int = 12) -> str:
    """Render the tiered client table.

    Args:
        frame: Frame produced by :func:`src.analysis.client_profitability`.
        limit: Number of clients to show, plus any client losing money.

    Returns:
        An HTML fragment.
    """
    shown = frame.head(limit)
    remainder = frame.iloc[limit:]
    negative = remainder[remainder["margin"] < 0]
    selected = pd.concat([shown, negative]).drop_duplicates(subset=["client_id"])

    rows = [
        [
            str(int(record["rank"])),
            str(record["client_name"]),
            str(record["tier"]),
            str(record["industry"]),
            f"{int(record['engagements']):,}",
            _usd(record["collected_amount"]),
            _usd(record["margin"]),
            _pct(record["margin_rate"]),
            _pct(record["realization_rate"]),
        ]
        for record in selected.to_dict("records")
    ]
    return _table(
        ["#", "Client", "Tier", "Industry", "Engagements", "Collected", "Margin", "Margin rate", "Realisation"],
        rows,
        ["right", "left", "left", "left", "right", "right", "right", "right", "right"],
    )


def _dimension_table(frame: pd.DataFrame, dimension: str, label: str) -> str:
    """Render a grouped metrics table for one dimension.

    Args:
        frame: Frame produced by :func:`src.analysis.group_by`.
        dimension: Column the frame is grouped by.
        label: Heading for that column.

    Returns:
        An HTML fragment.
    """
    rows = [
        [
            str(record[dimension]),
            f"{int(record['engagements']):,}",
            _usd(record["standard_value"]),
            _pct(record["realization_rate"]),
            _pct(record["collection_rate"]),
            _pct(record["margin_rate"]),
            f"{record['overrun_ratio']:.2f}x",
        ]
        for record in frame.to_dict("records")
    ]
    return _table(
        [label, "Engagements", "Rate-card value", "Realisation", "Collection", "Margin rate", "Overrun"],
        rows,
        ["left"] + ["right"] * 6,
    )


def _figure(path: Path, caption: str, alt: str) -> str:
    """Render one inlined figure with a caption.

    Args:
        path: PNG to inline.
        caption: Caption shown under the figure.
        alt: Alternative text for the image.

    Returns:
        An HTML fragment.
    """
    return (
        "<figure>"
        f'<img src="{_data_uri(path)}" alt="{html.escape(alt)}">'
        f"<figcaption>{html.escape(caption)}</figcaption>"
        "</figure>"
    )


def _recommendation_card(index: int, item: dict[str, Any]) -> str:
    """Render one prioritised recommendation.

    Args:
        index: One-based position in the list.
        item: Recommendation dictionary from :func:`recommendations`.

    Returns:
        An HTML fragment.
    """
    assumption = item.get("assumption")
    assumption_markup = (
        f'<p class="rec-assumption"><strong>Assumption.</strong> {html.escape(assumption)}</p>'
        if assumption
        else ""
    )
    return (
        '<div class="recommendation">'
        f'<div class="rec-head"><span class="rec-rank">{index}</span>'
        f'<span class="rec-title">{html.escape(item["title"])}</span>'
        f'<span class="rec-value">{_usd(item["value"])}</span></div>'
        f'<p class="rec-evidence">{html.escape(item["evidence"])}</p>'
        f"{assumption_markup}"
        f'<p class="rec-action"><strong>Action.</strong> {html.escape(item["action"])}</p>'
        "</div>"
    )


def render_html(bundle: AnalysisBundle) -> str:
    """Render the complete report as a single HTML document.

    Args:
        bundle: Populated analysis bundle.

    Returns:
        The HTML document as a string, with every asset inlined.
    """
    headline = bundle.headline_summary
    portfolio = bundle.portfolio
    year = bundle.settings.headline_year
    generated = date.today().isoformat()

    rate_card = float(headline["standard_value"])
    leakage = float(
        headline["discount_amount"] + headline["unbilled_value"] + headline["write_off_amount"]
    )
    sensitivity_five = float(
        bundle.sensitivity.loc[bundle.sensitivity["points"] == 5.0, "incremental_margin"].iloc[0]
    )
    tier_a = bundle.tiers[bundle.tiers["tier"] == "A"].iloc[0]
    tier_c = bundle.tiers[bundle.tiers["tier"] == "C"].iloc[0]
    low_realization = bottom_quartile_engagements(bundle.facts)

    kpis = "".join(
        [
            _kpi(f"Billed in {year}", _usd(headline["billed_amount"]), f"{int(headline['engagements'])} engagements"),
            _kpi("Margin rate", _pct(headline["margin_rate"]), f"{_usd(headline['margin'])} of margin"),
            _kpi("Realisation rate", _pct(headline["realization_rate"]), "billed over rate-card value"),
            _kpi("Collection rate", _pct(headline["collection_rate"]), "collected over billed"),
            _kpi("Firm utilisation", _pct(bundle.utilization_firm), "billable over available hours"),
            _kpi("Revenue leakage", _usd(leakage), f"{_pct(leakage / rate_card)} of rate-card value"),
        ]
    )

    top_band = bundle.overrun["bands"].iloc[0]
    bottom_band = bundle.overrun["bands"].iloc[-1]
    decile_gap = float(bundle.deciles["median_margin_rate"].iloc[-1] - bundle.deciles["median_margin_rate"].iloc[0])

    body = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Engagement Profitability Report {year}</title>
<style>
:root {{
  --ink: #1f2d3d; --body: #35414f; --muted: #6b7787; --rule: #e2e6ea;
  --lost: #b3453a; --kept: #2e7d6f; --accent: #3b6ea5; --tint: #f6f8fa;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; padding: 48px 24px 80px; background: #fff; color: var(--body);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  font-size: 15px; line-height: 1.6;
}}
main {{ max-width: 940px; margin: 0 auto; }}
h1 {{ font-size: 32px; line-height: 1.2; color: var(--ink); margin: 0 0 6px; letter-spacing: -0.4px; }}
h2 {{ font-size: 21px; color: var(--ink); margin: 52px 0 6px; padding-top: 22px; border-top: 1px solid var(--rule); }}
h3 {{ font-size: 16px; color: var(--ink); margin: 28px 0 8px; }}
p {{ margin: 0 0 14px; }}
.meta {{ color: var(--muted); font-size: 13px; margin-bottom: 28px; }}
.lede {{ font-size: 17px; color: var(--ink); border-left: 3px solid var(--accent); padding-left: 16px; margin: 26px 0; }}
.question {{ font-weight: 600; color: var(--ink); }}
.kpis {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px; background: var(--rule); border: 1px solid var(--rule); margin: 26px 0; }}
.kpi {{ background: #fff; padding: 16px 18px; }}
.kpi-label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.6px; color: var(--muted); }}
.kpi-value {{ font-size: 26px; font-weight: 600; color: var(--ink); margin: 4px 0 2px; }}
.kpi-note {{ font-size: 12px; color: var(--muted); }}
table {{ width: 100%; border-collapse: collapse; margin: 14px 0 22px; font-size: 13.5px; }}
th {{ text-align: left; font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--muted); border-bottom: 1px solid var(--ink); padding: 8px 10px; font-weight: 600; }}
td {{ padding: 8px 10px; border-bottom: 1px solid var(--rule); }}
tbody tr:hover {{ background: var(--tint); }}
figure {{ margin: 26px 0 6px; }}
img {{ width: 100%; height: auto; border: 1px solid var(--rule); }}
figcaption {{ font-size: 13px; color: var(--muted); margin-top: 8px; }}
.recommendation {{ border: 1px solid var(--rule); border-left: 3px solid var(--accent); padding: 18px 20px; margin: 16px 0; background: var(--tint); }}
.rec-head {{ display: flex; align-items: baseline; gap: 12px; margin-bottom: 10px; }}
.rec-rank {{ display: inline-flex; align-items: center; justify-content: center; width: 24px; height: 24px; border-radius: 50%; background: var(--ink); color: #fff; font-size: 13px; font-weight: 600; flex: none; }}
.rec-title {{ font-size: 16px; font-weight: 600; color: var(--ink); flex: 1; }}
.rec-value {{ font-size: 16px; font-weight: 600; color: var(--kept); white-space: nowrap; }}
.rec-evidence {{ margin: 0 0 10px; font-size: 14px; }}
.rec-assumption {{ margin: 0 0 10px; font-size: 13.5px; color: var(--muted); }}
.rec-action {{ margin: 0; font-size: 14px; }}
.note {{ font-size: 13px; color: var(--muted); background: var(--tint); border: 1px solid var(--rule); padding: 14px 16px; }}
.data-note {{ border-left: 3px solid var(--lost); background: #fdf6f5; border: 1px solid #f0dcd9; padding: 14px 16px; font-size: 13.5px; }}
ul {{ margin: 0 0 14px; padding-left: 20px; }}
li {{ margin-bottom: 7px; }}
footer {{ margin-top: 56px; padding-top: 20px; border-top: 1px solid var(--rule); font-size: 13px; color: var(--muted); }}
@media (max-width: 720px) {{ .kpis {{ grid-template-columns: 1fr; }} }}
@media print {{ body {{ padding: 0; font-size: 12px; }} h2 {{ page-break-after: avoid; }} figure {{ page-break-inside: avoid; }} }}
</style>
</head>
<body>
<main>
<h1>Engagement Profitability Analytics</h1>
<div class="meta">Fiscal year {year} · three-year portfolio {int(bundle.annual['year'].min())}-{int(bundle.annual['year'].max())} · generated {generated}</div>

<p class="lede"><span class="question">We billed {_usd(headline['billed_amount'])} this year. Why is margin only {_pct(headline['margin_rate'])}?</span></p>

<p>The firm delivered {_usd(rate_card)} of work at its own rate card in {year} and collected
{_usd(headline['collected_amount'])}. The {_usd(leakage)} in between - {_pct(leakage / rate_card)} of everything the
firm was worth on paper - never became revenue. That gap, not the cost of the work, is what holds margin
at {_pct(headline['margin_rate'])}. It is recoverable without winning a single new engagement, and the
three actions at the end of this report are together worth more than the firm's entire margin
on its smallest forty clients.</p>

<div class="kpis">{kpis}</div>

<h2>Executive summary</h2>

<p>Three findings explain the {_pct(headline['margin_rate'])} margin, and each one points at a different
part of the business.</p>

<h3>1. The money is lost before delivery, not during it</h3>
<p>Discounts, unbilled time and write-offs cost {_usd(leakage)} in {year}. Discounts alone account for
{_usd(headline['discount_amount'])}, or {_pct(headline['discount_amount'] / rate_card)} of rate-card value, which
is a pricing decision taken at the start of an engagement. Unbilled time - work delivered and never
invoiced - costs a further {_usd(headline['unbilled_value'])}. It is the quietest of the three, because
nothing on the engagement looks broken: the work was done, the client is happy, and the hours simply
never reached an invoice.</p>

<h3>2. Overrun and margin decay move together</h3>
<p>Engagements that deliver closest to budget carry a {_pct(top_band['margin_rate'])} margin rate. Those
in the loosest overrun quintile, delivering {bottom_band['mean_overrun']:.2f}x their budgeted hours,
carry {_pct(bottom_band['margin_rate'])} - a {(top_band['margin_rate'] - bottom_band['margin_rate']) * 100:.1f}
point swing across the portfolio. The pattern is monotone in every band, and it is worst where the
contract type removes the firm's ability to bill the extra hours: fixed-fee work overruns by
{bundle.by_contract.loc[bundle.by_contract['contract_type'] == 'fixed-fee', 'overrun_ratio'].iloc[0]:.2f}x
against
{bundle.by_contract.loc[bundle.by_contract['contract_type'] == 'time-and-materials', 'overrun_ratio'].iloc[0]:.2f}x
for time-and-materials, and realises
{(bundle.by_contract.loc[bundle.by_contract['contract_type'] == 'time-and-materials', 'realization_rate'].iloc[0] - bundle.by_contract.loc[bundle.by_contract['contract_type'] == 'fixed-fee', 'realization_rate'].iloc[0]) * 100:.1f}
points less.</p>

<h3>3. Margin is concentrated in a few clients, and destroyed in small engagements</h3>
<p>{int(tier_a['clients'])} tier A clients - {int(tier_a['clients']) / len(bundle.clients):.0%} of the client
base - carry {_pct(tier_a['margin_share'])} of firm margin. The {int(tier_c['clients'])} tier C clients carry
{_pct(tier_c['margin_share'])} of margin on {_pct(tier_c['revenue_share'])} of revenue, at a margin rate of
{_pct(tier_c['margin_rate'])} against {_pct(tier_a['margin_rate'])} for tier A. At the engagement level,
{int(portfolio['loss_making_engagements'])} engagements destroyed {_usd(-portfolio['loss_making_margin'])} of
margin outright, and the great majority of them are small: the median loss-maker carries
{_usd(bundle.facts[bundle.facts['margin'] < 0]['standard_value'].median() / 1e6, 2)} of rate-card value
against {_usd(bundle.facts[bundle.facts['margin'] >= 0]['standard_value'].median() / 1e6, 2)} for engagements
that make money. Small engagements cannot carry the fixed cost of mobilising a team.</p>

<h2>Why margin is {_pct(headline['margin_rate'])}</h2>
<p>The waterfall below starts at the rate-card value of everything delivered in {year} and subtracts each
source of loss in turn. Red steps are value that was earned and then lost commercially. Grey steps are
the genuine cost of doing the work, which no commercial action removes.</p>
{_figure(bundle.figures["waterfall"], f"Margin leakage, FY{year}. Each step is a deduction from rate-card value.", "Waterfall chart from rate-card value through discounts, unbilled time, write-offs and delivery cost to actual margin.")}

{_waterfall_table(bundle.waterfall_headline, rate_card)}

<p>Two numbers deserve separating. The commercial leakage - discounts, unbilled time and write-offs - is
{_usd(leakage)}, or {_pct(leakage / rate_card)} of rate-card value. It is the addressable part of the gap.
The remaining {_usd(headline['consultant_cost'] + headline['direct_delivery_cost'])} is what it costs to
deliver, and it is not a failure of commercial discipline.</p>

<h2>Realisation is the lever</h2>
<p>Realisation - billed value over rate-card value - is the single best predictor of whether an engagement
makes money. Across the portfolio the relationship is monotone: median margin rate is
{decile_gap * 100:.1f} points higher in the strongest realisation decile than in the weakest. It is not a
straight line, which is why the chart below plots decile medians rather than a fitted trend.</p>
{_figure(bundle.figures["scatter"], "Every engagement by realisation and margin rate, with decile medians overlaid.", "Scatter plot of engagement realisation against margin rate with a decile median line.")}
{_figure(bundle.figures["overrun"], "Margin rate by delivery overrun quintile.", "Bar chart of margin rate across five overrun quintiles, declining from 24 percent to 14 percent.")}

<p>Delivery overrun is the mechanism behind the decay. The correlation between the overrun ratio and the
margin rate across {len(bundle.facts):,} engagements is
{bundle.overrun['correlation']:.2f}, and each additional unit of overrun costs
{abs(bundle.overrun['slope']) * 100:.1f} points of margin rate. Because the hours have already been
worked, every point of overrun that can be billed or avoided falls almost entirely to margin.</p>

<h2>Client concentration</h2>
{_figure(bundle.figures["pareto"], "Clients ranked by lifetime margin, with cumulative share.", "Pareto chart of client margin with a cumulative share curve.")}

{_table(
    ["Tier", "Clients", "Revenue", "Revenue share", "Margin", "Margin share", "Margin rate"],
    [[str(r["tier"]), f"{int(r['clients'])}", _usd(r["revenues"]), _pct(r["revenue_share"]), _usd(r["margin"]), _pct(r["margin_share"]), _pct(r["margin_rate"])] for r in bundle.tiers.to_dict("records")],
    ["left", "right", "right", "right", "right", "right", "right"],
)}

<p>Concentration is not automatically a problem - large clients are usually profitable clients. The
difficulty is tier C: {int(tier_c['clients'])} clients generating {_pct(tier_c['revenue_share'])} of revenue
at {_pct(tier_c['margin_rate'])} margin, which is
{(tier_a['margin_rate'] - tier_c['margin_rate']) * 100:.1f} points below tier A. Those accounts consume
delivery capacity that the tier A relationships could absorb.</p>

<h3>Client detail</h3>
{_client_table(bundle.clients)}

<h2>Utilisation</h2>
{_figure(bundle.figures["utilization"], "Billable hours as a share of available hours, by grade.", "Bar chart of utilisation by grade: consultant 86 percent, analyst 79 percent, manager 72 percent, partner 33 percent.")}

<p>Firm utilisation is {_pct(bundle.utilization_firm)}, but the average hides the shape of the pyramid.
Consultants run at {_pct(bundle.utilization.loc[bundle.utilization['level'] == 'Consultant', 'utilization'].iloc[0])}
and partners at {_pct(bundle.utilization.loc[bundle.utilization['level'] == 'Partner', 'utilization'].iloc[0])}.
A partner's time is spent selling and supervising, so that gap is partly structural. It becomes a problem
only when the firm sells work faster than the senior bench can support it, which is precisely what an
overrun looks like from the delivery side.</p>

<h2>Service line and contract type</h2>
{_dimension_table(bundle.by_service_line, "service_line", "Service line")}
{_dimension_table(bundle.by_contract, "contract_type", "Contract type")}

<p>Realisation is a property of the engagement, not of the grade that delivered it: allocating each
engagement's billing across grades in proportion to the rate-card value each contributed puts every grade
between {_pct(bundle.by_level['realization_rate'].min())} and {_pct(bundle.by_level['realization_rate'].max())}.
Leakage is therefore a commercial and contractual problem, not a delivery one - no grade is systematically
giving margin away.</p>

<h2>Three years in context</h2>
{_annual_table(bundle.annual)}

<p class="note">Money is allocated across calendar years in proportion to the hours worked in each, so an
engagement spanning two years contributes to both. The sum of the year rows equals the engagement's
own figures exactly. Realisation has been broadly stable; the fall in margin rate in {year} comes from
the delivery-cost and leakage mix, not from pricing.</p>

<h2>What closing the gap is worth</h2>
<p>Recovering {SENSITIVITY_POINTS[1]:.0f} points of realisation on the
{len(low_realization)} worst-realising engagements - those carrying
{_usd(low_realization['standard_value'].sum())} of rate-card value at a
{_pct(low_realization['collected_amount'].sum() / low_realization['billed_amount'].sum())} collection rate -
is worth {_usd(sensitivity_five)} of margin. That is
{sensitivity_five / headline['margin']:.0%} of everything the firm earned in {year}, from work it has
already delivered.</p>

{_table(
    ["Realisation recovered", "Engagements", "Rate-card value", "Collection rate", "Incremental billing", "Incremental margin"],
    [[f"{r['points']:.0f} points", f"{int(r['engagements'])}", _usd(r["standard_value"]), _pct(r["collection_rate"]), _usd(r["incremental_billing"]), _usd(r["incremental_margin"])] for r in bundle.sensitivity.to_dict("records")],
    ["left", "right", "right", "right", "right", "right"],
)}

<p class="note">The estimate assumes recovered billing is collected at the group's existing collection
rate and carries no incremental delivery cost, because the hours have already been worked. It does not
assume the firm wins any new work.</p>

<h2>Recommendations</h2>
<p>Prioritised by value. The three levers act on overlapping engagements - a small fixed-fee engagement
with weak realisation is counted by all three - so the figures are individually defensible but should not
be added together.</p>

{"".join(_recommendation_card(index, item) for index, item in enumerate(recommendations(bundle), start=1))}

<h2>Method and limitations</h2>
<div class="data-note"><strong>The data in this report is synthetic.</strong> It was generated by
<code>src/generate_data.py</code> from a fixed seed of {bundle.settings.random_seed} and contains no real
clients, consultants or engagements. The structure is modelled on how consulting firms actually behave -
fixed-fee work overruns, some service lines carry structurally lower realisation, small engagements
struggle to cover mobilisation cost - but the magnitudes are assumptions, not observations. Every
assumption is listed in the generator's module docstring, and the full metric definitions are in
<code>docs/methodology.md</code>.</div>

<ul>
<li><strong>Realisation and collection are different failures.</strong> Realisation is lost before the
invoice exists; collection is lost after it. The firm loses {_pct(1 - headline['realization_rate'])} to the
first and {_pct(1 - headline['collection_rate'])} to the second, and they need different remedies.</li>
<li><strong>Assignment by grade is an allocation.</strong> Billing is recorded per engagement, so the
by-grade view spreads each engagement's billing in proportion to the rate-card value each grade
contributed. It shows which grades staff the engagements where realisation is won or lost; it does not
show that any grade prices differently.</li>
<li><strong>Money is allocated across years by hours.</strong> Percentage-of-completion is a reasonable
approximation for a portfolio of this shape, but it is an approximation - an engagement's cash may not
follow its hours within a year.</li>
<li><strong>The sensitivity is a static estimate.</strong> It holds collection rates and delivery costs
fixed. In practice, pushing realisation harder on a client can slow collection, which is the trade the
estimate does not price.</li>
</ul>

<footer>Engagement Profitability Analytics · synthetic data, seed {bundle.settings.random_seed} ·
generated {generated}</footer>
</main>
</body>
</html>
"""
    return body


def write_metrics_json(bundle: AnalysisBundle) -> Path:
    """Write the headline metrics to ``reports/metrics.json``.

    Args:
        bundle: Populated analysis bundle.

    Returns:
        The path written.
    """
    metrics: dict[str, Any] = {
        "generated": date.today().isoformat(),
        "headline_year": bundle.settings.headline_year,
        "random_seed": bundle.settings.random_seed,
        "synthetic_data": True,
        "headline": bundle.headline_summary,
        "portfolio": bundle.portfolio,
        "utilization_firm": bundle.utilization_firm,
        "waterfall_headline": bundle.waterfall_headline.to_dict("records"),
        "annual": bundle.annual.to_dict("records"),
        "sensitivity": bundle.sensitivity.to_dict("records"),
        "client_tiers": bundle.tiers.to_dict("records"),
        "overrun": {
            "correlation": bundle.overrun["correlation"],
            "slope": bundle.overrun["slope"],
        },
        "recommendations": [
            {"title": item["title"], "value": item["value"]}
            for item in recommendations(bundle)
        ],
    }
    path = bundle.settings.metrics_path
    path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote metrics to {}", path)
    return path


def write_tables(bundle: AnalysisBundle) -> list[Path]:
    """Write the analysis tables to ``reports/tables`` as CSV.

    Args:
        bundle: Populated analysis bundle.

    Returns:
        The paths written, in the order they were written.
    """
    tables_dir = bundle.settings.tables_dir
    tables_dir.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {
        "annual_summary": bundle.annual,
        "margin_leakage_waterfall": bundle.waterfall_headline,
        "margin_by_service_line": bundle.by_service_line,
        "margin_by_contract_type": bundle.by_contract,
        "realization_by_level": bundle.by_level,
        "utilization_by_level": bundle.utilization,
        "client_profitability": bundle.clients,
        "client_tier_summary": bundle.tiers,
        "margin_decay_by_overrun": bundle.overrun["bands"],
        "realization_deciles": bundle.deciles,
        "realization_sensitivity": bundle.sensitivity,
    }
    written: list[Path] = []
    for name, frame in frames.items():
        path = tables_dir / f"{name}.csv"
        export = frame.copy()
        for column in export.columns:
            if column in MONEY_COLUMNS:
                export[column] = export[column].round(2)
        export.to_csv(path, index=False)
        written.append(path)
    logger.info("Wrote {} tables to {}", len(written), tables_dir)
    return written


def write_report(bundle: AnalysisBundle) -> Path:
    """Write the self-contained HTML report.

    Args:
        bundle: Populated analysis bundle.

    Returns:
        The path written.
    """
    path = (
        bundle.settings.reports_dir
        / f"engagement_profitability_report_{date.today():%Y%m%d}.html"
    )
    path.write_text(render_html(bundle), encoding="utf-8")
    logger.info("Wrote report to {}", path)
    return path


def run_analysis(settings: Settings | None = None) -> AnalysisBundle:
    """Build the bundle, then write the metrics, tables and figures.

    Args:
        settings: Optional settings override; defaults to the process singleton.

    Returns:
        The populated analysis bundle that was written.
    """
    bundle = build_bundle(settings)
    write_metrics_json(bundle)
    write_tables(bundle)
    logger.info(
        "Analysis complete: {} {} margin at {}",
        bundle.settings.headline_year,
        _usd(bundle.headline_summary["billed_amount"]),
        _pct(bundle.headline_summary["margin_rate"]),
    )
    return bundle


def run_report(settings: Settings | None = None) -> Path:
    """Build the bundle and write the HTML report.

    Args:
        settings: Optional settings override; defaults to the process singleton.

    Returns:
        The report path.
    """
    return write_report(build_bundle(settings))
