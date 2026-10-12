"""Streamlit front end for the Engagement Profitability analysis.

The application is deliberately thin. Every figure it shows is produced by
:mod:`src.analysis`; the app filters rows and formats the result. Nothing is
recomputed here, so the numbers on screen and the numbers in the HTML report
cannot disagree.

Run it with::

    python cli.py app
    # or
    streamlit run app/app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

# Streamlit runs this file as a script, so the project root is not on the import
# path by default. Insert it before importing anything from ``src``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import (  # noqa: E402
    build_facts,
    client_profitability,
    firm_summary,
    firm_utilization,
    group_by,
    load_raw,
    overrun_vs_margin,
    realization_decile_profile,
    sensitivity_realization,
    tier_summary,
    utilization_by_consultant,
    utilization_by_level,
    waterfall,
)
from src.config import get_settings  # noqa: E402
from src.plots import (  # noqa: E402
    plot_client_pareto,
    plot_margin_waterfall,
    plot_overrun_bands,
    plot_realization_vs_margin,
    plot_utilization_by_level,
)

#: Scratch directory for figures the application renders on the fly. The report
#: owns ``reports/figures``; the app keeps its own output out of the way.
FIGURE_DIR = Path(tempfile.gettempdir()) / "engagement-profitability-app"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Engagement Profitability Analytics",
    page_icon="chart_with_downwards_trend",
    layout="wide",
)

PALETTE = {
    "lost": "#b3453a",
    "kept": "#2e7d6f",
    "muted": "#6b7787",
}


def _usd(value: float) -> str:
    """Format a currency value compactly.

    Args:
        value: Amount in currency units.

    Returns:
        A string such as ``$37.8M`` or ``$480k``.
    """
    magnitude = abs(value)
    sign = "-" if value < 0 else ""
    if magnitude and magnitude < 1e6:
        return f"{sign}${magnitude / 1e3:,.0f}k"
    return f"{sign}${magnitude / 1e6:,.1f}M"


def _pct(value: float) -> str:
    """Format a fraction as a percentage.

    Args:
        value: Fraction to format.

    Returns:
        A string such as ``84.4%``.
    """
    return f"{value:.1%}"


@st.cache_data(show_spinner=False)
def _load() -> dict[str, pd.DataFrame]:
    """Load the generated tables.

    Returns:
        Mapping of table name to DataFrame.
    """
    return load_raw(get_settings())


@st.cache_data(show_spinner=False)
def _facts() -> pd.DataFrame:
    """Return the engagement-level fact table.

    Returns:
        The fact table.
    """
    return build_facts(get_settings())


def _sidebar(facts: pd.DataFrame) -> pd.DataFrame:
    """Render the sidebar filters and apply them to the fact table.

    Args:
        facts: Unfiltered engagement-level fact table.

    Returns:
        The filtered fact table, with a note in the sidebar showing what is left.
    """
    st.sidebar.title("Filters")
    st.sidebar.caption(
        "Filters apply to the engagement portfolio across the whole three-year "
        "window. Clear them to return to the firm-wide view."
    )

    service_lines = st.sidebar.multiselect(
        "Service line",
        sorted(facts["service_line"].unique()),
        default=sorted(facts["service_line"].unique()),
    )
    contract_types = st.sidebar.multiselect(
        "Contract type",
        sorted(facts["contract_type"].unique()),
        default=sorted(facts["contract_type"].unique()),
    )
    tiers = st.sidebar.multiselect(
        "Client tier",
        ["A", "B", "C"],
        default=["A", "B", "C"],
    )

    start = pd.to_datetime(facts["start_date"]).min().date()
    end = pd.to_datetime(facts["end_date"]).max().date()
    date_range = st.sidebar.date_input(
        "Engagements starting between",
        value=(start, end),
        min_value=start,
        max_value=end,
    )

    tier_lookup = client_profitability(facts).set_index("client_id")["tier"]
    filtered = facts.assign(
        client_tier=facts["client_id"].map(tier_lookup)
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        low, high = date_range
        starts = pd.to_datetime(filtered["start_date"]).dt.date
        filtered = filtered[(starts >= low) & (starts <= high)]

    filtered = filtered[
        filtered["service_line"].isin(service_lines)
        & filtered["contract_type"].isin(contract_types)
        & filtered["client_tier"].isin(tiers)
    ]

    st.sidebar.divider()
    st.sidebar.metric("Engagements in view", f"{len(filtered):,}")
    if not filtered.empty:
        st.sidebar.metric("Billed", _usd(filtered["billed_amount"].sum()))
        st.sidebar.metric(
            "Margin rate",
            _pct(filtered["margin"].sum() / filtered["collected_amount"].sum()),
        )
    if filtered.empty:
        st.sidebar.warning("No engagements match these filters.")
    return filtered


def _executive_tab(facts: pd.DataFrame, utilization: pd.DataFrame) -> None:
    """Render the executive summary tab.

    Args:
        facts: Filtered engagement-level fact table.
        utilization: Frame produced by
            :func:`src.analysis.utilization_by_level`.

    Returns:
        None.
    """
    summary = firm_summary(facts)
    leakage = (
        summary["discount_amount"] + summary["unbilled_value"] + summary["write_off_amount"]
    )

    columns = st.columns(4)
    columns[0].metric("Standard value", _usd(summary["standard_value"]))
    columns[1].metric("Billed", _usd(summary["billed_amount"]))
    columns[2].metric("Collected", _usd(summary["collected_amount"]))
    columns[3].metric("Margin", _usd(summary["margin"]))

    columns = st.columns(4)
    columns[0].metric("Realisation rate", _pct(summary["realization_rate"]))
    columns[1].metric("Collection rate", _pct(summary["collection_rate"]))
    columns[2].metric("Margin rate", _pct(summary["margin_rate"]))
    columns[3].metric(
        "Firm utilisation",
        _pct(firm_utilization(utilization)),
    )

    st.markdown(
        f"""
**Reading the numbers.** The engagements in view delivered
{_usd(summary['standard_value'])} of work at the firm's rate card and collected
{_usd(summary['collected_amount'])}. The {_usd(leakage)} in between is commercial
leakage: {_usd(summary['discount_amount'])} of discounts,
{_usd(summary['unbilled_value'])} of time never invoiced and
{_usd(summary['write_off_amount'])} written off after invoicing.
{int(summary['loss_making_engagements'])} engagements lost money outright, costing
{_usd(-summary['loss_making_margin'])}.
"""
    )

    st.subheader("Engagement detail")
    detail = facts[
        [
            "engagement_id",
            "client_name",
            "service_line",
            "contract_type",
            "standard_value",
            "billed_amount",
            "collected_amount",
            "margin",
            "realization_rate",
            "collection_rate",
            "margin_rate",
            "overrun_ratio",
        ]
    ].sort_values("margin", ascending=False)
    st.dataframe(
        detail,
        width="stretch",
        hide_index=True,
        column_config={
            "standard_value": st.column_config.NumberColumn("Rate-card value", format="$%,.0f"),
            "billed_amount": st.column_config.NumberColumn("Billed", format="$%,.0f"),
            "collected_amount": st.column_config.NumberColumn("Collected", format="$%,.0f"),
            "margin": st.column_config.NumberColumn("Margin", format="$%,.0f"),
            "realization_rate": st.column_config.NumberColumn("Realisation", format="%.1f%%"),
            "collection_rate": st.column_config.NumberColumn("Collection", format="%.1f%%"),
            "margin_rate": st.column_config.NumberColumn("Margin rate", format="%.1f%%"),
            "overrun_ratio": st.column_config.NumberColumn("Overrun", format="%.2fx"),
        },
    )


def _leakage_tab(facts: pd.DataFrame) -> None:
    """Render the margin-leakage tab.

    Args:
        facts: Filtered engagement-level fact table.

    Returns:
        None.
    """
    if facts.empty:
        st.info("No engagements match the current filters.")
        return

    standard_value = float(facts["standard_value"].sum())
    figure = plot_margin_waterfall(
        waterfall(facts),
        FIGURE_DIR / "app_waterfall.png",
        "current selection",
    )
    st.image(str(figure), width="stretch")

    st.subheader("Waterfall detail")
    frame = waterfall(facts).copy()
    frame["share_of_rate_card"] = frame["amount"].abs() / standard_value
    st.dataframe(
        frame,
        width="stretch",
        hide_index=True,
        column_config={
            "label": "Step",
            "amount": st.column_config.NumberColumn("Amount", format="$%,.0f"),
            "share_of_rate_card": st.column_config.NumberColumn(
                "Share of rate card", format="%.1f%%"
            ),
            "running_total": st.column_config.NumberColumn("Running total", format="$%,.0f"),
            "step": None,
            "is_opening": None,
        },
    )

    left, right = st.columns(2)
    with left:
        st.subheader("By service line")
        st.dataframe(
            group_by(facts, "service_line")[
                ["service_line", "engagements", "collected_amount", "realization_rate", "margin_rate"]
            ],
            width="stretch",
            hide_index=True,
            column_config={
                "service_line": "Service line",
                "collected_amount": st.column_config.NumberColumn("Collected", format="$%,.0f"),
                "realization_rate": st.column_config.NumberColumn("Realisation", format="%.1f%%"),
                "margin_rate": st.column_config.NumberColumn("Margin rate", format="%.1f%%"),
            },
        )
    with right:
        st.subheader("By contract type")
        st.dataframe(
            group_by(facts, "contract_type")[
                ["contract_type", "engagements", "collected_amount", "realization_rate", "margin_rate"]
            ],
            width="stretch",
            hide_index=True,
            column_config={
                "contract_type": "Contract type",
                "collected_amount": st.column_config.NumberColumn("Collected", format="$%,.0f"),
                "realization_rate": st.column_config.NumberColumn("Realisation", format="%.1f%%"),
                "margin_rate": st.column_config.NumberColumn("Margin rate", format="%.1f%%"),
            },
        )

    if len(facts) >= 20:
        st.subheader("Margin decay with overrun")
        bands = overrun_vs_margin(facts)["bands"]
        st.image(
            str(plot_overrun_bands(bands, FIGURE_DIR / "app_overrun.png")),
            width="stretch",
        )

        st.subheader("Realisation and margin")
        st.image(
            str(
                plot_realization_vs_margin(
                    facts,
                    realization_decile_profile(facts),
                    FIGURE_DIR / "app_scatter.png",
                )
            ),
            width="stretch",
        )

    st.subheader("What recovering realisation is worth")
    st.caption(
        "Incremental margin from lifting realisation on the worst-realising "
        "quarter of the engagements in view. The hours are already worked, so the "
        "recovered billing carries no incremental delivery cost."
    )
    st.dataframe(
        sensitivity_realization(facts),
        width="stretch",
        hide_index=True,
        column_config={
            "points": st.column_config.NumberColumn("Points recovered", format="%.0f"),
            "engagements": "Engagements",
            "standard_value": st.column_config.NumberColumn("Rate-card value", format="$%,.0f"),
            "collection_rate": st.column_config.NumberColumn("Collection rate", format="%.1f%%"),
            "incremental_billing": st.column_config.NumberColumn("Incremental billing", format="$%,.0f"),
            "incremental_margin": st.column_config.NumberColumn("Incremental margin", format="$%,.0f"),
        },
    )


def _clients_tab(facts: pd.DataFrame) -> None:
    """Render the client profitability tab.

    Args:
        facts: Filtered engagement-level fact table.

    Returns:
        None.
    """
    if facts.empty:
        st.info("No engagements match the current filters.")
        return

    clients = client_profitability(facts)
    tiers = tier_summary(clients)

    st.subheader("Tier summary")
    st.dataframe(
        tiers,
        width="stretch",
        hide_index=True,
        column_config={
            "tier": "Tier",
            "clients": "Clients",
            "revenues": st.column_config.NumberColumn("Revenue", format="$%,.0f"),
            "margin": st.column_config.NumberColumn("Margin", format="$%,.0f"),
            "margin_rate": st.column_config.NumberColumn("Margin rate", format="%.1f%%"),
            "revenue_share": st.column_config.NumberColumn("Revenue share", format="%.1f%%"),
            "margin_share": st.column_config.NumberColumn("Margin share", format="%.1f%%"),
        },
    )

    st.image(
        str(
            plot_client_pareto(
                clients, FIGURE_DIR / "app_pareto.png"
            )
        ),
        width="stretch",
    )

    st.subheader("Client detail")
    st.dataframe(
        clients[
            [
                "rank",
                "client_name",
                "tier",
                "industry",
                "engagements",
                "collected_amount",
                "margin",
                "margin_rate",
                "realization_rate",
            ]
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "rank": "#",
            "client_name": "Client",
            "tier": "Tier",
            "collected_amount": st.column_config.NumberColumn("Collected", format="$%,.0f"),
            "margin": st.column_config.NumberColumn("Margin", format="$%,.0f"),
            "margin_rate": st.column_config.NumberColumn("Margin rate", format="%.1f%%"),
            "realization_rate": st.column_config.NumberColumn("Realisation", format="%.1f%%"),
        },
    )


def _utilization_tab(raw: dict[str, pd.DataFrame], facts: pd.DataFrame) -> None:
    """Render the utilisation tab.

    Args:
        raw: Generated tables, used for timesheets and the consultant roster.
        facts: Filtered engagement-level fact table.

    Returns:
        None.
    """
    timesheets = raw["timesheets"]
    if not facts.empty:
        timesheets = timesheets[timesheets["engagement_id"].isin(facts["engagement_id"])]
    utilization = utilization_by_consultant(timesheets, raw["consultants"])

    if timesheets.empty:
        st.info("No timesheet rows match the current filters.")
        return

    by_level = utilization_by_level(utilization)
    st.image(
        str(
            plot_utilization_by_level(
                by_level,
                FIGURE_DIR / "app_utilization.png",
                firm_utilization(utilization),
            )
        ),
        width="stretch",
    )

    st.dataframe(
        by_level,
        width="stretch",
        hide_index=True,
        column_config={
            "level": "Grade",
            "consultants": "People",
            "billable_hours": st.column_config.NumberColumn("Billable hours", format="%,.0f"),
            "available_hours": st.column_config.NumberColumn("Available hours", format="%,.0f"),
            "utilization": st.column_config.NumberColumn("Utilisation", format="%.1f%%"),
        },
    )

    st.subheader("Consultant detail")
    st.dataframe(
        utilization[
            ["consultant_id", "consultant_name", "level", "service_line", "billable_hours", "available_hours", "utilization"]
        ].sort_values("utilization", ascending=False),
        width="stretch",
        hide_index=True,
        column_config={
            "consultant_id": "ID",
            "consultant_name": "Consultant",
            "level": "Grade",
            "service_line": "Service line",
            "billable_hours": st.column_config.NumberColumn("Billable hours", format="%,.0f"),
            "available_hours": st.column_config.NumberColumn("Available hours", format="%,.0f"),
            "utilization": st.column_config.NumberColumn("Utilisation", format="%.1f%%"),
        },
    )


def main() -> None:
    """Render the application.

    Returns:
        None.
    """
    st.title("Engagement Profitability Analytics")
    st.caption(
        "Synthetic data, seed 42. Every figure is computed by src/analysis.py; the "
        "application only filters and formats."
    )

    try:
        raw = _load()
        facts = _facts()
    except FileNotFoundError as error:
        st.error(str(error))
        st.code("python cli.py generate-data", language="bash")
        return

    filtered = _sidebar(facts)
    utilization = utilization_by_level(
        utilization_by_consultant(raw["timesheets"], raw["consultants"])
    )

    tabs = st.tabs(
        ["Executive summary", "Margin leakage", "Client profitability", "Utilisation"]
    )
    with tabs[0]:
        _executive_tab(filtered, utilization)
    with tabs[1]:
        _leakage_tab(filtered)
    with tabs[2]:
        _clients_tab(filtered)
    with tabs[3]:
        _utilization_tab(raw, filtered)

    with st.expander("How these metrics are defined"):
        st.markdown(
            """
- **Standard value** - delivered hours valued at the rate card. What the work
  was worth before any commercial conversation.
- **Realisation rate** - billed over standard value. How much of the rate card
  survived pricing and delivery.
- **Collection rate** - collected over billed. How much of the invoice survived
  the client's payment process.
- **Margin rate** - margin over collections. Margin is always measured against
  net revenue, never against the rate card.
- **Utilisation** - billable hours over available hours, recomputed from summed
  hours so part-year joiners do not distort the average.

Full definitions, and the assumptions behind the synthetic data, are in
`docs/methodology.md`.
"""
        )


if __name__ == "__main__":
    main()
