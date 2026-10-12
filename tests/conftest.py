"""Shared fixtures for the test suite.

The fixtures here are small tables built by hand. Every expected value in the
tests is worked out on paper from these numbers rather than taken from a run of
the code under test, so the tests would still fail if a formula quietly changed
in a way that happened to stay self-consistent.

Two kinds of fixture are provided:

``facts`` and friends
    Hand-built frames with round numbers, used to check the arithmetic.
``generated``
    The real generator output, built once per session, used to check the
    invariants that must hold across the whole synthetic dataset.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def facts() -> pd.DataFrame:
    """Return a two-engagement fact table with hand-computed totals.

    Both engagements are internally consistent, so the waterfall identity holds
    by construction:

    * E1 - 1,000 rate card, 100 discount, 50 unbilled, 30 written off,
      80 delivery cost, 400 consultant cost, 340 margin.
    * E2 - 500 rate card, no discount, 20 unbilled, 10 written off,
      40 delivery cost, 200 consultant cost, 230 margin.

    Firm totals: 1,500 rate card, 1,330 billed, 1,290 collected, 570 margin.

    Returns:
        The fact table.
    """
    records: list[dict[str, Any]] = [
        {
            "engagement_id": "E1",
            "client_id": "C1",
            "client_name": "Alpha",
            "industry": "Energy",
            "service_line": "Strategy",
            "contract_type": "fixed-fee",
            "size_tier": "small",
            "status": "Completed",
            "budgeted_hours": 100.0,
            "delivered_hours": 110.0,
            "standard_value": 1000.0,
            "discount_amount": 100.0,
            "unbilled_value": 50.0,
            "billed_amount": 850.0,
            "write_off_amount": 30.0,
            "collected_amount": 820.0,
            "direct_delivery_cost": 80.0,
            "consultant_cost": 400.0,
            "margin": 340.0,
        },
        {
            "engagement_id": "E2",
            "client_id": "C2",
            "client_name": "Beta",
            "industry": "Retail",
            "service_line": "Digital",
            "contract_type": "time-and-materials",
            "size_tier": "small",
            "status": "Completed",
            "budgeted_hours": 50.0,
            "delivered_hours": 45.0,
            "standard_value": 500.0,
            "discount_amount": 0.0,
            "unbilled_value": 20.0,
            "billed_amount": 480.0,
            "write_off_amount": 10.0,
            "collected_amount": 470.0,
            "direct_delivery_cost": 40.0,
            "consultant_cost": 200.0,
            "margin": 230.0,
        },
    ]
    frame = pd.DataFrame.from_records(records)
    frame["realization_rate"] = frame["billed_amount"] / frame["standard_value"]
    frame["collection_rate"] = frame["collected_amount"] / frame["billed_amount"]
    frame["margin_rate"] = frame["margin"] / frame["collected_amount"]
    frame["overrun_ratio"] = frame["delivered_hours"] / frame["budgeted_hours"]
    return frame


@pytest.fixture(scope="session")
def consultants() -> pd.DataFrame:
    """Return a three-person roster carrying only what the firm knows.

    Availability is a property of the roster; billable hours are not. They are
    supplied per test as a timesheet, because that is how the analysis receives
    them:

    * C1 - 1,000 available.
    * C2 - 500 available.
    * C3 - 2,000 available.

    Firm availability is 3,500 hours. The tests book 800, 300 and 1,900 hours
    respectively, giving 80%, 60% and 95% and a firm rate of 3,000/3,500 = 6/7.

    Returns:
        The consultant roster.
    """
    return pd.DataFrame.from_records(
        [
            {
                "consultant_id": "C1",
                "consultant_name": "One",
                "level": "Partner",
                "service_line": "Strategy",
                "standard_rate": 600.0,
                "cost_rate": 300.0,
                "available_hours": 1000.0,
            },
            {
                "consultant_id": "C2",
                "consultant_name": "Two",
                "level": "Consultant",
                "service_line": "Strategy",
                "standard_rate": 300.0,
                "cost_rate": 150.0,
                "available_hours": 500.0,
            },
            {
                "consultant_id": "C3",
                "consultant_name": "Three",
                "level": "Consultant",
                "service_line": "Digital",
                "standard_rate": 300.0,
                "cost_rate": 150.0,
                "available_hours": 2000.0,
            },
        ]
    )


@pytest.fixture(scope="session")
def sensitivity_facts() -> pd.DataFrame:
    """Return four engagements spanning the realisation range.

    Realisation rates are 0.50, 0.90, 0.95 and 0.98. The 25th percentile of that
    set, by linear interpolation, is 0.80 - so the bottom quartile is exactly one
    engagement, E1, with 1,000 of rate-card value, 500 billed, 400 collected and
    therefore a collection rate of 0.8.

    Returns:
        The fact table.
    """
    records: list[dict[str, Any]] = []
    for identifier, billed, collected in (
        ("E1", 500.0, 400.0),
        ("E2", 900.0, 800.0),
        ("E3", 950.0, 900.0),
        ("E4", 980.0, 900.0),
    ):
        records.append(
            {
                "engagement_id": identifier,
                "client_id": f"C{identifier[-1]}",
                "client_name": f"Client {identifier[-1]}",
                "industry": "Energy",
                "service_line": "Strategy",
                "contract_type": "fixed-fee",
                "size_tier": "small",
                "status": "Completed",
                "budgeted_hours": 100.0,
                "delivered_hours": 100.0,
                "standard_value": 1000.0,
                "discount_amount": 0.0,
                "unbilled_value": 0.0,
                "billed_amount": billed,
                "write_off_amount": 0.0,
                "collected_amount": collected,
                "direct_delivery_cost": 0.0,
                "consultant_cost": 0.0,
                "margin": collected,
            }
        )
    frame = pd.DataFrame.from_records(records)
    frame["realization_rate"] = frame["billed_amount"] / frame["standard_value"]
    frame["collection_rate"] = frame["collected_amount"] / frame["billed_amount"]
    frame["margin_rate"] = frame["margin"] / frame["collected_amount"]
    frame["overrun_ratio"] = frame["delivered_hours"] / frame["budgeted_hours"]
    return frame


@pytest.fixture(scope="session")
def generated() -> dict[str, pd.DataFrame]:
    """Generate the full synthetic dataset once for the whole session.

    The tables are also written to ``data/raw`` so that tests which exercise the
    loading path read exactly the data the generator produced in this run, rather
    than whatever happened to be left on disk by an earlier one. Any cached load
    is cleared first, for the same reason.

    Returns:
        Mapping of table name to DataFrame, as produced by the generator.
    """
    from src.analysis import build_facts, load_raw
    from src.config import get_settings
    from src.generate_data import generate_all, write_csvs

    settings = get_settings()
    settings.ensure_directories()
    tables = generate_all(settings)
    write_csvs(tables, settings)
    load_raw.cache_clear()
    build_facts.cache_clear()
    return tables
