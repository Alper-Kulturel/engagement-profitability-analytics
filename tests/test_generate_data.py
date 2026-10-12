"""Tests for the synthetic dataset generator.

These tests check the invariants the rest of the pipeline relies on: that the
data is reproducible from the seed, that the tables reference each other
correctly, and that the money is internally consistent. They do not check the
particular values the generator happens to produce, which would make the suite
a change detector rather than a test.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from src.analysis import MONEY_COLUMNS, WATERFALL_DEDUCTIONS, waterfall_reconciles
from src.config import (
    ANALYSIS_END,
    ANALYSIS_START,
    LEVELS,
    N_CLIENTS,
    N_CONSULTANTS,
    N_ENGAGEMENTS,
    get_settings,
)
from src.generate_data import generate_all

TABLE_NAMES = ("consultants", "clients", "engagements", "timesheets", "billing")


class TestReproducibility:
    """The seed is the only input to the shape of the dataset."""

    def test_the_same_seed_gives_the_same_tables(self, generated: dict[str, pd.DataFrame]) -> None:
        """Regenerating from seed 42 reproduces every table exactly."""
        again = generate_all(get_settings())
        for name in TABLE_NAMES:
            pd.testing.assert_frame_equal(generated[name], again[name])

    def test_a_different_seed_gives_different_money(self) -> None:
        """Change the seed and the portfolio changes with it."""
        settings = get_settings()
        other = dataclasses.replace(settings, random_seed=settings.random_seed + 1)
        baseline = generate_all(settings)["billing"]
        shifted = generate_all(other)["billing"]
        assert not baseline["margin"].equals(shifted["margin"])

    def test_firm_size_is_unaffected_by_the_seed(self) -> None:
        """The seed changes the draws, not how big the firm is."""
        settings = get_settings()
        other = dataclasses.replace(settings, random_seed=settings.random_seed + 1)
        baseline = generate_all(settings)
        shifted = generate_all(other)
        for name in ("consultants", "clients", "engagements", "billing"):
            assert len(baseline[name]) == len(shifted[name]), name

    def test_a_different_seed_gives_a_different_rota(self) -> None:
        """Staffing is a draw, so a new seed produces different timesheets."""
        settings = get_settings()
        other = dataclasses.replace(settings, random_seed=settings.random_seed + 1)
        baseline = generate_all(settings)["timesheets"].reset_index(drop=True)
        shifted = generate_all(other)["timesheets"].reset_index(drop=True)
        assert not baseline.equals(shifted)

    def test_the_rota_covers_every_engagement_and_grade(
        self, generated: dict[str, pd.DataFrame]
    ) -> None:
        """Every engagement is staffed, by at least one person of every grade."""
        timesheets = generated["timesheets"]
        assert timesheets["engagement_id"].nunique() == N_ENGAGEMENTS
        per_engagement = timesheets.groupby("engagement_id")["level"].nunique()
        assert (per_engagement == len(LEVELS)).all()

    def test_the_rota_is_bounded_by_the_calendar(
        self, generated: dict[str, pd.DataFrame]
    ) -> None:
        """No engagement books more person-months than it has months and people."""
        timesheets = generated["timesheets"]
        row_count = len(timesheets)
        person_months = timesheets.groupby("engagement_id").apply(
            lambda group: group["month"].nunique() * group["consultant_id"].nunique(),
            include_groups=False,
        )
        assert (timesheets.groupby("engagement_id").size() <= person_months).all()
        assert row_count < N_ENGAGEMENTS * N_CONSULTANTS * 36


class TestShapes:
    """The portfolio is the size the configuration says it is."""

    def test_consultant_and_client_counts(self, generated: dict[str, pd.DataFrame]) -> None:
        """Roster and client book match the configured firm size."""
        assert len(generated["consultants"]) == N_CONSULTANTS
        assert len(generated["clients"]) == N_CLIENTS

    def test_engagement_count(self, generated: dict[str, pd.DataFrame]) -> None:
        """Every configured engagement is generated."""
        assert len(generated["engagements"]) == N_ENGAGEMENTS

    def test_identifiers_are_unique(self, generated: dict[str, pd.DataFrame]) -> None:
        """Primary keys do not repeat."""
        for name, key in (
            ("consultants", "consultant_id"),
            ("clients", "client_id"),
            ("engagements", "engagement_id"),
            ("billing", "engagement_id"),
        ):
            column = generated[name][key]
            assert column.is_unique, name

    def test_no_missing_values_in_key_columns(self, generated: dict[str, pd.DataFrame]) -> None:
        """Nothing the analysis joins on is null."""
        for name, keys in (
            ("consultants", ["consultant_id", "level", "available_hours"]),
            ("clients", ["client_id", "client_name", "industry"]),
            ("engagements", ["engagement_id", "client_id", "start_date", "end_date"]),
            ("timesheets", ["engagement_id", "consultant_id", "hours", "year"]),
            ("billing", ["engagement_id", *MONEY_COLUMNS]),
        ):
            assert not generated[name][keys].isna().any().any(), name


class TestReferentialIntegrity:
    """Foreign keys resolve, and nothing is orphaned."""

    def test_engagements_reference_real_clients(self, generated: dict[str, pd.DataFrame]) -> None:
        """Every engagement belongs to a client on the books."""
        known = set(generated["clients"]["client_id"])
        assert set(generated["engagements"]["client_id"]) <= known

    def test_every_client_has_at_least_one_engagement(
        self, generated: dict[str, pd.DataFrame]
    ) -> None:
        """No client sits on the books with no work attached."""
        booked = set(generated["engagements"]["client_id"])
        assert set(generated["clients"]["client_id"]) <= booked

    def test_timesheets_reference_real_engagements_and_people(
        self, generated: dict[str, pd.DataFrame]
    ) -> None:
        """Timesheet rows point at engagements that exist and staff who exist."""
        timesheets = generated["timesheets"]
        assert set(timesheets["engagement_id"]) <= set(generated["engagements"]["engagement_id"])
        assert set(timesheets["consultant_id"]) <= set(generated["consultants"]["consultant_id"])

    def test_billing_covers_every_engagement_exactly_once(
        self, generated: dict[str, pd.DataFrame]
    ) -> None:
        """Billing and engagements are the same set of identifiers."""
        assert set(generated["billing"]["engagement_id"]) == set(
            generated["engagements"]["engagement_id"]
        )

    def test_time_is_worked_inside_the_engagement_window(
        self, generated: dict[str, pd.DataFrame]
    ) -> None:
        """No timesheet hour precedes the start or follows the end of its engagement."""
        month = generated["timesheets"]["month"].str.split("-", expand=True).astype(int)
        # ``start_month_index`` counts months from the analysis window start, so
        # the month key has to be converted to the same axis before comparing.
        worked = (month[0] - ANALYSIS_START.year) * 12 + (month[1] - ANALYSIS_START.month)
        merged = generated["timesheets"][["engagement_id"]].assign(worked=worked).merge(
            generated["engagements"][["engagement_id", "start_month_index", "duration_months"]],
            on="engagement_id",
            how="left",
            validate="many_to_one",
        )
        offset = merged["worked"] - merged["start_month_index"]
        assert (offset >= 0).all()
        assert (offset < merged["duration_months"]).all()

    def test_engagement_dates_match_the_month_offsets(
        self, generated: dict[str, pd.DataFrame]
    ) -> None:
        """The start date on every engagement is the first of its start month."""
        engagements = generated["engagements"]
        expected = pd.to_datetime(
            [
                f"{ANALYSIS_START.year + (ANALYSIS_START.month - 1 + offset) // 12:04d}-"
                f"{(ANALYSIS_START.month - 1 + offset) % 12 + 1:02d}-01"
                for offset in engagements["start_month_index"]
            ]
        )
        assert (pd.to_datetime(engagements["start_date"]).to_numpy() == expected.to_numpy()).all()

    def test_timesheet_level_and_rate_match_the_roster(
        self, generated: dict[str, pd.DataFrame]
    ) -> None:
        """The grade and rate stamped on an hour are the consultant's own."""
        roster = generated["consultants"].set_index("consultant_id")
        timesheets = generated["timesheets"]
        assert (
            timesheets["level"].to_numpy() == roster.loc[timesheets["consultant_id"], "level"].to_numpy()
        ).all()
        assert (
            timesheets["standard_rate"].to_numpy()
            == roster.loc[timesheets["consultant_id"], "standard_rate"].to_numpy()
        ).all()


class TestHours:
    """Hours delivered are the hours the engagement was planned to take."""

    def test_timesheet_hours_reconcile_to_the_plan(self, generated: dict[str, pd.DataFrame]) -> None:
        """Summed timesheet hours equal the engagement's planned delivered hours.

        The generator books time in half hours, so the total can sit at most half
        an hour away from the plan it was distributed from.
        """
        delivered = generated["timesheets"].groupby("engagement_id")["hours"].sum()
        planned = generated["engagements"].set_index("engagement_id")["delivered_hours_plan"]
        difference = (delivered - planned.loc[delivered.index]).abs()
        assert difference.max() <= 0.5

    def test_hours_are_positive_and_in_half_hour_steps(
        self, generated: dict[str, pd.DataFrame]
    ) -> None:
        """The generator books time in half hours, never in negative amounts."""
        hours = generated["timesheets"]["hours"]
        assert (hours > 0).all()
        assert ((hours * 2) % 1 == 0).all()

    def test_billable_hours_never_exceed_availability(
        self, generated: dict[str, pd.DataFrame]
    ) -> None:
        """Nobody is booked for more hours than the calendar gives them."""
        billable = generated["timesheets"].groupby("consultant_id")["hours"].sum()
        available = generated["consultants"].set_index("consultant_id")["available_hours"]
        assert (billable <= available.loc[billable.index] + 1e-6).all()

    def test_utilisation_is_below_the_physical_ceiling(
        self, generated: dict[str, pd.DataFrame]
    ) -> None:
        """Firm utilisation is a plausible number above zero and below one."""
        billable = float(generated["timesheets"]["hours"].sum())
        available = float(generated["consultants"]["available_hours"].sum())
        assert 0.5 < billable / available < 1.0


class TestMoney:
    """The commercial columns are internally consistent."""

    def test_the_waterfall_adds_up_on_every_engagement(
        self, generated: dict[str, pd.DataFrame]
    ) -> None:
        """The billing table satisfies the identity the report depends on."""
        assert waterfall_reconciles(generated["billing"])

    def test_the_waterfall_adds_up_engagement_by_engagement(
        self, generated: dict[str, pd.DataFrame]
    ) -> None:
        """The identity is per row, not just in the total."""
        billing = generated["billing"]
        residual = billing["standard_value"].copy()
        for column in WATERFALL_DEDUCTIONS:
            residual = residual - billing[column]
        assert (residual - billing["margin"]).abs().max() < 1e-6

    def test_billing_never_exceeds_the_rate_card(self, generated: dict[str, pd.DataFrame]) -> None:
        """Discounts and unbilled time only ever reduce what is invoiced."""
        billing = generated["billing"]
        assert (billing["billed_amount"] <= billing["standard_value"] + 1e-9).all()
        assert (billing["billed_amount"] >= 0).all()

    def test_collections_never_exceed_the_invoice(self, generated: dict[str, pd.DataFrame]) -> None:
        """Write-offs reduce an invoice; they never increase it."""
        billing = generated["billing"]
        assert (billing["collected_amount"] <= billing["billed_amount"] + 1e-9).all()
        assert (billing["collected_amount"] >= 0).all()

    def test_every_money_column_is_two_decimal_currency(
        self, generated: dict[str, pd.DataFrame]
    ) -> None:
        """The published columns are rounded to the cent so the report ties out."""
        billing = generated["billing"]
        for column in MONEY_COLUMNS:
            if column == "margin":
                continue
            values = billing[column]
            assert ((values * 100).round(6) % 1 == 0).all(), column

    def test_the_portfolio_spans_the_configured_window(
        self, generated: dict[str, pd.DataFrame]
    ) -> None:
        """Time is worked in every year of the analysis window, and no other."""
        years = sorted(int(value) for value in generated["timesheets"]["year"].unique())
        assert years == list(range(ANALYSIS_START.year, ANALYSIS_END.year + 1))

    def test_margin_is_positive_in_aggregate_but_not_everywhere(
        self, generated: dict[str, pd.DataFrame]
    ) -> None:
        """The firm is profitable overall, and some engagements are not."""
        margin = generated["billing"]["margin"]
        assert margin.sum() > 0
        assert (margin < 0).any()


@pytest.mark.parametrize("name", TABLE_NAMES)
def test_every_table_is_non_empty(name: str, generated: dict[str, pd.DataFrame]) -> None:
    """No stage of the generator silently produces nothing."""
    assert not generated[name].empty
