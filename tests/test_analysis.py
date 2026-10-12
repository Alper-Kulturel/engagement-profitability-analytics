"""Tests for the engagement profitability calculations.

Every expected value in this module is derived by hand from the fixtures in
``conftest.py``. The arithmetic is deliberately round-numbered so that the
expected answers can be checked by reading the fixture and doing the sum.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import (
    MONEY_COLUMNS,
    allocate_money_by_year,
    annual_summary,
    assign_client_tiers,
    bottom_quartile_engagements,
    client_profitability,
    firm_summary,
    firm_utilization,
    group_by,
    overrun_vs_margin,
    realization_decile_profile,
    sensitivity_realization,
    tier_summary,
    uplift_value,
    utilization_by_consultant,
    utilization_by_level,
    waterfall,
    waterfall_reconciles,
)


class TestFirmSummary:
    """Firm-wide rates are measured against the right denominator."""

    def test_totals_are_the_sum_of_the_engagements(self, facts: pd.DataFrame) -> None:
        """Each money column totals across the two fixture engagements."""
        summary = firm_summary(facts)
        assert summary["standard_value"] == pytest.approx(1500.0)
        assert summary["billed_amount"] == pytest.approx(1330.0)
        assert summary["collected_amount"] == pytest.approx(1290.0)
        assert summary["margin"] == pytest.approx(570.0)
        assert summary["engagements"] == 2
        assert summary["clients"] == 2

    def test_realization_is_billed_over_rate_card(self, facts: pd.DataFrame) -> None:
        """Realisation is 1,330 of billing against 1,500 of rate-card value."""
        assert firm_summary(facts)["realization_rate"] == pytest.approx(1330.0 / 1500.0)

    def test_collection_is_collected_over_billed(self, facts: pd.DataFrame) -> None:
        """Collection is measured against the invoice, not the rate card."""
        assert firm_summary(facts)["collection_rate"] == pytest.approx(1290.0 / 1330.0)

    def test_margin_rate_is_measured_against_collections(self, facts: pd.DataFrame) -> None:
        """Margin is a share of net revenue, not of rate-card value."""
        assert firm_summary(facts)["margin_rate"] == pytest.approx(570.0 / 1290.0)

    def test_zero_denominator_yields_nan_not_an_exception(self) -> None:
        """An empty fact table summarises without raising."""
        empty = pd.DataFrame(
            {column: pd.Series(dtype=float) for column in MONEY_COLUMNS}
            | {
                "engagement_id": pd.Series(dtype=str),
                "client_id": pd.Series(dtype=str),
            }
        )
        summary = firm_summary(empty)
        assert np.isnan(summary["margin_rate"])
        assert np.isnan(summary["realization_rate"])


class TestWaterfall:
    """The waterfall must add up exactly, and each step must be attributed once."""

    def test_steps_sum_to_the_closing_margin(self, facts: pd.DataFrame) -> None:
        """Opening value less every deduction equals the margin."""
        assert waterfall_reconciles(facts)

    def test_opening_and_closing_bars_are_positive(self, facts: pd.DataFrame) -> None:
        """The standard value and the margin are drawn above the axis."""
        frame = waterfall(facts)
        opening = frame[frame["is_opening"]]
        assert list(opening["step"]) == ["standard_value", "margin"]
        assert (opening["amount"] > 0).all()

    def test_every_deduction_is_negative(self, facts: pd.DataFrame) -> None:
        """Discounts, unbilled time, write-offs and both cost lines reduce value."""
        frame = waterfall(facts)
        deductions = frame[~frame["is_opening"]]
        assert (deductions["amount"] < 0).all()
        assert len(deductions) == 5

    def test_running_total_ends_at_the_margin(self, facts: pd.DataFrame) -> None:
        """The running total after the final deduction is the margin."""
        frame = waterfall(facts)
        assert frame["running_total"].iloc[-1] == pytest.approx(570.0)

    def test_amounts_match_the_hand_computed_values(self, facts: pd.DataFrame) -> None:
        """Each step carries the fixture's own figure."""
        frame = waterfall(facts).set_index("step")["amount"].to_dict()
        assert frame["standard_value"] == pytest.approx(1500.0)
        assert frame["discount_amount"] == pytest.approx(-100.0)
        assert frame["unbilled_value"] == pytest.approx(-70.0)
        assert frame["write_off_amount"] == pytest.approx(-40.0)
        assert frame["direct_delivery_cost"] == pytest.approx(-120.0)
        assert frame["consultant_cost"] == pytest.approx(-600.0)
        assert frame["margin"] == pytest.approx(570.0)

    def test_reconciliation_detects_a_broken_waterfall(self, facts: pd.DataFrame) -> None:
        """A deliberately inconsistent margin fails the identity check."""
        broken = facts.copy()
        broken.loc[0, "margin"] = broken.loc[0, "margin"] + 1.0
        assert not waterfall_reconciles(broken)


class TestUtilization:
    """Utilisation is billable hours over available hours."""

    def test_per_consultant_utilisation(self, consultants: pd.DataFrame) -> None:
        """Each consultant's ratio is computed from their own two numbers."""
        frame = utilization_by_consultant(
            pd.DataFrame(
                {
                    "consultant_id": ["C1", "C2", "C3"],
                    "hours": [800.0, 300.0, 1900.0],
                }
            ),
            consultants,
        ).set_index("consultant_id")
        assert frame.loc["C1", "utilization"] == pytest.approx(0.80)
        assert frame.loc["C2", "utilization"] == pytest.approx(0.60)
        assert frame.loc["C3", "utilization"] == pytest.approx(0.95)

    def test_unused_hours_are_the_remainder(self, consultants: pd.DataFrame) -> None:
        """Available less billable is what the firm is paying for and not selling."""
        frame = utilization_by_consultant(
            pd.DataFrame({"consultant_id": ["C1"], "hours": [800.0]}), consultants
        ).set_index("consultant_id")
        assert frame.loc["C1", "unused_hours"] == pytest.approx(200.0)

    def test_firm_utilisation_weights_by_available_hours(self, consultants: pd.DataFrame) -> None:
        """The firm figure is 3,000 of 3,500 hours, not the mean of the three rates."""
        frame = utilization_by_consultant(
            pd.DataFrame(
                {
                    "consultant_id": ["C1", "C2", "C3"],
                    "hours": [800.0, 300.0, 1900.0],
                }
            ),
            consultants,
        )
        assert firm_utilization(frame) == pytest.approx(3000.0 / 3500.0)
        assert firm_utilization(frame) != pytest.approx((0.80 + 0.60 + 0.95) / 3)

    def test_level_utilisation_is_recomputed_from_summed_hours(
        self, consultants: pd.DataFrame
    ) -> None:
        """Consultants C2 and C3 pool to 2,200 of 2,500 hours."""
        frame = utilization_by_consultant(
            pd.DataFrame(
                {
                    "consultant_id": ["C1", "C2", "C3"],
                    "hours": [800.0, 300.0, 1900.0],
                }
            ),
            consultants,
        )
        by_level = utilization_by_level(frame).set_index("level")
        assert by_level.loc["Consultant", "billable_hours"] == pytest.approx(2200.0)
        assert by_level.loc["Consultant", "available_hours"] == pytest.approx(2500.0)
        assert by_level.loc["Consultant", "utilization"] == pytest.approx(2200.0 / 2500.0)
        assert by_level.loc["Consultant", "consultants"] == 2

    def test_a_consultant_with_no_hours_is_not_dropped(self, consultants: pd.DataFrame) -> None:
        """Someone on the bench still counts against the firm's denominator."""
        frame = utilization_by_consultant(
            pd.DataFrame({"consultant_id": ["C1"], "hours": [800.0]}), consultants
        )
        assert len(frame) == 3
        assert frame["utilization"].isna().sum() == 0


class TestUpliftAndSensitivity:
    """Recovering realisation is worth the billing it recovers, net of collection."""

    def test_bottom_quartile_selects_only_the_weakest(
        self, sensitivity_facts: pd.DataFrame
    ) -> None:
        """The 25th percentile of 0.50, 0.90, 0.95, 0.98 is 0.80."""
        selected = bottom_quartile_engagements(sensitivity_facts)
        assert list(selected["engagement_id"]) == ["E1"]

    def test_uplift_is_rate_card_times_points_times_collection(
        self, sensitivity_facts: pd.DataFrame
    ) -> None:
        """Five points on 1,000 of rate card at a 0.8 collection rate is 40."""
        selected = bottom_quartile_engagements(sensitivity_facts)
        assert uplift_value(selected, 5.0) == pytest.approx(40.0)

    def test_sensitivity_scales_linearly_with_points(
        self, sensitivity_facts: pd.DataFrame
    ) -> None:
        """Two, five and ten points give 16, 40 and 80."""
        table = sensitivity_realization(sensitivity_facts, [2.0, 5.0, 10.0]).set_index("points")
        assert table.loc[2.0, "incremental_margin"] == pytest.approx(16.0)
        assert table.loc[5.0, "incremental_margin"] == pytest.approx(40.0)
        assert table.loc[10.0, "incremental_margin"] == pytest.approx(80.0)

    def test_sensitivity_reports_the_group_it_used(
        self, sensitivity_facts: pd.DataFrame
    ) -> None:
        """The table states the population so the figure can be audited."""
        table = sensitivity_realization(sensitivity_facts, [5.0])
        assert int(table.loc[0, "engagements"]) == 1
        assert table.loc[0, "standard_value"] == pytest.approx(1000.0)
        assert table.loc[0, "collection_rate"] == pytest.approx(0.8)
        assert table.loc[0, "incremental_billing"] == pytest.approx(50.0)

    def test_zero_points_recovers_nothing(self, sensitivity_facts: pd.DataFrame) -> None:
        """The floor of the sensitivity is zero, not the existing margin."""
        assert uplift_value(sensitivity_facts, 0.0) == pytest.approx(0.0)


class TestClientTiers:
    """Tiers are cut on cumulative margin share."""

    def test_tier_boundaries(self) -> None:
        """Margins of 500, 300, 100, 100 become A, B, C, C.

        Cumulative shares are 0.5, 0.8, 0.9 and 1.0 against cuts at 0.5 and 0.8,
        and both cuts are inclusive.
        """
        tiers = assign_client_tiers(pd.Series([500.0, 300.0, 100.0, 100.0], index=list("ABCD")))
        assert list(tiers["tier"]) == ["A", "B", "C", "C"]
        assert list(tiers["rank"]) == [1, 2, 3, 4]

    def test_cumulative_share_is_monotone_and_ends_at_one(self) -> None:
        """The curve reaches exactly 1.0 at the last client."""
        tiers = assign_client_tiers(pd.Series([500.0, 300.0, 100.0, 100.0], index=list("ABCD")))
        shares = tiers["cumulative_share"].to_numpy()
        assert np.all(np.diff(shares) >= 0)
        assert shares[-1] == pytest.approx(1.0)

    def test_largest_client_is_always_tier_a(self) -> None:
        """A client carrying more than half of margin is still tier A."""
        tiers = assign_client_tiers(pd.Series([900.0, 50.0, 50.0], index=list("ABC")))
        assert tiers["tier"].iloc[0] == "A"
        assert list(tiers["tier"]) == ["A", "C", "C"]

    def test_a_single_client_is_tier_a(self) -> None:
        """One client holds all the margin, and tier A is never empty."""
        tiers = assign_client_tiers(pd.Series([100.0], index=["A"]))
        assert list(tiers["tier"]) == ["A"]

    def test_exact_cut_lands_in_the_lower_tier(self) -> None:
        """Both cuts are inclusive, so a share of exactly 0.5 is still tier A."""
        tiers = assign_client_tiers(pd.Series([500.0, 500.0], index=list("AB")))
        assert tiers["cumulative_share"].iloc[0] == pytest.approx(0.5)
        assert list(tiers["tier"]) == ["A", "C"]

    def test_non_positive_total_margin_is_rejected(self) -> None:
        """Cumulative share is undefined when the firm loses money overall."""
        with pytest.raises(ValueError, match="must be positive"):
            assign_client_tiers(pd.Series([-100.0, 50.0], index=list("AB")))

    def test_tier_summary_shares_sum_to_one(self, facts: pd.DataFrame) -> None:
        """Revenue and margin shares each account for the whole firm."""
        clients = client_profitability(facts)
        tiers = tier_summary(clients)
        assert tiers["revenue_share"].sum() == pytest.approx(1.0)
        assert tiers["margin_share"].sum() == pytest.approx(1.0)
        assert tiers["clients"].sum() == len(clients)

    def test_client_margin_totals_match_the_firm(self, facts: pd.DataFrame) -> None:
        """Splitting by client conserves the firm's margin."""
        clients = client_profitability(facts)
        assert clients["margin"].sum() == pytest.approx(facts["margin"].sum())


class TestGrouping:
    """Grouping conserves money and recomputes rates from summed values."""

    def test_group_totals_conserve_the_firm(self, facts: pd.DataFrame) -> None:
        """Each money column survives a group-by unchanged."""
        grouped = group_by(facts, "service_line")
        for column in MONEY_COLUMNS:
            assert grouped[column].sum() == pytest.approx(facts[column].sum())

    def test_group_rates_are_not_averages_of_rates(self, facts: pd.DataFrame) -> None:
        """Realisation for the fixed-fee group is that group's own ratio."""
        grouped = group_by(facts, "contract_type").set_index("contract_type")
        assert grouped.loc["fixed-fee", "realization_rate"] == pytest.approx(850.0 / 1000.0)
        assert grouped.loc["time-and-materials", "realization_rate"] == pytest.approx(480.0 / 500.0)

    def test_unknown_dimension_is_rejected(self, facts: pd.DataFrame) -> None:
        """A typo in the dimension name raises rather than returning nonsense."""
        with pytest.raises(KeyError):
            group_by(facts, "not_a_column")

    def test_groups_are_ordered_by_margin(self, facts: pd.DataFrame) -> None:
        """The most profitable group comes first."""
        grouped = group_by(facts, "service_line")
        assert list(grouped["service_line"]) == ["Strategy", "Digital"]


class TestOverrunAndDeciles:
    """Overrun and realisation are related to margin in the expected direction."""

    def test_negative_correlation_when_the_overrun_earns_less(self, facts: pd.DataFrame) -> None:
        """E1 overran to 1.1x on a 41.5% margin; E2 came in at 0.9x on 48.9%."""
        result = overrun_vs_margin(facts, bins=2)
        assert result["correlation"] == pytest.approx(-1.0)
        assert result["slope"] < 0

    def test_bands_partition_the_engagements(self, facts: pd.DataFrame) -> None:
        """Every engagement lands in exactly one band, and none is counted twice."""
        bands = overrun_vs_margin(facts, bins=2)["bands"]
        assert bands["engagements"].sum() == len(facts)
        assert bands["margin"].sum() == pytest.approx(facts["margin"].sum())

    def test_decile_profile_is_ordered_and_complete(self, facts: pd.DataFrame) -> None:
        """Groups run from worst realisation upward and account for every row."""
        profile = realization_decile_profile(facts, bins=2)
        assert profile["engagements"].sum() == len(facts)
        assert profile["median_realization"].is_monotonic_increasing
        assert profile["margin"].sum() == pytest.approx(facts["margin"].sum())

    def test_decile_medians_track_margin_upward(self, sensitivity_facts: pd.DataFrame) -> None:
        """A fixture built so that margin rises with realisation shows it."""
        frame = sensitivity_facts.assign(
            margin_rate=[-0.20, -0.05, 0.10, 0.25],
        )
        profile = realization_decile_profile(frame, bins=4)
        assert profile["median_margin_rate"].is_monotonic_increasing
        assert profile["median_margin_rate"].iloc[0] < 0 < profile["median_margin_rate"].iloc[-1]


class TestYearAllocation:
    """Splitting money across years must conserve every column."""

    def test_allocation_conserves_money_and_hours(self, facts: pd.DataFrame) -> None:
        """Year rows sum back to the engagement's own figures."""
        timesheets = pd.DataFrame.from_records(
            [
                {"engagement_id": "E1", "year": 2023, "hours": 60.0},
                {"engagement_id": "E1", "year": 2024, "hours": 50.0},
                {"engagement_id": "E2", "year": 2024, "hours": 45.0},
            ]
        )
        allocated = allocate_money_by_year(facts, timesheets)
        for column in MONEY_COLUMNS:
            assert allocated[column].sum() == pytest.approx(facts[column].sum())
        assert allocated["delivered_hours"].sum() == pytest.approx(155.0)

    def test_allocation_uses_the_hours_worked_in_each_year(self, facts: pd.DataFrame) -> None:
        """E1 splits 60:50, so 2023 carries 60/110 of its margin."""
        timesheets = pd.DataFrame.from_records(
            [
                {"engagement_id": "E1", "year": 2023, "hours": 60.0},
                {"engagement_id": "E1", "year": 2024, "hours": 50.0},
            ]
        )
        allocated = allocate_money_by_year(facts[facts["engagement_id"] == "E1"], timesheets)
        row_2023 = allocated[allocated["year"] == 2023].iloc[0]
        assert row_2023["margin"] == pytest.approx(340.0 * 60.0 / 110.0)

    def test_allocation_preserves_the_waterfall_identity(self, facts: pd.DataFrame) -> None:
        """The identity survives the split, so year rows can be reported on."""
        timesheets = pd.DataFrame.from_records(
            [
                {"engagement_id": "E1", "year": 2023, "hours": 60.0},
                {"engagement_id": "E1", "year": 2024, "hours": 50.0},
                {"engagement_id": "E2", "year": 2024, "hours": 45.0},
            ]
        )
        allocated = allocate_money_by_year(facts, timesheets)
        assert waterfall_reconciles(allocated)

    def test_annual_summary_reports_each_year(self, facts: pd.DataFrame) -> None:
        """One row per year, with the year column present."""
        timesheets = pd.DataFrame.from_records(
            [
                {"engagement_id": "E1", "year": 2023, "hours": 60.0},
                {"engagement_id": "E1", "year": 2024, "hours": 50.0},
                {"engagement_id": "E2", "year": 2024, "hours": 45.0},
            ]
        )
        summary = annual_summary(allocate_money_by_year(facts, timesheets))
        assert list(summary["year"]) == [2023, 2024]
        assert summary["margin"].sum() == pytest.approx(570.0)
