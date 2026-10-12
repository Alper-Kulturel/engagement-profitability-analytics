"""Tests for the configuration module.

Configuration is the one place where a path or a constant is allowed to be
written down. These tests check that the singleton behaves, that the paths hang
together, and that the modelling constants are internally coherent - a rate card
that prices a grade below its own cost would make every downstream margin
meaningless.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src.config import (
    ANALYSIS_END,
    ANALYSIS_START,
    COST_RATES,
    DISCOUNT_MEAN_BY_SERVICE_LINE,
    HEADLINE_YEAR,
    LEVELS,
    LEVEL_MIX_BY_TIER,
    N_CLIENTS,
    N_CONSULTANTS,
    N_ENGAGEMENTS,
    PROJECT_ROOT,
    RANDOM_SEED,
    SENSITIVITY_POINTS,
    SERVICE_LINES,
    STANDARD_RATES,
    TIER_A_CUMULATIVE_CUT,
    TIER_B_CUMULATIVE_CUT,
    UTILIZATION_TARGET,
    Settings,
    get_settings,
)


class TestSingleton:
    """One settings object is shared by every module."""

    def test_get_settings_returns_the_same_object(self) -> None:
        """The cache means repeated calls do not rebuild the object."""
        assert get_settings() is get_settings()

    def test_settings_are_immutable(self) -> None:
        """Nothing can quietly reconfigure the pipeline at runtime."""
        with pytest.raises(FrozenInstanceError):
            get_settings().random_seed = 7  # type: ignore[misc]

    def test_manual_construction_matches_the_singleton(self) -> None:
        """A hand-built Settings resolves to the same paths as the singleton."""
        settings = Settings()
        assert settings.project_root == get_settings().project_root
        assert settings.raw_dir == get_settings().raw_dir


class TestPaths:
    """Every path is derived from the project root, not hard-coded."""

    def test_paths_sit_under_the_project_root(self) -> None:
        """Data, reports and figures all live inside the project."""
        settings = get_settings()
        for path in (
            settings.raw_dir,
            settings.reports_dir,
            settings.figures_dir,
            settings.tables_dir,
            settings.app_dir,
            settings.docs_dir,
        ):
            assert path.is_relative_to(settings.project_root)

    def test_expected_directory_names(self) -> None:
        """The layout matches the house style: data/raw, reports, app."""
        settings = get_settings()
        assert settings.raw_dir == settings.project_root / "data" / "raw"
        assert settings.reports_dir == settings.project_root / "reports"
        assert settings.figures_dir == settings.reports_dir / "figures"
        assert settings.tables_dir == settings.reports_dir / "tables"
        assert settings.app_dir == settings.project_root / "app"

    def test_project_root_is_the_package_parent(self) -> None:
        """``src`` is importable from the project root."""
        assert (PROJECT_ROOT / "cli.py").exists()
        assert (PROJECT_ROOT / "src" / "config.py").exists()

    def test_ensure_directories_is_idempotent(self) -> None:
        """Calling it twice is not an error."""
        settings = get_settings()
        settings.ensure_directories()
        settings.ensure_directories()
        assert settings.raw_dir.is_dir()


class TestRateCard:
    """The rate card is economically coherent."""

    def test_every_grade_has_a_rate_and_a_cost(self) -> None:
        """No grade is missing from either table."""
        assert set(STANDARD_RATES) == set(LEVELS)
        assert set(COST_RATES) == set(LEVELS)

    def test_every_grade_is_profitable_at_rate_card(self) -> None:
        """Billing at the standard rate covers the grade's own cost."""
        for level in LEVELS:
            assert STANDARD_RATES[level] > COST_RATES[level], level

    def test_grades_are_ordered_by_seniority(self) -> None:
        """Rates and costs both fall from Partner down to Analyst."""
        assert list(LEVELS) == ["Partner", "Manager", "Consultant", "Analyst"]
        rates = [STANDARD_RATES[level] for level in LEVELS]
        costs = [COST_RATES[level] for level in LEVELS]
        assert rates == sorted(rates, reverse=True)
        assert costs == sorted(costs, reverse=True)

    def test_cost_rates_are_a_reasonable_share_of_the_rate_card(self) -> None:
        """A grade costs between a third and two thirds of what it is billed at."""
        for level in LEVELS:
            ratio = COST_RATES[level] / STANDARD_RATES[level]
            assert 0.30 < ratio < 0.70, level

    def test_the_firm_can_be_profitable_at_its_target_utilisation(self) -> None:
        """At target utilisation the blended rate card beats the blended cost.

        This is the sanity check that stops the whole model from being
        structurally loss-making: an evenly mixed hour must make money.
        """
        blended_rate = sum(STANDARD_RATES[level] for level in LEVELS) / len(LEVELS)
        blended_cost = sum(COST_RATES[level] for level in LEVELS) / len(LEVELS)
        assert blended_rate * UTILIZATION_TARGET > blended_cost * UTILIZATION_TARGET


class TestModelConstants:
    """The modelling switches hold together."""

    def test_level_mixes_are_distributions_over_the_grades(self) -> None:
        """Each tier's mix sums to one and names only real grades."""
        for tier, mix in LEVEL_MIX_BY_TIER.items():
            assert set(mix) == set(LEVELS), tier
            assert sum(mix.values()) == pytest.approx(1.0), tier

    def test_larger_engagements_are_less_partner_heavy(self) -> None:
        """Small work carries the senior hours, large work carries the delivery."""
        assert (
            LEVEL_MIX_BY_TIER["large"]["Partner"]
            < LEVEL_MIX_BY_TIER["medium"]["Partner"]
            < LEVEL_MIX_BY_TIER["small"]["Partner"]
        )

    def test_service_line_discounts_are_plausible_fractions(self) -> None:
        """Discounts are a small share of the rate card, never a giveaway."""
        assert set(DISCOUNT_MEAN_BY_SERVICE_LINE) == set(SERVICE_LINES)
        for line, discount in DISCOUNT_MEAN_BY_SERVICE_LINE.items():
            assert 0.0 < discount < 0.25, line

    def test_tier_cuts_are_ordered_and_leave_room_for_tier_c(self) -> None:
        """Tier A closes before tier B, and tier C is never empty by definition."""
        assert 0.0 < TIER_A_CUMULATIVE_CUT < TIER_B_CUMULATIVE_CUT < 1.0

    def test_sensitivity_points_are_positive_and_ascending(self) -> None:
        """The scenarios are quoted in whole realisation points."""
        assert list(SENSITIVITY_POINTS) == sorted(SENSITIVITY_POINTS)
        assert all(point > 0 for point in SENSITIVITY_POINTS)

    def test_the_analysis_window_contains_the_headline_year(self) -> None:
        """The headline year is one of the years in the data."""
        assert ANALYSIS_START.year <= HEADLINE_YEAR <= ANALYSIS_END.year
        assert ANALYSIS_START < ANALYSIS_END

    def test_firm_size_and_seed(self) -> None:
        """The portfolio is the size and seed the README quotes."""
        assert RANDOM_SEED == 42
        assert N_ENGAGEMENTS == 200
        assert N_CLIENTS == 60
        assert N_CONSULTANTS == 90

    def test_the_utilisation_target_is_achievable_but_not_trivial(self) -> None:
        """The target sits in the band a real firm would recognise."""
        assert 0.60 <= UTILIZATION_TARGET <= 0.85
