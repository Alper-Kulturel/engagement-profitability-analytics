"""Tests for the HTML report and the analysis bundle behind it.

The report is the deliverable a reader actually opens, so the tests here check
the things that would embarrass it: a stray external reference that breaks when
the file is opened offline, an unbalanced tag that mangles the layout, or a
headline number that does not match the data it claims to describe.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
import pytest

from src.analysis import firm_summary, waterfall_reconciles
from src.config import get_settings
from src.report import AnalysisBundle, build_bundle, recommendations, render_html

VOID_ELEMENTS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}


class _TagBalance(HTMLParser):
    """Track open and close tags so an unbalanced document can be detected."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.tags: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append(tag)
        if tag not in VOID_ELEMENTS:
            self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in VOID_ELEMENTS:
            return
        if not self.stack:
            self.errors.append(f"closing </{tag}> with nothing open")
            return
        if self.stack[-1] != tag:
            self.errors.append(f"expected </{self.stack[-1]}>, found </{tag}>")
            return
        self.stack.pop()


@pytest.fixture(scope="module")
def bundle(generated: dict[str, pd.DataFrame]) -> AnalysisBundle:
    """Build the analysis bundle once for the module.

    Args:
        generated: The session's generated tables, which this fixture depends on
            so that the CSVs on disk are the ones this run produced.

    Returns:
        The fully computed bundle, including rendered figures.
    """
    return build_bundle(get_settings())


@pytest.fixture(scope="module")
def html(bundle: AnalysisBundle) -> str:
    """Render the report once for the module.

    Args:
        bundle: The computed analysis bundle.

    Returns:
        The HTML document as a string.
    """
    return render_html(bundle)


class TestBundle:
    """The bundle carries everything the report needs, computed once."""

    def test_bundle_reconciles_its_own_waterfall(self, bundle: AnalysisBundle) -> None:
        """The headline waterfall adds up on the year it reports."""
        assert waterfall_reconciles(bundle.headline)

    def test_headline_thesis_holds(self, bundle: AnalysisBundle) -> None:
        """The gap between rate card and collections is the leakage, to the cent."""
        summary = bundle.headline_summary
        leakage = (
            summary["discount_amount"] + summary["unbilled_value"] + summary["write_off_amount"]
        )
        gap = summary["standard_value"] - summary["collected_amount"]
        assert leakage == pytest.approx(gap, abs=0.01)

    def test_headline_summary_matches_the_headline_rows(self, bundle: AnalysisBundle) -> None:
        """The summary is a summary of the data it is published beside."""
        recomputed = firm_summary(bundle.headline)
        assert bundle.headline_summary["margin"] == pytest.approx(recomputed["margin"])
        assert bundle.headline_summary["realization_rate"] == pytest.approx(
            recomputed["realization_rate"]
        )

    def test_every_figure_was_rendered(self, bundle: AnalysisBundle) -> None:
        """All five figures exist on disk and carry image data."""
        assert set(bundle.figures) == {"waterfall", "pareto", "utilization", "scatter", "overrun"}
        for name, path in bundle.figures.items():
            assert path.exists(), name
            assert path.stat().st_size > 5_000, name

    def test_dimension_tables_conserve_the_portfolio_margin(self, bundle: AnalysisBundle) -> None:
        """Splitting the portfolio by any dimension preserves its margin."""
        for frame in (bundle.by_service_line, bundle.by_contract, bundle.by_level):
            assert frame["margin"].sum() == pytest.approx(bundle.portfolio["margin"], abs=0.01)

    def test_client_table_conserves_the_portfolio_margin(self, bundle: AnalysisBundle) -> None:
        """Every client's margin added up is the portfolio's margin."""
        assert bundle.clients["margin"].sum() == pytest.approx(
            bundle.portfolio["margin"], abs=0.01
        )

    def test_tiers_cover_every_client_exactly_once(self, bundle: AnalysisBundle) -> None:
        """The A/B/C split is a partition of the client book."""
        assert int(bundle.tiers["clients"].sum()) == len(bundle.clients)
        assert set(bundle.tiers["tier"]) <= {"A", "B", "C"}

    def test_sensitivity_uses_the_same_arithmetic_as_the_memo(
        self, bundle: AnalysisBundle
    ) -> None:
        """The published sensitivity and the recommendations agree."""
        from src.analysis import uplift_value

        bottom = bundle.facts[bundle.facts["realization_rate"] <= bundle.facts["realization_rate"].quantile(0.25)]
        row = bundle.sensitivity.set_index("points").loc[5.0]
        assert row["incremental_margin"] == pytest.approx(uplift_value(bottom, 5.0))


class TestRecommendations:
    """The memo's recommendations are ordered, quantified and honest."""

    def test_three_recommendations_are_returned(self, bundle: AnalysisBundle) -> None:
        """The memo makes exactly three asks."""
        assert len(recommendations(bundle)) == 3

    def test_recommendations_are_ordered_by_value(self, bundle: AnalysisBundle) -> None:
        """The biggest prize is argued first."""
        values = [item["value"] for item in recommendations(bundle)]
        assert values == sorted(values, reverse=True)

    def test_every_recommendation_has_a_title_a_value_and_a_basis(
        self, bundle: AnalysisBundle
    ) -> None:
        """Nothing is recommended without a number, evidence and an action."""
        for item in recommendations(bundle):
            assert item["title"]
            assert item["value"] > 0
            assert item["evidence"]
            assert item["action"]

    def test_the_headline_recommendation_states_its_assumption(
        self, bundle: AnalysisBundle
    ) -> None:
        """An estimate built on a comparison says what it is compared against."""
        by_title = {item["title"]: item for item in recommendations(bundle)}
        gap = by_title["Close the fixed-fee realisation gap"]
        assert "assumes" in gap["assumption"].lower()
        assert "point" in gap["assumption"]

    def test_the_biggest_recommendation_is_inside_the_firm(self, bundle: AnalysisBundle) -> None:
        """A recovery cannot exceed the leakage that created the gap."""
        leakage = (
            bundle.headline_summary["discount_amount"]
            + bundle.headline_summary["unbilled_value"]
            + bundle.headline_summary["write_off_amount"]
        )
        assert max(item["value"] for item in recommendations(bundle)) <= leakage

    def test_the_recommendation_values_are_individually_achievable(
        self, bundle: AnalysisBundle
    ) -> None:
        """Each ask is a fraction of rate-card value, not a multiple of it."""
        standard_value = bundle.headline_summary["standard_value"]
        for item in recommendations(bundle):
            assert item["value"] < 0.10 * standard_value, item["title"]


class TestHtml:
    """The rendered document is self-contained and well formed."""

    def test_no_external_references(self, html: str) -> None:
        """Nothing is fetched over the network, so the file opens offline."""
        assert "http://" not in html
        assert "https://" not in html
        assert "//cdn" not in html
        assert "<script" not in html.lower()

    def test_all_images_are_inlined(self, html: str) -> None:
        """Figures are base64 data URIs rather than links to files."""
        assert html.count("data:image/png;base64,") == 5
        assert not re.search(r'<img[^>]+src="(?!data:)', html)

    def test_tags_balance(self, html: str) -> None:
        """The document has no unclosed or stray tags."""
        parser = _TagBalance()
        parser.feed(html)
        assert parser.errors == []
        assert parser.stack == []

    def test_document_scaffolding_is_present(self, html: str) -> None:
        """It is a complete document with a title and a viewport."""
        assert html.lstrip().lower().startswith("<!doctype html>")
        assert "<title>" in html
        assert 'name="viewport"' in html
        assert html.rstrip().endswith("</html>")

    def test_headline_numbers_appear_in_the_document(self, html: str, bundle: AnalysisBundle) -> None:
        """The published margin rate and realisation rate are on the page."""
        summary = bundle.headline_summary
        assert f"{summary['margin_rate'] * 100:.1f}%" in html
        assert f"{summary['realization_rate'] * 100:.1f}%" in html

    def test_the_synthetic_data_caveat_is_stated(self, html: str) -> None:
        """A reader cannot mistake the dataset for real client work."""
        lowered = html.lower()
        assert "synthetic" in lowered
        assert "seed 42" in lowered
        assert "contains no real" in lowered

    def test_the_limitations_are_stated(self, html: str) -> None:
        """The allocation and sensitivity caveats travel with the numbers."""
        lowered = html.lower()
        assert "limitations" in lowered
        assert "allocation" in lowered
        assert "static estimate" in lowered

    def test_the_recommendations_are_flagged_as_overlapping(self, html: str) -> None:
        """A reader is told not to add the three figures together."""
        lowered = html.lower()
        assert "overlapping" in lowered
        assert "added together" in lowered

    def test_no_placeholder_text_survived(self, html: str) -> None:
        """Nothing was left as a template token or an unresolved format string."""
        assert "TODO" not in html
        assert "FIXME" not in html
        assert not re.search(r"\{[a-zA-Z_][a-zA-Z0-9_.\[\]'\"]*[!:}]", html)
        assert not re.search(r"<span[^>]*>\s*</span>", html)

    def test_the_document_is_a_reasonable_size(self, html: str) -> None:
        """It carries five inlined figures without becoming unopenable."""
        assert 200_000 < len(html) < 12_000_000


class TestWriters:
    """Writing the report to disk is idempotent and self-describing."""

    def test_write_report_is_reproducible(self, bundle: AnalysisBundle) -> None:
        """Rerunning the report overwrites the same file with the same content."""
        from src.report import write_report

        first = write_report(bundle)
        content = first.read_text(encoding="utf-8")
        second = write_report(bundle)
        assert first == second
        assert second.read_text(encoding="utf-8") == content

    def test_report_filename_carries_the_run_date(self, bundle: AnalysisBundle) -> None:
        """The file name says when it was produced."""
        from src.report import write_report

        path = write_report(bundle)
        assert re.fullmatch(r"engagement_profitability_report_\d{8}\.html", path.name)

    def test_metrics_json_is_written_and_parsable(self, bundle: AnalysisBundle) -> None:
        """The machine-readable metrics file round-trips."""
        import json

        from src.report import write_metrics_json

        path = write_metrics_json(bundle)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["headline_year"] == get_settings().headline_year
        assert payload["headline"]["margin"] == pytest.approx(bundle.headline_summary["margin"])

    def test_tables_are_written_for_every_dimension(self, bundle: AnalysisBundle) -> None:
        """Each published table lands on disk and is readable."""
        from src.report import write_tables

        paths = write_tables(bundle)
        assert len(paths) >= 10
        for path in paths:
            assert path.exists()
            assert not pd.read_csv(path).empty, path.name


def test_figures_are_not_written_outside_the_reports_directory(bundle: AnalysisBundle) -> None:
    """Rendering keeps its output inside the project."""
    root = get_settings().project_root
    for path in bundle.figures.values():
        assert Path(path).resolve().is_relative_to(root)
