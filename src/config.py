"""Central configuration for the Engagement Profitability Analytics pipeline.

Every filesystem path and every shared modelling constant used by this project
is declared here. No other module hard-codes a directory or a rate: they all
import from this file. That keeps the pipeline relocatable and keeps the
synthetic-data assumptions in one auditable place.

Typical use::

    from src.config import get_settings

    settings = get_settings()
    engagements = settings.raw_dir / "engagements.csv"
    print(settings.random_seed, settings.service_lines)

The settings object is a frozen dataclass produced by a cached factory, so
repeated calls within a process return the same instance and directory
creation happens exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path

#: Root of the project (the directory containing ``cli.py``).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

#: Fixed pseudo-random seed. Every generated figure in this repository is
#: reproducible from this single value.
RANDOM_SEED: int = 42

#: The four service lines the firm sells.
SERVICE_LINES: tuple[str, ...] = ("Strategy", "Digital", "Operations", "Risk")

#: The three commercial contract types.
CONTRACT_TYPES: tuple[str, ...] = ("time-and-materials", "fixed-fee", "retainer")

#: Consultant grades, most senior first.
LEVELS: tuple[str, ...] = ("Partner", "Manager", "Consultant", "Analyst")

#: Rate-card value of one hour by grade, in US dollars.
STANDARD_RATES: dict[str, float] = {
    "Partner": 650.0,
    "Manager": 450.0,
    "Consultant": 300.0,
    "Analyst": 180.0,
}

#: Fully loaded internal cost of one delivered hour by grade, in US dollars.
#: Cost sits at roughly 50-60% of the rate card, which is what makes an hour
#: delivered but not billed immediately margin-negative.
COST_RATES: dict[str, float] = {
    "Partner": 340.0,
    "Manager": 230.0,
    "Consultant": 160.0,
    "Analyst": 108.0,
}

#: Share of an engagement's delivered hours by grade, split by engagement size
#: tier. Smaller engagements are staffed more senior, so their cost per hour is
#: structurally higher.
LEVEL_MIX_BY_TIER: dict[str, dict[str, float]] = {
    "small": {"Partner": 0.14, "Manager": 0.32, "Consultant": 0.36, "Analyst": 0.18},
    "medium": {"Partner": 0.05, "Manager": 0.23, "Consultant": 0.46, "Analyst": 0.26},
    "large": {"Partner": 0.02, "Manager": 0.19, "Consultant": 0.50, "Analyst": 0.29},
}

#: Mean overrun factor (delivered hours / budgeted hours) by contract type.
#: Time-and-materials work is delivered at budget; fixed-fee work overruns.
OVERRUN_MEAN_BY_CONTRACT: dict[str, float] = {
    "time-and-materials": 1.00,
    "fixed-fee": 1.12,
    "retainer": 1.07,
}

#: Dispersion of the overrun factor by contract type. Fixed-fee work carries the
#: widest spread, which is why its realisation distribution has a long left tail.
OVERRUN_SIGMA_BY_CONTRACT: dict[str, float] = {
    "time-and-materials": 0.07,
    "fixed-fee": 0.12,
    "retainer": 0.10,
}

#: Mean negotiated reduction off the rate card by service line.
DISCOUNT_MEAN_BY_SERVICE_LINE: dict[str, float] = {
    "Strategy": 0.070,
    "Operations": 0.085,
    "Digital": 0.100,
    "Risk": 0.120,
}

#: Share of a billing that is invoiced but never collected, by contract type.
WRITE_OFF_MEAN_BY_CONTRACT: dict[str, float] = {
    "time-and-materials": 0.025,
    "fixed-fee": 0.045,
    "retainer": 0.030,
}

#: Non-labour delivery cost as a share of billings, by service line. Digital
#: work carries the heaviest technology and subcontractor load.
DELIVERY_COST_PCT_BY_SERVICE_LINE: dict[str, float] = {
    "Strategy": 0.060,
    "Operations": 0.100,
    "Risk": 0.090,
    "Digital": 0.160,
}

#: Share of the budget overrun that is un-invoiceable on fixed-fee and retainer
#: work. The client pays the agreed fee, not the hours actually consumed.
UNBILLABLE_OVERRUN_SHARE: dict[str, float] = {
    "fixed-fee": 0.90,
    "retainer": 0.55,
    "time-and-materials": 0.00,
}

#: Mean unsubmitted (and therefore unbillable) share of delivered time on
#: time-and-materials work.
TM_UNSUBMITTED_MEAN: float = 0.030

#: Minimum non-labour delivery cost charged to any engagement, in US dollars.
#: This mobilisation floor is what makes very small engagements unprofitable.
MOBILISATION_COST_MEDIAN: float = 16000.0

#: Log-normal dispersion of the mobilisation floor.
MOBILISATION_COST_SIGMA: float = 0.55

#: Chargeable hours a consultant makes available in a full year, after holiday,
#: training and internal time.
AVAILABLE_HOURS_PER_YEAR: float = 1900.0

#: Firm-wide target for billable hours as a share of available hours. The
#: generator calibrates engagement sizes to land close to this figure.
UTILIZATION_TARGET: float = 0.76

#: Size of the synthetic firm.
N_CONSULTANTS: int = 90
N_CLIENTS: int = 60
N_ENGAGEMENTS: int = 200

#: Share of consultants already on the payroll when the analysis window opens.
SHARE_ESTABLISHED_CONSULTANTS: float = 0.80

#: Shape parameter of the Pareto draw used for client revenue weight. Lower
#: values concentrate more revenue in fewer clients.
CLIENT_CONCENTRATION_SHAPE: float = 1.15

#: Log-normal dispersion of engagement size.
ENGAGEMENT_SIZE_SIGMA: float = 0.85

#: Quantile boundaries that split engagements into small / medium / large tiers.
ENGAGEMENT_TIER_QUANTILES: tuple[float, float] = (0.35, 0.80)

#: First and last day of the analysis window, inclusive.
ANALYSIS_START: date = date(2023, 1, 1)
ANALYSIS_END: date = date(2025, 12, 31)

#: The most recent complete fiscal year. The headline question - "we billed $X
#: this year" - is answered against this year.
HEADLINE_YEAR: int = 2025

#: Cumulative margin share that closes each client tier: A carries the first
#: half of firm margin, B the next 30 points, C the remainder.
TIER_A_CUMULATIVE_CUT: float = 0.50
TIER_B_CUMULATIVE_CUT: float = 0.80

#: Realisation points recovered in the sensitivity analysis.
SENSITIVITY_POINTS: tuple[float, ...] = (2.0, 5.0, 10.0)

#: Share of engagements treated as the "bottom quartile" on realisation.
SENSITIVITY_BOTTOM_QUANTILE: float = 0.25

#: Ordered stages of the margin-leakage waterfall. The first entry is the
#: starting value; every later entry is subtracted from it.
WATERFALL_STAGES: tuple[str, ...] = (
    "Standard value",
    "Discounts",
    "Unbilled time",
    "Write-offs",
    "Direct delivery cost",
    "Consultant cost",
    "Actual margin",
)

#: DPI used for every figure written to ``reports/figures``.
FIGURE_DPI: int = 300

#: Figure size in inches, shared by all report figures.
FIGURE_SIZE: tuple[float, float] = (12.0, 7.0)


@dataclass(frozen=True)
class Settings:
    """Resolved paths and modelling constants for one pipeline run.

    Attributes:
        project_root: Directory containing ``cli.py``.
        random_seed: Seed handed to the generator's random number source.
        headline_year: Fiscal year used for the headline question.
    """

    #: Directory containing ``cli.py``.
    project_root: Path = PROJECT_ROOT
    #: Seed handed to every pseudo-random draw in the generator.
    random_seed: int = RANDOM_SEED
    #: Fiscal year used to answer the headline question.
    headline_year: int = HEADLINE_YEAR
    #: Extra directories created on first access, relative to ``project_root``.
    _managed_dirs: tuple[str, ...] = field(
        default=("data/raw", "reports/figures", "reports/tables"),
        repr=False,
    )

    @property
    def data_dir(self) -> Path:
        """Return the top-level data directory."""
        return self.project_root / "data"

    @property
    def raw_dir(self) -> Path:
        """Return the directory holding generated CSVs."""
        return self.data_dir / "raw"

    @property
    def reports_dir(self) -> Path:
        """Return the directory holding generated reports."""
        return self.project_root / "reports"

    @property
    def figures_dir(self) -> Path:
        """Return the directory holding generated figures."""
        return self.reports_dir / "figures"

    @property
    def tables_dir(self) -> Path:
        """Return the directory holding generated CSV tables."""
        return self.reports_dir / "tables"

    @property
    def docs_dir(self) -> Path:
        """Return the directory holding narrative documentation."""
        return self.project_root / "docs"

    @property
    def app_dir(self) -> Path:
        """Return the directory holding the Streamlit application."""
        return self.project_root / "app"

    @property
    def metrics_path(self) -> Path:
        """Return the path of the serialised metrics file."""
        return self.reports_dir / "metrics.json"

    def ensure_directories(self) -> None:
        """Create every managed output directory if it does not exist.

        Returns:
            None. The call is idempotent.
        """
        for relative in self._managed_dirs:
            (self.project_root / relative).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Returns:
        The cached :class:`Settings` instance, with output directories created.
    """
    settings = Settings()
    settings.ensure_directories()
    return settings
