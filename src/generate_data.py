"""Synthetic engagement-economics generator for the profitability model.

The firm modelled here is fictional. Nothing in ``data/raw`` describes a real
client, a real consultant or a real engagement. The generator exists so that the
analysis, the report and the application can be demonstrated end to end on data
whose every structural property is known and documented.

The output is four CSVs written to ``data/raw``:

``consultants.csv``
    One row per consultant: grade, home service line, rate-card value per hour,
    internal cost per hour, hire date and hours made available inside the
    analysis window.
``engagements.csv``
    One row per engagement: client, industry, service line, contract type,
    start and end dates, budgeted hours, budgeted fee and status.
``timesheets.csv``
    One row per engagement x consultant x month, carrying the grade and the
    rate-card value applied to those hours.
``billing.csv``
    One row per engagement: rate-card value, discount, un-invoiceable time,
    billed amount, write-off, collected amount and non-labour delivery cost.

Behavioural assumptions
-----------------------
Every number below is a modelling choice, not an observation. They are listed
in one place so a reviewer can challenge any of them individually.

1. **Firm shape.** 90 consultants across four grades and four service lines,
   60 clients, and 200 engagements. 80% of consultants are already on the
   payroll when the window opens; the rest are hired during it, which is why
   their available hours are lower. Every client on the roster is dealt one
   engagement before the remaining work is allocated, so no account is dormant.

2. **Window scoping.** The analysis window is January 2023 to December 2025. The
   firm is a going concern, so engagements may start before the window opens or
   run past the month it closes. Only the months that fall inside the window are
   recorded, and each engagement's budget - and therefore its fee, its overrun
   and its billing row - is scoped to that same span. Budget and delivered hours
   always describe the same months.

3. **Client concentration.** Client revenue weight follows a Pareto(1.15) draw.
   A handful of clients therefore dominate billings and margin, which is the
   precondition for the Pareto analysis in the report.

4. **Engagement size.** Delivery intensity per active month is log-normal with
   sigma 0.85 and a floor at 20% of the firm average, so a longer engagement is
   a larger engagement. Intensities are then scaled by a single constant so that
   firm-wide delivered hours land near 76% of available hours. Utilisation is
   therefore an output of the calibration, not an input to it.

5. **Engagement timing.** Start months are assigned by systematic sampling: one
   evenly spaced quantile per engagement, shuffled, drawn from a start-month
   distribution that grows across the window. Independent draws would let two or
   three very large engagements land in the same year and swamp it, because size
   is heavy-tailed. The result is delivered hours of roughly 24%, 35% and 41% in
   2023, 2024 and 2025.

6. **Overrun by contract type.** Time-and-materials work is delivered at budget
   (mean factor 1.00). Retainers overrun by about 7%. Fixed-fee work overruns by
   about 12%, with the widest dispersion of the three. This is the single largest
   structural driver of realisation loss in the dataset.

7. **Discount by service line.** Negotiated reductions off the rate card average
   7.0% in Strategy, 8.5% in Operations, 10.0% in Digital and 12.0% in Risk, each
   with a log-normal spread and clipped to a 2-35% band.

8. **Unbilled time.** On fixed-fee work, 90% of the hours delivered beyond the
   budgeted envelope are un-invoiceable - the client pays the agreed fee. On
   retainers the equivalent figure is 55%. On time-and-materials, unbilled time
   is a small unsubmitted share (mean 3.0%) of delivered value rather than an
   overrun effect.

9. **Write-offs.** Between 2.5% and 4.5% of billings are invoiced but never
   collected, depending on contract type, with a heavy right tail and an extra
   5% chance of a disputed invoice so that a minority of engagements carry a
   genuinely bad debt.

10. **Direct delivery cost.** Non-labour delivery cost - subcontractors,
    technology, travel, data licences - scales with billings and varies by
    service line: highest in Digital (16%), lowest in Strategy (6%).

11. **Mobilisation floor.** Every engagement carries a minimum non-labour cost
    draw (median $22,000, log-normal). Small engagements cannot cover it. This,
    not the rate card, is what makes them loss-making.

12. **Seniority mix by size.** Small engagements are staffed more senior
    (12% Partner hours) than large ones (4% Partner hours), so their cost per
    delivered hour is higher at the same rate card.

13. **Loss-making minority.** The combination of assumptions 6, 10 and 11 puts a
    minority of engagements into negative margin, concentrated in small
    fixed-fee work and in the lower-realisation service lines.

14. **Money follows hours.** Every engagement-level money figure is derived from
    the timesheet it is attached to, never from the plan. Rounding is reconciled
    inside each engagement so that timesheet hours sum exactly to the delivered
    hours the billing row was computed from.

Usage::

    from src.generate_data import generate_all

    frames = generate_all()
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta

import numpy as np
import pandas as pd
from loguru import logger

from src.config import (
    ANALYSIS_END,
    ANALYSIS_START,
    AVAILABLE_HOURS_PER_YEAR,
    CLIENT_CONCENTRATION_SHAPE,
    COST_RATES,
    DELIVERY_COST_PCT_BY_SERVICE_LINE,
    DISCOUNT_MEAN_BY_SERVICE_LINE,
    ENGAGEMENT_SIZE_SIGMA,
    ENGAGEMENT_TIER_QUANTILES,
    LEVEL_MIX_BY_TIER,
    LEVELS,
    MOBILISATION_COST_MEDIAN,
    MOBILISATION_COST_SIGMA,
    N_CLIENTS,
    N_CONSULTANTS,
    N_ENGAGEMENTS,
    OVERRUN_MEAN_BY_CONTRACT,
    OVERRUN_SIGMA_BY_CONTRACT,
    SERVICE_LINES,
    SHARE_ESTABLISHED_CONSULTANTS,
    STANDARD_RATES,
    TM_UNSUBMITTED_MEAN,
    UNBILLABLE_OVERRUN_SHARE,
    UTILIZATION_TARGET,
    WRITE_OFF_MEAN_BY_CONTRACT,
    Settings,
    get_settings,
)

#: Grade headcount for the synthetic firm. Sums to ``N_CONSULTANTS``.
_LEVEL_HEADCOUNT: dict[str, int] = {
    "Partner": 8,
    "Manager": 20,
    "Consultant": 38,
    "Analyst": 24,
}

#: Contract-type mix. Roughly half the firm's work is time-and-materials.
_CONTRACT_MIX: dict[str, float] = {
    "time-and-materials": 0.45,
    "fixed-fee": 0.35,
    "retainer": 0.20,
}

#: Industries assigned to clients.
_INDUSTRIES: tuple[str, ...] = (
    "Energy",
    "Financial Services",
    "Healthcare",
    "Manufacturing",
    "Retail",
    "Technology",
    "Public Sector",
    "Telecoms",
    "Transport",
    "Utilities",
)

#: Name fragments used to build synthetic client names.
_CLIENT_PREFIXES: tuple[str, ...] = (
    "Meridian",
    "Halcyon",
    "Northwind",
    "Ardent",
    "Blue Harbour",
    "Cobalt",
    "Draymore",
    "Eastgate",
    "Fairweather",
    "Granite",
    "Hollowbrook",
    "Ironvale",
    "Juniper",
    "Kestrel",
    "Larkspur",
    "Marlow",
    "Nightingale",
    "Orchard",
    "Pinnacle",
    "Quarry",
    "Redwood",
    "Saltmarsh",
    "Thornbury",
    "Uplands",
    "Verity",
    "Westbrook",
    "Yarrow",
    "Zenith",
)

#: Name fragments appended to client names.
_CLIENT_SUFFIXES: tuple[str, ...] = (
    "Group",
    "Holdings",
    "Industries",
    "Partners",
    "Systems",
    "Logistics",
    "Capital",
    "Energy",
    "Health",
    "Retail",
    "Networks",
    "Resources",
)

#: Synthetic consultant given names.
_FIRST_NAMES: tuple[str, ...] = (
    "Adrian",
    "Beatriz",
    "Callum",
    "Derya",
    "Elena",
    "Farid",
    "Grace",
    "Hugo",
    "Iris",
    "Jonas",
    "Karina",
    "Liam",
    "Maya",
    "Nadia",
    "Omar",
    "Priya",
    "Quentin",
    "Rosa",
    "Sami",
    "Tessa",
    "Ulrich",
    "Vera",
    "Wren",
    "Xavier",
    "Yusuf",
    "Zara",
    "Anika",
    "Bruno",
    "Chiara",
    "Dmitri",
)

#: Synthetic consultant family names.
_LAST_NAMES: tuple[str, ...] = (
    "Aksoy",
    "Bianchi",
    "Carter",
    "Demir",
    "Eriksen",
    "Fontaine",
    "Gallagher",
    "Hoffmann",
    "Ibarra",
    "Jensen",
    "Kowalski",
    "Lindqvist",
    "Moreau",
    "Nakamura",
    "Okafor",
    "Petrov",
    "Quinn",
    "Rasmussen",
    "Silva",
    "Tremblay",
    "Ueda",
    "Vasquez",
    "Whitfield",
    "Xu",
    "Yilmaz",
    "Zielinski",
    "Barros",
    "Novak",
    "Sorensen",
    "Marchetti",
)

#: Team size band (inclusive) by engagement size tier.
_TEAM_SIZE_BY_TIER: dict[str, tuple[int, int]] = {
    "small": (3, 5),
    "medium": (5, 8),
    "large": (8, 14),
}

#: Length of the analysis window in whole months.
_WINDOW_MONTHS: int = (
    (ANALYSIS_END.year - ANALYSIS_START.year) * 12
    + ANALYSIS_END.month
    - ANALYSIS_START.month
    + 1
)

#: Shortest and longest engagement duration, in months.
_MIN_ENGAGEMENT_MONTHS: int = 3
_MAX_ENGAGEMENT_MONTHS: int = 15

#: Fewest in-window months any engagement must contribute, so that the budget
#: scoped to the window is never a one-month sliver.
_MIN_IN_WINDOW_MONTHS: int = 2

#: Earliest month in which an engagement may start. The firm is a going concern,
#: so work is already in progress when the window opens; only the portion of each
#: engagement that falls inside the window is recorded. The floor is set so that
#: even the longest engagement can start early enough to contribute its minimum.
_FIRST_START_MONTH_INDEX: int = -(_MAX_ENGAGEMENT_MONTHS - _MIN_IN_WINDOW_MONTHS)

#: Latest month in which an engagement may start. Engagements that begin here run
#: past the window close and are clipped, exactly as those already in flight when
#: it opened are clipped at the other end.
_LAST_START_MONTH_INDEX: int = _WINDOW_MONTHS - _MIN_ENGAGEMENT_MONTHS

#: Smallest delivery intensity, as a multiple of the firm's average engagement.
#: Expressed per month rather than as an absolute hour floor so that an
#: engagement clipped by the window is not silently resized.
_MIN_SIZE_INTENSITY: float = 0.20

#: Ratio between the chance of starting an engagement in the last eligible month
#: and the first. The firm grows across the window, so later fiscal years carry
#: more work than earlier ones: roughly 24%, 35% and 41% of delivered hours fall
#: in 2023, 2024 and 2025 respectively.
_START_GROWTH: float = 1.5

#: Month offsets from which an engagement may start.
_START_OFFSETS: np.ndarray = np.arange(
    _FIRST_START_MONTH_INDEX, _LAST_START_MONTH_INDEX + 1
)

#: Unnormalised growth weight attached to each eligible start month.
_START_OFFSET_WEIGHTS: np.ndarray = np.power(
    _START_GROWTH, (_START_OFFSETS - _START_OFFSETS[0]) / (_START_OFFSETS[-1] - _START_OFFSETS[0])
)


def _month_index_to_key(month_index: int) -> str:
    """Convert a month offset from the window start into a ``YYYY-MM`` key.

    Args:
        month_index: Whole months since ``ANALYSIS_START``; ``0`` is January 2023.

    Returns:
        The corresponding month key.
    """
    year = ANALYSIS_START.year + (ANALYSIS_START.month - 1 + month_index) // 12
    month = (ANALYSIS_START.month - 1 + month_index) % 12 + 1
    return f"{year:04d}-{month:02d}"


def _month_index_to_date(month_index: int) -> date:
    """Return the first day of the month at a given offset from the window start.

    Args:
        month_index: Whole months since ``ANALYSIS_START``.

    Returns:
        The first calendar day of that month.
    """
    key = _month_index_to_key(month_index)
    year, month = (int(part) for part in key.split("-"))
    return date(year, month, 1)


def _stratified_quantiles(weights: np.ndarray, offset: float) -> np.ndarray:
    """Spread units across the quantile axis in proportion to their weight.

    Each unit is placed at the midpoint of its own slice of the cumulative weight
    distribution, with a single random origin so the arrangement is not identical
    on every run. This is Madow's systematic sampling scheme.

    Args:
        weights: Positive relative weight of each unit.
        offset: Starting position in ``[0, 1)``, normally a single random draw.

    Returns:
        Array of quantiles in ``[0, 1)``, one per unit, in the input order.
    """
    order = np.argsort(-weights, kind="stable")
    ordered = weights[order]
    cumulative = np.cumsum(ordered)
    total = cumulative[-1]
    positions = (cumulative - ordered + offset * ordered) / total

    quantiles = np.empty_like(weights, dtype=float)
    quantiles[order] = positions
    return np.clip(quantiles, 0.0, 1.0 - 1e-12)


def _weighted_quantile_offset(low: int, high: int, quantile: float) -> int:
    """Pick the start month sitting at a given quantile of the weighted range.

    Args:
        low: Lowest eligible start month offset, inclusive.
        high: Highest eligible start month offset, inclusive.
        quantile: Position in ``(0, 1)`` of the growth-weighted distribution.

    Returns:
        The month offset at that quantile.
    """
    eligible = (_START_OFFSETS >= low) & (_START_OFFSETS <= high)
    offsets = _START_OFFSETS[eligible]
    weights = _START_OFFSET_WEIGHTS[eligible]
    cumulative = np.cumsum(weights)
    cumulative = cumulative / cumulative[-1]
    position = int(np.searchsorted(cumulative, quantile, side="left"))
    return int(offsets[min(position, len(offsets) - 1)])


def _window_month_indices(start_index: int, duration_months: int) -> list[int]:
    """List the engagement's month offsets that fall inside the analysis window.

    Args:
        start_index: Month offset of the engagement start, relative to
            ``ANALYSIS_START``. May be negative when the engagement was already
            running as the window opened.
        duration_months: Length of the engagement in whole months.

    Returns:
        Ascending month offsets in ``[0, _WINDOW_MONTHS)``. Empty when the
        engagement falls entirely outside the window.
    """
    return [
        index
        for index in range(start_index, start_index + duration_months)
        if 0 <= index < _WINDOW_MONTHS
    ]


def _month_starts(start: date, end: date) -> list[str]:
    """List every calendar month touched by an inclusive date range.

    Args:
        start: First day of the range.
        end: Last day of the range.

    Returns:
        Month keys in ``YYYY-MM`` form, ascending, covering ``start`` through
        ``end`` inclusive.
    """
    months: list[str] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return months


def _month_end(month_key: str) -> date:
    """Return the last calendar day of a ``YYYY-MM`` month key.

    Args:
        month_key: Month in ``YYYY-MM`` form.

    Returns:
        The final day of that month as a :class:`datetime.date`.
    """
    year, month = (int(part) for part in month_key.split("-"))
    return date(year, month, calendar.monthrange(year, month)[1])


def _available_hours(hire_date: date, window_start: date, window_end: date) -> float:
    """Count the chargeable hours a consultant makes available inside a window.

    A consultant accrues ``AVAILABLE_HOURS_PER_YEAR`` pro rata for every month
    between their hire date and the end of the window, clipped to the window.

    Args:
        hire_date: Date the consultant joined the firm.
        window_start: First day of the analysis window.
        window_end: Last day of the analysis window.

    Returns:
        Available hours as a float; ``0.0`` when the hire date is after the
        window closes.
    """
    if hire_date > window_end:
        return 0.0
    effective_start = max(hire_date, window_start)
    months = _month_starts(effective_start, window_end)
    return len(months) * AVAILABLE_HOURS_PER_YEAR / 12.0


def _distribute_exact(total: float, weights: np.ndarray, step: float = 0.5) -> np.ndarray:
    """Split a total into ``step``-aligned parts proportional to weights.

    The parts sum *exactly* to ``total`` (itself snapped to ``step``); any
    rounding residue is absorbed by the largest part. This is what lets the
    tests assert that timesheet hours reconcile to the billing row.

    Args:
        total: Amount to split. Snapped to the nearest multiple of ``step``.
        weights: Non-negative relative weights, one per part.
        step: Granularity of the resulting parts, in hours.

    Returns:
        Array of parts, each a multiple of ``step``, summing to the snapped
        total.
    """
    snapped_total = round(total / step) * step
    units = np.maximum(np.round(snapped_total / step).astype(int), len(weights))
    safe_weights = np.where(weights > 0, weights, 0.0)
    if safe_weights.sum() == 0:
        safe_weights = np.ones_like(safe_weights)
    raw = safe_weights / safe_weights.sum() * units
    whole = np.floor(raw).astype(int)
    whole = np.maximum(whole, 1)
    residue = units - int(whole.sum())
    if residue != 0:
        order = np.argsort(-(raw - np.floor(raw)))
        index = 0
        while residue != 0 and index < len(order) * 4:
            target = order[index % len(order)]
            delta = 1 if residue > 0 else -1
            if whole[target] + delta >= 1:
                whole[target] += delta
                residue -= delta
            index += 1
    return whole.astype(float) * step


def generate_consultants(settings: Settings) -> pd.DataFrame:
    """Build the consultant roster.

    Args:
        settings: Resolved settings supplying the random seed.

    Returns:
        DataFrame with one row per consultant, including the hours they make
        available inside the analysis window.
    """
    rng = np.random.default_rng(settings.random_seed)
    records: list[dict[str, object]] = []
    consultant_id = 0

    established_cut = date(2022, 12, 31)
    window_days = (ANALYSIS_END - date(2023, 1, 1)).days

    for level in LEVELS:
        for _ in range(_LEVEL_HEADCOUNT[level]):
            consultant_id += 1
            if rng.random() < SHARE_ESTABLISHED_CONSULTANTS:
                hire_date = date(2013, 1, 1) + timedelta(
                    days=int(rng.integers(0, (established_cut - date(2013, 1, 1)).days))
                )
            else:
                hire_date = date(2023, 1, 1) + timedelta(
                    days=int(rng.integers(0, window_days))
                )

            standard_rate = round(
                STANDARD_RATES[level] * float(rng.normal(1.0, 0.04)), 2
            )
            cost_rate = round(COST_RATES[level] * float(rng.normal(1.0, 0.05)), 2)
            records.append(
                {
                    "consultant_id": f"C{consultant_id:03d}",
                    "consultant_name": (
                        f"{_FIRST_NAMES[int(rng.integers(0, len(_FIRST_NAMES)))]} "
                        f"{_LAST_NAMES[int(rng.integers(0, len(_LAST_NAMES)))]}"
                    ),
                    "level": level,
                    "service_line": SERVICE_LINES[
                        int(rng.integers(0, len(SERVICE_LINES)))
                    ],
                    "standard_rate": standard_rate,
                    "cost_rate": cost_rate,
                    "hire_date": hire_date.isoformat(),
                    "available_hours": round(
                        _available_hours(hire_date, ANALYSIS_START, ANALYSIS_END), 1
                    ),
                }
            )

    frame = pd.DataFrame.from_records(records)
    logger.debug("Generated {} consultants", len(frame))
    return frame


def generate_clients(settings: Settings) -> pd.DataFrame:
    """Build the client list with a deliberately uneven revenue weight.

    Args:
        settings: Resolved settings supplying the random seed.

    Returns:
        DataFrame with one row per client and a normalised ``revenue_weight``.
    """
    rng = np.random.default_rng(settings.random_seed + 1)
    used_names: set[str] = set()
    records: list[dict[str, object]] = []

    for index in range(1, N_CLIENTS + 1):
        while True:
            name = (
                f"{_CLIENT_PREFIXES[int(rng.integers(0, len(_CLIENT_PREFIXES)))]} "
                f"{_CLIENT_SUFFIXES[int(rng.integers(0, len(_CLIENT_SUFFIXES)))]}"
            )
            if name not in used_names:
                used_names.add(name)
                break
        records.append(
            {
                "client_id": f"CL{index:03d}",
                "client_name": name,
                "industry": _INDUSTRIES[int(rng.integers(0, len(_INDUSTRIES)))],
                "revenue_weight": float(rng.pareto(CLIENT_CONCENTRATION_SHAPE)) + 1.0,
            }
        )

    frame = pd.DataFrame.from_records(records)
    frame["revenue_weight"] = frame["revenue_weight"] / frame["revenue_weight"].sum()
    logger.debug("Generated {} clients", len(frame))
    return frame


def generate_engagements(
    settings: Settings,
    clients: pd.DataFrame,
) -> pd.DataFrame:
    """Build the engagement portfolio and calibrate its size distribution.

    Engagement sizes are drawn as relative weights, converted to budgeted hours
    by a single calibration constant, and then inflated by a contract-type
    overrun factor. The constant is chosen so that total delivered hours land on
    the firm's utilisation target.

    Args:
        settings: Resolved settings supplying the random seed.
        clients: Client frame carrying the ``revenue_weight`` used to assign
            engagements to clients.

    Returns:
        DataFrame with one row per engagement.
    """
    rng = np.random.default_rng(settings.random_seed + 2)

    total_available = float(generate_consultants(settings)["available_hours"].sum())
    target_delivered = total_available * UTILIZATION_TARGET

    client_ids = clients["client_id"].to_numpy()
    weights = clients["revenue_weight"].to_numpy()
    weights = weights / weights.sum()

    contract_types = list(_CONTRACT_MIX)
    contract_probs = np.array([_CONTRACT_MIX[key] for key in contract_types])

    raw_sizes = rng.lognormal(
        mean=-(ENGAGEMENT_SIZE_SIGMA**2) / 2.0,
        sigma=ENGAGEMENT_SIZE_SIGMA,
        size=N_ENGAGEMENTS,
    )

    # Drawn once and reused, so an engagement's overrun always matches its own
    # contract type.
    chosen_contracts = rng.choice(contract_types, size=N_ENGAGEMENTS, p=contract_probs)

    # One engagement per client is dealt out first, so every account on the
    # roster is live at least once. The remainder is drawn on revenue weight,
    # which is what produces the concentration the Pareto analysis measures.
    seed_clients = client_ids[rng.permutation(len(client_ids))]
    remainder = rng.choice(
        client_ids, size=max(N_ENGAGEMENTS - len(seed_clients), 0), p=weights
    )
    chosen_clients = np.concatenate([seed_clients, remainder])
    rng.shuffle(chosen_clients)
    chosen_lines = rng.choice(
        list(SERVICE_LINES), size=N_ENGAGEMENTS, p=np.full(4, 0.25)
    )

    overruns = np.array(
        [
            float(
                rng.lognormal(
                    mean=np.log(OVERRUN_MEAN_BY_CONTRACT[contract_type])
                    - OVERRUN_SIGMA_BY_CONTRACT[contract_type] ** 2 / 2.0,
                    sigma=OVERRUN_SIGMA_BY_CONTRACT[contract_type],
                )
            )
            for contract_type in chosen_contracts
        ]
    )

    # ``intensity`` is delivery per active month, so a longer engagement is a
    # larger engagement. Total hours scale with the months that fall inside the
    # window, which is also the span the budget covers.
    intensity = np.maximum(raw_sizes, _MIN_SIZE_INTENSITY)

    durations = rng.integers(
        _MIN_ENGAGEMENT_MONTHS, _MAX_ENGAGEMENT_MONTHS + 1, size=N_ENGAGEMENTS
    )
    # An engagement may start before the window opens or run past the month it
    # closes; only the months inside the window are recorded, and the budget is
    # scoped to match. Both ends are therefore clipped symmetrically.
    earliest = np.maximum(
        _FIRST_START_MONTH_INDEX, _MIN_IN_WINDOW_MONTHS - durations
    )
    latest = np.full(N_ENGAGEMENTS, _LAST_START_MONTH_INDEX)

    # Start months are assigned by systematic sampling proportional to size, not
    # by an independent draw per engagement. Engagement size is heavy-tailed, so
    # independent draws let two or three very large engagements land in the same
    # fiscal year and swamp it. Instead each engagement occupies a slice of the
    # quantile axis as wide as its own weight, so every slice of the window
    # carries a comparable volume of work while the weighted start-month
    # distribution is preserved.
    quantiles = _stratified_quantiles(intensity * durations * overruns, rng.random())
    start_indices = np.array(
        [
            _weighted_quantile_offset(low, high, quantile)
            for (low, high), quantile in zip(zip(earliest, latest), quantiles)
        ]
    )

    in_window_months = np.array(
        [
            len(_window_month_indices(int(start), int(duration)))
            for start, duration in zip(start_indices, durations)
        ],
        dtype=float,
    )

    baseline = target_delivered / float((intensity * overruns * in_window_months).sum())
    blended_rate = sum(
        STANDARD_RATES[level] * LEVEL_MIX_BY_TIER["medium"][level] for level in LEVELS
    )

    records: list[dict[str, object]] = []
    for index in range(N_ENGAGEMENTS):
        month_index = int(start_indices[index])
        duration_months = int(durations[index])
        start = _month_index_to_date(month_index)
        end = _month_end(_month_index_to_key(month_index + duration_months - 1))

        # Budget covers only the portion of the engagement that falls inside the
        # analysis window, so budget and delivered hours describe the same span.
        budgeted_hours = max(
            5.0,
            round(baseline * intensity[index] * in_window_months[index] / 5.0) * 5.0,
        )
        delivered_hours = round(budgeted_hours * overruns[index] * 2.0) / 2.0
        budgeted_fee = round(budgeted_hours * blended_rate * 0.90, 2)

        records.append(
            {
                "engagement_id": f"E{index + 1:04d}",
                "client_id": str(chosen_clients[index]),
                "service_line": str(chosen_lines[index]),
                "contract_type": str(chosen_contracts[index]),
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "budgeted_hours": budgeted_hours,
                "delivered_hours_plan": delivered_hours,
                "budgeted_fee": budgeted_fee,
                "status": "Completed" if end <= date(2025, 9, 30) else "Active",
                "start_month_index": month_index,
                "duration_months": duration_months,
            }
        )

    frame = pd.DataFrame.from_records(records)
    frame = frame.merge(
        clients[["client_id", "client_name", "industry"]], on="client_id", how="left"
    )

    quantiles = frame["budgeted_hours"].quantile(list(ENGAGEMENT_TIER_QUANTILES))
    frame["size_tier"] = np.where(
        frame["budgeted_hours"] <= quantiles.iloc[0],
        "small",
        np.where(frame["budgeted_hours"] <= quantiles.iloc[1], "medium", "large"),
    )

    logger.debug(
        "Generated {} engagements; planned delivered hours {:.0f}",
        len(frame),
        frame["delivered_hours_plan"].sum(),
    )
    return frame


#: How much less appealing a consultant from another service line is than one
#: from the engagement's own practice. Staffing stays local unless the local
#: bench has roughly three times less room than an outsider.
_FOREIGN_BENCH_PENALTY = 0.35

#: Jitter applied to each candidate's remaining capacity when staffing, so the
#: same names are not picked for every engagement that has room for them.
_STAFFING_JITTER_SIGMA = 0.35

#: Hours held back per person when staffing, so that rounding a share to the
#: nearest half hour cannot push someone past their own availability.
_STAFFING_ROUNDING_RESERVE = 0.5


def _staffing_room(
    record: dict[str, object],
    months: list[str],
    remaining_total: dict[str, float],
    remaining_month: dict[str, dict[str, float]],
) -> float:
    """Return the hours a consultant can still give inside an engagement window.

    A consultant's capacity is bounded twice over: by the hours they have left
    in total, and by the hours they have left in each month the engagement
    touches. An engagement can only book the smaller of the two.

    Args:
        record: Consultant roster row.
        months: Month keys the engagement spans.
        remaining_total: Unbooked hours by consultant.
        remaining_month: Unbooked hours by consultant and month.

    Returns:
        Hours available inside the window; ``0.0`` when the consultant had not
        joined yet or is already fully booked.
    """
    identifier = str(record["consultant_id"])
    monthly = remaining_month[identifier]
    return min(
        remaining_total[identifier],
        sum(monthly.get(month, 0.0) for month in months),
    )


def _monthly_capacity(
    record: dict[str, object],
) -> dict[str, float]:
    """Spread a consultant's available hours evenly across the months they work.

    Args:
        record: Consultant roster row carrying ``hire_date`` and
            ``available_hours``.

    Returns:
        Mapping of month key to the hours available in that month.
    """
    months = _month_starts(
        max(date.fromisoformat(str(record["hire_date"])), ANALYSIS_START),
        ANALYSIS_END,
    )
    if not months:
        return {}
    per_month = float(record["available_hours"]) / len(months)
    return {month: per_month for month in months}


def generate_timesheets(
    settings: Settings,
    consultants: pd.DataFrame,
    engagements: pd.DataFrame,
) -> pd.DataFrame:
    """Expand the engagement portfolio into monthly timesheet rows.

    Each engagement is staffed from its own service line, weighted towards the
    seniority mix appropriate to its size, and never booked beyond what the
    people on it have left to sell. Delivered hours are split across the team
    and then across the months the engagement spans, with the split reconciled
    exactly at each step.

    Capacity is the binding constraint. Without it the same partner gets drawn
    for a dozen engagements and the roster ends up with people booked at four
    times their availability, which makes the utilisation view meaningless.

    Args:
        settings: Resolved settings supplying the random seed.
        consultants: Consultant roster carrying ``available_hours``.
        engagements: Engagement portfolio carrying ``delivered_hours_plan``.

    Returns:
        DataFrame with one row per engagement x consultant x month.
    """
    rng = np.random.default_rng(settings.random_seed + 3)
    consultant_records = consultants.to_dict("records")
    by_line: dict[str, list[dict[str, object]]] = {}
    for record in consultant_records:
        by_line.setdefault(str(record["service_line"]), []).append(record)

    # Hours each consultant has left to sell, in total and month by month. Both
    # are drawn down as engagements are staffed, in start-date order, so an
    # earlier engagement cannot be crowded out by a later one.
    remaining_total: dict[str, float] = {}
    remaining_month: dict[str, dict[str, float]] = {}
    for record in consultant_records:
        identifier = str(record["consultant_id"])
        remaining_total[identifier] = float(record["available_hours"])
        remaining_month[identifier] = _monthly_capacity(record)

    rows: list[dict[str, object]] = []
    overflow_hours = 0.0

    for engagement in engagements.to_dict("records"):
        tier = str(engagement["size_tier"])
        service_line = str(engagement["service_line"])
        local_ids = {
            str(record["consultant_id"])
            for record in (by_line.get(service_line) or consultant_records)
        }
        mix = LEVEL_MIX_BY_TIER[tier]

        # Every engagement carries at least one person per grade, so the mix
        # below can be read directly as each grade's share of the delivered
        # hours. Any further members are drawn on the mix, which is what makes
        # large engagements bottom-heavy without changing the hours split.
        low, high = _TEAM_SIZE_BY_TIER[tier]
        team_size = max(int(rng.integers(low, high + 1)), len(LEVELS))
        extra_levels = rng.choice(
            LEVELS,
            size=team_size - len(LEVELS),
            p=np.array([mix[level] for level in LEVELS]),
        )

        months = [
            _month_index_to_key(index)
            for index in _window_month_indices(
                int(engagement["start_month_index"]),
                int(engagement["duration_months"]),
            )
        ]
        if len(months) == 1:
            month_weights = np.ones(1)
        else:
            ramp = np.linspace(0.0, np.pi, len(months))
            month_weights = 0.6 + 0.4 * np.sin(ramp)
        month_weights = month_weights * rng.lognormal(
            mean=0.0, sigma=0.18, size=len(months)
        )

        delivered_hours = float(engagement["delivered_hours_plan"])
        members: list[dict[str, object]] = []
        member_hours: list[float] = []

        for level in LEVELS:
            level_hours = mix[level] * delivered_hours
            if level_hours <= 0:
                continue
            headcount = 1 + int(np.count_nonzero(extra_levels == level))

            # Rank the whole grade by how much room each person has, scaled by
            # a local-bench preference and a random draw. Someone from another
            # service line is only taken when the local bench is close to full.
            bench = [record for record in consultant_records if record["level"] == level]
            jitter = rng.lognormal(
                mean=0.0, sigma=_STAFFING_JITTER_SIGMA, size=len(bench)
            )
            scored = sorted(
                zip(bench, jitter, strict=True),
                key=lambda pair: -(
                    (
                        _staffing_room(pair[0], months, remaining_total, remaining_month)
                        - _STAFFING_ROUNDING_RESERVE
                    )
                    * pair[1]
                    * (
                        1.0
                        if str(pair[0]["consultant_id"]) in local_ids
                        else _FOREIGN_BENCH_PENALTY
                    )
                ),
            )

            # Take the people the mix asks for, and keep taking more until the
            # grade's share of the hours has somewhere to go.
            team: list[tuple[dict[str, object], float]] = []
            room_total = 0.0
            for candidate, _ in scored:
                room = (
                    _staffing_room(candidate, months, remaining_total, remaining_month)
                    - _STAFFING_ROUNDING_RESERVE
                )
                if room <= 0:
                    continue
                if room_total >= level_hours and len(team) >= headcount:
                    break
                team.append((candidate, room))
                room_total += room

            if not team:
                overflow_hours += level_hours
                continue
            if room_total < level_hours:
                # The grade has no more room anywhere in the firm for this
                # window. Book what fits and carry the remainder on the same
                # people rather than dropping hours the engagement was planned
                # to deliver; the shortfall is logged below.
                overflow_hours += level_hours - room_total

            # The grade's share of the engagement is fixed by the mix; it is
            # only the split between people within the grade that varies. The
            # spare room each person has is what varies it.
            weights = np.array([room for _, room in team], dtype=float)
            for (member, _), hours in zip(
                team, _distribute_exact(level_hours, weights), strict=True
            ):
                members.append(member)
                member_hours.append(float(hours))

        for member, hours in zip(members, member_hours, strict=True):
            if hours <= 0:
                continue
            identifier = str(member["consultant_id"])
            monthly = remaining_month[identifier]
            per_month = _distribute_exact(float(hours), month_weights)
            for month_key, month_hours in zip(months, per_month, strict=True):
                if month_hours <= 0:
                    continue
                rows.append(
                    {
                        "engagement_id": engagement["engagement_id"],
                        "consultant_id": member["consultant_id"],
                        "month": month_key,
                        "level": member["level"],
                        "hours": float(month_hours),
                        "standard_rate": float(member["standard_rate"]),
                    }
                )
                remaining_total[identifier] -= float(month_hours)
                if month_key in monthly:
                    monthly[month_key] -= float(month_hours)

    frame = pd.DataFrame.from_records(rows)
    frame["year"] = frame["month"].str.slice(0, 4).astype(int)
    logger.debug("Generated {} timesheet rows", len(frame))
    if overflow_hours:
        logger.debug(
            "Booked {:.0f} hours beyond the roster's remaining capacity", overflow_hours
        )
    return frame


def derive_billing(
    settings: Settings,
    engagements: pd.DataFrame,
    timesheets: pd.DataFrame,
    consultants: pd.DataFrame,
) -> pd.DataFrame:
    """Derive the billing row for every engagement from its own timesheet.

    Nothing here is read from the plan. Rate-card value and delivery cost are
    aggregated straight from timesheet hours, so the billing row and the
    timesheet can never disagree.

    Args:
        settings: Resolved settings supplying the random seed.
        engagements: Engagement portfolio.
        timesheets: Timesheet rows produced by :func:`generate_timesheets`.
        consultants: Consultant roster carrying internal cost rates.

    Returns:
        DataFrame with one billing row per engagement.
    """
    rng = np.random.default_rng(settings.random_seed + 4)

    cost_lookup = dict(
        zip(consultants["consultant_id"], consultants["cost_rate"], strict=True)
    )
    enriched = timesheets.assign(
        standard_value=timesheets["hours"] * timesheets["standard_rate"],
        hour_cost=timesheets["hours"]
        * timesheets["consultant_id"].map(cost_lookup).astype(float),
    )

    totals = enriched.groupby("engagement_id", as_index=False).agg(
        delivered_hours=("hours", "sum"),
        standard_value=("standard_value", "sum"),
        consultant_cost=("hour_cost", "sum"),
    )
    totals["standard_value"] = totals["standard_value"].round(2)
    totals["consultant_cost"] = totals["consultant_cost"].round(2)

    frame = engagements.merge(totals, on="engagement_id", how="left")
    size = len(frame)

    discount_pct = np.array(
        [
            float(
                rng.lognormal(
                    mean=np.log(DISCOUNT_MEAN_BY_SERVICE_LINE[line]),
                    sigma=0.28,
                )
            )
            for line in frame["service_line"]
        ]
    )
    frame["discount_pct"] = np.clip(discount_pct, 0.02, 0.35)
    frame["discount_amount"] = (frame["standard_value"] * frame["discount_pct"]).round(2)

    overrun_hours = np.maximum(
        frame["delivered_hours"] - frame["budgeted_hours"], 0.0
    )
    blended_rate = np.where(
        frame["delivered_hours"] > 0,
        frame["standard_value"] / frame["delivered_hours"],
        0.0,
    )
    unbillable_share = frame["contract_type"].map(UNBILLABLE_OVERRUN_SHARE).astype(float)
    overrun_value = overrun_hours * blended_rate * unbillable_share
    tm_share = frame["contract_type"].eq("time-and-materials")
    tm_unsubmitted = (
        np.array(
            [
                float(rng.lognormal(mean=np.log(TM_UNSUBMITTED_MEAN), sigma=0.45))
                for _ in range(size)
            ]
        )
        * frame["standard_value"].to_numpy()
    )
    unbilled = np.where(tm_share, tm_unsubmitted, overrun_value.to_numpy())
    value_after_discount = frame["standard_value"] - frame["discount_amount"]
    frame["unbilled_value"] = np.minimum(unbilled, value_after_discount * 0.85).round(2)

    frame["billed_amount"] = (
        frame["standard_value"] - frame["discount_amount"] - frame["unbilled_value"]
    ).clip(lower=0.0)

    write_off_pct = np.array(
        [
            float(
                rng.lognormal(
                    mean=np.log(WRITE_OFF_MEAN_BY_CONTRACT[contract_type]), sigma=0.90
                )
            )
            for contract_type in frame["contract_type"]
        ]
    )
    dispute = rng.random(size) < 0.05
    write_off_pct = np.where(dispute, write_off_pct + 0.12, write_off_pct)
    frame["write_off_pct"] = np.clip(write_off_pct, 0.0, 0.45)
    frame["write_off_amount"] = (frame["billed_amount"] * frame["write_off_pct"]).round(2)
    frame["collected_amount"] = frame["billed_amount"] - frame["write_off_amount"]

    delivery_pct = frame["service_line"].map(DELIVERY_COST_PCT_BY_SERVICE_LINE).astype(
        float
    ) * rng.lognormal(mean=0.0, sigma=0.20, size=size)
    mobilisation = np.array(
        [
            float(rng.lognormal(mean=np.log(MOBILISATION_COST_MEDIAN), sigma=MOBILISATION_COST_SIGMA))
            for _ in range(size)
        ]
    )
    frame["delivery_cost_pct"] = delivery_pct
    frame["direct_delivery_cost"] = np.maximum(
        frame["billed_amount"] * delivery_pct, mobilisation
    ).round(2)

    # Margin is defined as the waterfall residual rather than re-derived from
    # collections, and is deliberately not rounded. Every column it subtracts is
    # already a two-decimal figure, so the published waterfall adds up exactly
    # instead of drifting by a cent per engagement.
    frame["margin"] = (
        frame["standard_value"]
        - frame["discount_amount"]
        - frame["unbilled_value"]
        - frame["write_off_amount"]
        - frame["direct_delivery_cost"]
        - frame["consultant_cost"]
    )

    logger.debug(
        "Derived billing for {} engagements; margin {:.0f}",
        len(frame),
        frame["margin"].sum(),
    )
    return frame


def generate_all(settings: Settings | None = None) -> dict[str, pd.DataFrame]:
    """Generate every table in the synthetic dataset.

    Args:
        settings: Optional settings override; defaults to the process singleton.

    Returns:
        Mapping of table name to DataFrame with keys ``consultants``,
        ``clients``, ``engagements``, ``timesheets`` and ``billing``.
    """
    resolved = settings or get_settings()
    consultants = generate_consultants(resolved)
    clients = generate_clients(resolved)
    engagements = generate_engagements(resolved, clients)
    timesheets = generate_timesheets(resolved, consultants, engagements)
    billing = derive_billing(resolved, engagements, timesheets, consultants)

    return {
        "consultants": consultants,
        "clients": clients,
        "engagements": engagements,
        "timesheets": timesheets,
        "billing": billing,
    }


def write_csvs(frames: dict[str, pd.DataFrame], settings: Settings) -> list[str]:
    """Write the generated tables to ``data/raw`` as CSV.

    Args:
        frames: Mapping produced by :func:`generate_all`.
        settings: Resolved settings supplying the output directory.

    Returns:
        Paths of the files written, as strings, in write order.
    """
    settings.ensure_directories()
    clients = frames["clients"]
    billing = frames["billing"]

    engagements = frames["engagements"].merge(
        clients[["client_id", "client_name", "industry"]],
        on="client_id",
        how="left",
        suffixes=("", "_dup"),
    )
    engagement_columns = [
        "engagement_id",
        "client_id",
        "client_name",
        "industry",
        "service_line",
        "contract_type",
        "start_date",
        "end_date",
        "budgeted_hours",
        "budgeted_fee",
        "size_tier",
        "status",
    ]
    engagement_out = engagements[engagement_columns]

    billing_columns = [
        "engagement_id",
        "standard_value",
        "discount_amount",
        "unbilled_value",
        "billed_amount",
        "write_off_amount",
        "collected_amount",
        "direct_delivery_cost",
        "consultant_cost",
        "margin",
    ]
    billing_out = billing[billing_columns]

    consultant_columns = [
        "consultant_id",
        "consultant_name",
        "level",
        "service_line",
        "standard_rate",
        "cost_rate",
        "hire_date",
        "available_hours",
    ]
    consultant_out = frames["consultants"][consultant_columns]
    client_out = clients[["client_id", "client_name", "industry"]]

    written: list[str] = []
    for name, frame in (
        ("consultants", consultant_out),
        ("clients", client_out),
        ("engagements", engagement_out),
        ("timesheets", frames["timesheets"]),
        ("billing", billing_out),
    ):
        path = settings.raw_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        written.append(str(path))
        logger.debug("Wrote {} rows to {}", len(frame), path)

    return written


def main() -> None:
    """Generate the dataset and write it to ``data/raw``.

    Returns:
        None.
    """
    settings = get_settings()
    frames = generate_all(settings)
    written = write_csvs(frames, settings)
    for path in written:
        logger.info("Wrote {}", path)


if __name__ == "__main__":
    main()
