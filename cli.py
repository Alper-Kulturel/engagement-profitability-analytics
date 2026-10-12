#!/usr/bin/env python3
"""Command line entry point for the Engagement Profitability Analytics pipeline.

The pipeline runs in four stages. Each stage writes its output to disk and can be
re-run on its own without disturbing the others, so a change to the analysis does
not require regenerating the data and a regeneration does not invalidate the
report until it is rebuilt.

Usage::

    python cli.py generate-data          # write the synthetic CSVs to data/raw
    python cli.py analyze                # metrics, tables and figures
    python cli.py report                 # self-contained HTML report
    python cli.py app                    # launch the Streamlit application
    python cli.py all                    # generate, analyze and report
    python cli.py info                   # show resolved paths and constants

Every command accepts ``-v`` to turn on debug logging.

Note:
    The dataset is synthetic. It is produced from a fixed seed and describes no
    real firm, client or person.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from loguru import logger

from src.config import (
    ANALYSIS_END,
    ANALYSIS_START,
    HEADLINE_YEAR,
    N_CLIENTS,
    N_CONSULTANTS,
    N_ENGAGEMENTS,
    RANDOM_SEED,
    get_settings,
)


def _configure_logging(verbose: bool) -> None:
    """Send pipeline logging to stderr in a compact, coloured format.

    Args:
        verbose: When True, emit DEBUG records; otherwise emit INFO and above.

    Returns:
        None.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if verbose else "INFO",
        colorize=True,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
        ),
    )


def _cmd_generate_data(args: argparse.Namespace) -> int:
    """Generate the synthetic dataset and write it to ``data/raw``.

    Args:
        args: Parsed command line arguments.

    Returns:
        Process exit code.
    """
    from src.generate_data import generate_all, write_csvs

    settings = get_settings()
    if args.force:
        removed = 0
        for path in sorted(settings.raw_dir.glob("*.csv")):
            path.unlink()
            removed += 1
        logger.info("Removed {} existing CSV files from {}", removed, settings.raw_dir)

    tables = generate_all(settings)
    write_csvs(tables, settings)
    logger.info(
        "Generated {} tables with seed {} covering {}-{}",
        len(tables),
        settings.random_seed,
        ANALYSIS_START,
        ANALYSIS_END,
    )
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    """Compute every metric and write the tables and figures.

    Args:
        args: Parsed command line arguments.

    Returns:
        Process exit code.
    """
    from src.report import run_analysis

    settings = get_settings()
    if not (settings.raw_dir / "billing.csv").exists():
        logger.error(
            "No generated data found in {}. Run 'python cli.py generate-data' first.",
            settings.raw_dir,
        )
        return 1

    bundle = run_analysis(settings)
    summary = bundle.headline_summary
    logger.info(
        "{}: billed ${:,.0f} of ${:,.0f} rate-card value at {:.1%} realisation; "
        "margin ${:,.0f} at {:.1%}",
        settings.headline_year,
        summary["billed_amount"],
        summary["standard_value"],
        summary["realization_rate"],
        summary["margin"],
        summary["margin_rate"],
    )
    logger.info("Metrics: {}", settings.metrics_path)
    logger.info("Tables:  {}", settings.tables_dir)
    logger.info("Figures: {}", settings.figures_dir)
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    """Build the self-contained HTML report.

    Args:
        args: Parsed command line arguments.

    Returns:
        Process exit code.
    """
    from src.report import run_report

    settings = get_settings()
    if not (settings.raw_dir / "billing.csv").exists():
        logger.error(
            "No generated data found in {}. Run 'python cli.py generate-data' first.",
            settings.raw_dir,
        )
        return 1

    path = run_report(settings)
    logger.info("Report written to {}", path)
    logger.info("Open it directly in a browser; it needs no server and no network.")
    return 0


def _cmd_app(args: argparse.Namespace) -> int:
    """Launch the Streamlit application.

    Args:
        args: Parsed command line arguments.

    Returns:
        Process exit code.
    """
    settings = get_settings()
    app_path = settings.app_dir / "app.py"
    if not app_path.exists():
        logger.error("Application not found at {}", app_path)
        return 1

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(args.port),
        "--server.headless",
        "true" if args.headless else "false",
    ]
    logger.info("Starting Streamlit on port {}", args.port)
    logger.info("Command: {}", " ".join(command))
    return subprocess.call(command)


def _cmd_all(args: argparse.Namespace) -> int:
    """Run generate, analyze and report in order.

    Args:
        args: Parsed command line arguments.

    Returns:
        Process exit code. The first non-zero stage stops the run.
    """
    from src.generate_data import generate_all, write_csvs
    from src.report import run_analysis, run_report

    settings = get_settings()

    logger.info("[1/3] Generating the synthetic dataset")
    tables = generate_all(settings)
    write_csvs(tables, settings)

    logger.info("[2/3] Computing metrics, tables and figures")
    bundle = run_analysis(settings)

    logger.info("[3/3] Building the HTML report")
    path = run_report(settings)

    summary = bundle.headline_summary
    logger.info(
        "Done. {} billed ${:,.0f} at {:.1%} margin. Report: {}",
        settings.headline_year,
        summary["billed_amount"],
        summary["margin_rate"],
        path,
    )
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    """Print the resolved configuration and the state of the pipeline.

    Args:
        args: Parsed command line arguments.

    Returns:
        Process exit code.
    """
    settings = get_settings()
    print("Engagement Profitability Analytics")
    print("=" * 40)
    print(f"Project root      {settings.project_root}")
    print(f"Python            {sys.version.split()[0]}")
    print()
    print("Model")
    print(f"  Random seed     {RANDOM_SEED}")
    print(f"  Window          {ANALYSIS_START} to {ANALYSIS_END}")
    print(f"  Headline year   {HEADLINE_YEAR}")
    print(f"  Firm size       {N_CONSULTANTS} consultants, {N_CLIENTS} clients, {N_ENGAGEMENTS} engagements")
    print()
    print("Data")
    for name in ("consultants", "clients", "engagements", "timesheets", "billing"):
        path = settings.raw_dir / f"{name}.csv"
        if path.exists():
            rows = sum(1 for _ in path.open(encoding="utf-8")) - 1
            print(f"  {name:<14} {rows:>7,} rows   {path.name}")
        else:
            print(f"  {name:<14} {'missing':>7}      run 'python cli.py generate-data'")
    print()
    print("Outputs")
    for label, directory in (
        ("Figures", settings.figures_dir),
        ("Tables", settings.tables_dir),
        ("Reports", settings.reports_dir),
    ):
        count = len(list(directory.glob("*"))) if directory.exists() else 0
        print(f"  {label:<14} {count:>7,} files  {directory}")
    metrics = settings.metrics_path
    print(f"  {'Metrics':<14} {'present' if metrics.exists() else 'missing':>7}      {metrics.name}")
    print()
    print("Note: the dataset is synthetic and contains no real information.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description=(
            "Engagement profitability pipeline: generate a synthetic consulting "
            "portfolio, analyse where margin leaks, and report on what closing "
            "the gap is worth."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python cli.py all -v\n"
            "  python cli.py generate-data --force\n"
            "  python cli.py report\n"
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable debug logging",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    generate = subparsers.add_parser(
        "generate-data",
        help="write the synthetic CSVs to data/raw",
        description="Generate the synthetic dataset from the fixed seed.",
    )
    generate.add_argument(
        "--force",
        action="store_true",
        help="delete existing CSVs in data/raw before generating",
    )
    generate.set_defaults(func=_cmd_generate_data)

    analyze = subparsers.add_parser(
        "analyze",
        help="compute metrics, tables and figures",
        description="Compute every metric and write reports/metrics.json, "
        "reports/tables/*.csv and reports/figures/*.png.",
    )
    analyze.set_defaults(func=_cmd_analyze)

    report = subparsers.add_parser(
        "report",
        help="build the self-contained HTML report",
        description="Render the HTML report with all images inlined. The result "
        "opens from the filesystem with no server and no network.",
    )
    report.set_defaults(func=_cmd_report)

    app = subparsers.add_parser(
        "app",
        help="launch the Streamlit application",
        description="Start the interactive Streamlit front end.",
    )
    app.add_argument("--port", type=int, default=8501, help="port to serve on (default: 8501)")
    app.add_argument(
        "--headless",
        action="store_true",
        help="do not open a browser window",
    )
    app.set_defaults(func=_cmd_app)

    all_stages = subparsers.add_parser(
        "all",
        help="run generate-data, analyze and report in order",
        description="Run the full pipeline end to end.",
    )
    all_stages.set_defaults(func=_cmd_all)

    info = subparsers.add_parser(
        "info",
        help="show resolved paths, constants and pipeline state",
        description="Print the configuration and the state of the generated data.",
    )
    info.set_defaults(func=_cmd_info)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the requested stage.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(getattr(args, "verbose", False))

    if not getattr(args, "command", None):
        parser.print_help()
        return 1

    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        return 130
    except FileNotFoundError as error:
        logger.error("{}", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
