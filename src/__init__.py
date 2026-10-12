"""Engagement Profitability Analytics source package.

Modules are imported by their full path so that ``cli.py``, the Streamlit app and
the test suite all share one implementation of every calculation::

    from src.analysis import build_facts
    from src.config import get_settings
"""
