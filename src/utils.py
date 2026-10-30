"""Shared helpers: project paths, custom exceptions and logging.

Every other module imports its file locations from here, so nothing depends on
the folder you launch Python from (works the same in VS Code, PowerShell, pytest).
"""
from __future__ import annotations

import logging
from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_CSV = RAW_DIR / "retail_store_inventory.csv"
CLEANED_CSV = PROCESSED_DIR / "retail_inventory_cleaned.csv"
FEATURES_CSV = PROCESSED_DIR / "retail_inventory_features.csv"


# ----------------------------------------------------------- exceptions
class InventoryProjectError(Exception):
    """Base class for all project-specific errors."""


class DataFileError(InventoryProjectError):
    """The CSV is missing, empty, unreadable or malformed."""


class SchemaError(InventoryProjectError):
    """Required columns are missing or column names are ambiguous."""


class DataValidationError(InventoryProjectError):
    """Data is readable but violates a rule (empty result, leakage, bad split...)."""


# -------------------------------------------------------------- helpers
def ensure_dirs() -> None:
    """Create the output folders if they do not exist yet."""
    for folder in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def get_logger(name: str = "smart_inventory") -> logging.Logger:
    """Return a console logger (plain ASCII output so Windows consoles never crash)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
