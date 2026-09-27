"""Vercel entry point for the Confidence Engine FastAPI application."""

from __future__ import annotations

import sys
from pathlib import Path


# The application was intentionally kept in ``backend`` so local development
# and the existing test suite continue to work unchanged.  Vercel requires
# Python Functions to live under ``api/``, so this adapter exposes the same
# FastAPI instance from a supported entry point.
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
backend_path = str(BACKEND_DIR)
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

# When this file is imported as ``api.index`` (as it is in local verification),
# Python has already created the top-level ``api`` namespace for this directory.
# Include the application's router package in that namespace as well so
# ``backend/main.py`` can keep importing ``api.sessions`` in both environments.
api_package = sys.modules.get("api")
backend_api_path = str(BACKEND_DIR / "api")
if api_package is not None and hasattr(api_package, "__path__"):
    package_paths = api_package.__path__
    if backend_api_path not in package_paths:
        package_paths.append(backend_api_path)

from main import app  # noqa: E402,F401
