"""
Configuration helpers for RFQ Cybersecurity.

This module centralises the loading of environment variables and
provides a simple wrapper for accessing Azure OpenAI configuration.
Having a single location for configuration makes it easier to manage
deployment differences (development, staging, production) and avoids
polluting business logic with environment lookups.  Environment
variables are loaded from a ``.env`` file via :func:`dotenv.load_dotenv`.
"""

from __future__ import annotations

import os
from typing import Dict

from dotenv import load_dotenv

# Import the package logger
from .logging_config import get_logger

# Create a logger for this module
logger = get_logger(__name__)


def load_config() -> Dict[str, str | None]:
    """Load and return Azure OpenAI configuration values.

    The environment variables ``AZURE_OPENAI_ENDPOINT``,
    ``AZURE_OPENAI_API_KEY``, ``AZURE_OPENAI_API_VERSION``,
    ``EMBEDDING_MODEL`` and ``LLM_MODEL`` are read.  Values are
    returned as a dictionary; missing keys yield ``None`` values.

    Returns
    -------
    dict
        A mapping of configuration names to their loaded values.
    """
    # Ensure values from a `.env` file are available.  If the file
    # cannot be read, the dotenv loader will simply return without
    # raising an exception.  We still wrap this in a try/except to
    # log unexpected errors during environment loading.
    try:
        load_dotenv()
    except Exception as exc:
        # Log the exception but continue; environment variables may
        # still be available in the process environment.
        logger.error("Failed to load .env file: %s", exc)
    # Collect configuration values from environment variables.  If
    # variables are missing they will be ``None``, which allows
    # callers to handle defaults appropriately.
    config = {
        "openai_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
        "openai_key": os.environ.get("AZURE_OPENAI_API_KEY"),
        "openai_version": os.environ.get("AZURE_OPENAI_API_VERSION"),
        "embedding_model": os.environ.get("EMBEDDING_MODEL"),
        "llm_model": os.environ.get("LLM_MODEL"),
    }
    logger.debug("Loaded configuration: %s", {k: bool(v) for k, v in config.items()})
    return config