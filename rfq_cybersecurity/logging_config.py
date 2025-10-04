"""
Logging configuration for RFQ Cybersecurity.

This module defines a package‑wide logger that can be imported by
other modules.  It sets up a simple stream handler with a uniform
format.  The logger is created lazily and configured only once.
"""

import logging

def get_logger(name: str = "rfq_cybersecurity") -> logging.Logger:
    """Return a configured logger instance.

    The logger will output messages to the console with a
    timestamp, module name and log level.  Repeated calls
    return the same logger, so handlers are not duplicated.

    Parameters
    ----------
    name: str
        The name of the logger to create or retrieve (default
        ``rfq_cybersecurity``).

    Returns
    -------
    logging.Logger
        A configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Create a console handler
        handler = logging.StreamHandler()
        # Define a log message format
        # Use classic logging format: timestamp, logger name, level and message
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # Default logging level can be configured via an environment variable
        level_name = logging.getLevelName(logging.INFO)
        logger.setLevel(level_name)
    return logger