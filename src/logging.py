"""
Provides a wrapper around Python's logger.
"""

import logging


def get_logger(name: str) -> logging.Logger:
    """
    Custom logger for the project.
    """
    logger = logging.getLogger(name)
    logger.setLevel("INFO")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s")

    ch.setFormatter(formatter)

    logger.addHandler(ch)

    return logger
