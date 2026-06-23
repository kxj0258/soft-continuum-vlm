import logging

from soft_continuum_vlm.utils.logging import get_logger


def test_get_logger_returns_named_logger() -> None:
    logger = get_logger("soft_continuum_vlm.tests")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "soft_continuum_vlm.tests"
    assert logger.level == logging.INFO
