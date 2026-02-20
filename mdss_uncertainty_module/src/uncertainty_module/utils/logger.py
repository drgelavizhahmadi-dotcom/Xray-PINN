"""Logging configuration."""

import logging
import sys
from pathlib import Path
from typing import Optional


def configure_logging(log_level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """Configure logging."""
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)
