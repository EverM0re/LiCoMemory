import logging
import os
import sys
from pathlib import Path

def setup_logger(name: str = "DynamicMemory", level: str = "INFO", log_dir: str = "./results") -> logging.Logger:
    """Setup logger with proper formatting."""

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    log_file = Path(os.path.join(log_dir, "dynamic_memory.log"))
    log_file.parent.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def update_logger_path(log_dir: str):
    """Update the logger file path dynamically."""
    global logger
    logger = setup_logger(log_dir=log_dir)

# Global logger instance
logger = setup_logger()
