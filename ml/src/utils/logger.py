"""
Logging utilities for ML pipeline.
Provides structured logging with proper formatting and file output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class MLPipelineLogger:
    """Custom logger for ML pipeline with structured output."""
    
    def __init__(
        self,
        name: str,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        log_format: Optional[str] = None
    ):
        """
        Initialize ML pipeline logger.
        
        Args:
            name: Logger name
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file (optional)
            log_format: Custom log format (optional)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Default format
        if log_format is None:
            log_format = (
                "%(asctime)s - %(name)s - %(levelname)s - "
                "%(funcName)s:%(lineno)d - %(message)s"
            )
        
        formatter = logging.Formatter(log_format)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(self._format_message(message, **kwargs))
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.logger.info(self._format_message(message, **kwargs))
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(self._format_message(message, **kwargs))
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.logger.error(self._format_message(message, **kwargs))
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self.logger.critical(self._format_message(message, **kwargs))
    
    def _format_message(self, message: str, **kwargs) -> str:
        """Format message with additional context."""
        if kwargs:
            context = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
            return f"{message} | {context}"
        return message
    
    def log_experiment_start(self, experiment_name: str, **kwargs) -> None:
        """Log experiment start with metadata."""
        self.info(
            f"Starting experiment: {experiment_name}",
            timestamp=datetime.now().isoformat(),
            **kwargs
        )
    
    def log_experiment_end(self, experiment_name: str, **kwargs) -> None:
        """Log experiment end with metadata."""
        self.info(
            f"Completed experiment: {experiment_name}",
            timestamp=datetime.now().isoformat(),
            **kwargs
        )
    
    def log_model_training(self, model_name: str, **kwargs) -> None:
        """Log model training with metadata."""
        self.info(
            f"Training model: {model_name}",
            **kwargs
        )
    
    def log_model_evaluation(self, model_name: str, metrics: dict, **kwargs) -> None:
        """Log model evaluation results."""
        self.info(
            f"Model evaluation completed: {model_name}",
            metrics=metrics,
            **kwargs
        )
    
    def log_data_loading(self, data_source: str, rows: int, **kwargs) -> None:
        """Log data loading with statistics."""
        self.info(
            f"Data loaded from {data_source}",
            rows=rows,
            **kwargs
        )
    
    def log_error(self, error: Exception, context: str = "", **kwargs) -> None:
        """Log error with context."""
        self.error(
            f"Error in {context}: {str(error)}",
            error_type=type(error).__name__,
            **kwargs
        )


def get_logger(
    name: str,
    log_level: str = "INFO",
    log_file: Optional[str] = None
) -> MLPipelineLogger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name
        log_level: Logging level
        log_file: Path to log file
        
    Returns:
        Configured logger instance
    """
    return MLPipelineLogger(name, log_level, log_file)


# Default logger for the module
default_logger = get_logger(__name__)

