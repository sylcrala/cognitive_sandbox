"""
Central logging utility for all modules in the Cognitive Sandbox.
Provides a consistent logging interface across the application.
"""
import uuid
import datetime as dt
import numpy as np
import os
import json
from main.config import get_config
import logging
from typing import Union, Dict, Any, Optional


class Logger:
    """
    Logging wrapper that provides convenient methods for different log levels.
    Automatically handles JSON serialization for dict logging.
    """
    
    def __init__(self):
        """Initialize logger with empty state."""
        self._logger: Optional[logging.Logger] = None
        self._name: Optional[str] = None
    
    def get_logger(self, name: str) -> "Logger":
        """
        Get a logger instance with a specific name.

        Args:
            name (str): The name of the logger (typically module or class name).

        Returns:
            Logger: This logger instance configured with the given name.
            
        Raises:
            RuntimeError: If logger configuration fails.
        """
        try:
            config = get_config()
            self._name = name
            self._logger = logging.getLogger(name)
            
            if not self._logger.hasHandlers():
                self._logger.setLevel(logging.DEBUG)
                
                log_file = os.path.join(str(config.LOGS_DIR), f"{config.session_id}.log")
                ch = logging.FileHandler(log_file)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                ch.setFormatter(formatter)
                self._logger.addHandler(ch)
        except Exception as e:
            # Fallback if file logging fails
            print(f"Warning: Could not set up file logging for {name}: {e}")
            self._logger = logging.getLogger(name)
        
        return self
    
    def log(self, msg: Union[str, Dict[str, Any]]) -> None:
        """
        Log a message at INFO level. Accepts strings or dicts.

        Args:
            msg (Union[str, Dict]): The message to log (string or dict for JSON logging).
            
        Raises:
            RuntimeError: If logger is not initialized.
        """
        if self._logger is None:
            raise RuntimeError("Logger not initialized. Call get_logger() first.")
        
        try:
            if isinstance(msg, dict):
                self._logger.info(json.dumps(msg))
            else:
                self._logger.info(str(msg))
        except Exception as e:
            self._logger.error(f"Failed to log message: {e}")
    
    def debug(self, msg: Union[str, Dict[str, Any]]) -> None:
        """
        Log a debug message.

        Args:
            msg (Union[str, Dict]): The message to log.
            
        Raises:
            RuntimeError: If logger is not initialized.
        """
        if self._logger is None:
            raise RuntimeError("Logger not initialized. Call get_logger() first.")
        
        try:
            if isinstance(msg, dict):
                self._logger.debug(json.dumps(msg))
            else:
                self._logger.debug(str(msg))
        except Exception as e:
            self._logger.error(f"Failed to log debug message: {e}")
    
    def warning(self, msg: Union[str, Dict[str, Any]]) -> None:
        """
        Log a warning message.

        Args:
            msg (Union[str, Dict]): The message to log.
            
        Raises:
            RuntimeError: If logger is not initialized.
        """
        if self._logger is None:
            raise RuntimeError("Logger not initialized. Call get_logger() first.")
        
        try:
            if isinstance(msg, dict):
                self._logger.warning(json.dumps(msg))
            else:
                self._logger.warning(str(msg))
        except Exception as e:
            self._logger.error(f"Failed to log warning: {e}")
    
    def error(self, msg: Union[str, Dict[str, Any]]) -> None:
        """
        Log an error message.

        Args:
            msg (Union[str, Dict]): The message to log.
            
        Raises:
            RuntimeError: If logger is not initialized.
        """
        if self._logger is None:
            raise RuntimeError("Logger not initialized. Call get_logger() first.")
        
        try:
            if isinstance(msg, dict):
                self._logger.error(json.dumps(msg))
            else:
                self._logger.error(str(msg))
        except Exception as e:
            print(f"Critical: Failed to log error: {e}")

