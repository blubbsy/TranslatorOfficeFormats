"""
Custom logging handler that streams logs to Streamlit session state.
"""

import logging
from collections import deque
from datetime import datetime
from typing import Optional

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx


class StreamlitLogHandler(logging.Handler):
    """
    Custom logging handler that writes log records to st.session_state.
    Maintains a ring buffer to limit memory usage.
    """
    
    MAX_LOGS = 100
    
    def __init__(self, level: int = logging.DEBUG):
        super().__init__(level)
        # We don't check session state here, only on emit
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to session state."""
        # Check if we are in a Streamlit thread context
        if get_script_run_ctx() is None:
            return

        try:
            if "logs" not in st.session_state:
                st.session_state.logs = deque(maxlen=self.MAX_LOGS)
            
            log_entry = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "level": record.levelname,
                "message": self.format(record),
                "logger": record.name,
            }
            st.session_state.logs.append(log_entry)
        except Exception:
            # Don't let logging errors crash the app
            self.handleError(record)


def setup_logging(level: str = "INFO", app_name: str = "OfficeTranslator") -> logging.Logger:
    """
    Set up application logging with both console and Streamlit handlers.
    
    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR)
        app_name: Name for the root logger
        
    Returns:
        Configured logger instance
    """
    # Get or create logger
    logger = logging.getLogger(app_name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Clear existing handlers to avoid duplicates on Streamlit reruns
    logger.handlers.clear()
    
    # Console handler for terminal output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # Streamlit handler for UI display
    try:
        streamlit_handler = StreamlitLogHandler()
        streamlit_format = logging.Formatter("%(levelname)s: %(message)s")
        streamlit_handler.setFormatter(streamlit_format)
        logger.addHandler(streamlit_handler)
    except Exception:
        # Streamlit context may not be available in tests
        pass
    
    return logger


def get_log_entries() -> list:
    """Get current log entries from session state."""
    if "logs" not in st.session_state:
        return []
    return list(st.session_state.logs)


def clear_logs() -> None:
    """Clear all log entries from session state."""
    if "logs" in st.session_state:
        st.session_state.logs.clear()
