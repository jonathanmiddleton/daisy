import logging.config
from pathlib import Path

import yaml, os, sys

def log_level_from_env():
    import os, logging
    levels = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }
    return levels.get(os.environ.get("DAISY_LOG_LEVEL", "INFO").upper(), logging.INFO)

def _configure_fallback_console():
    import logging
    logging.basicConfig(
        level=log_level_from_env(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def _prepare_logging_paths(config: dict) -> dict:
    handlers = config.get("handlers", {}) or {}
    for name, h in handlers.items():
        if not isinstance(h, dict):
            continue
        # dictConfig canonical format uses "filename" for file handlers
        filename = h.get("filename")
        if filename:
            # Expand env vars and ~, and normalize the path
            expanded = os.path.expanduser(os.path.expandvars(str(filename)))
            expanded = os.path.normpath(expanded)
            # Make sure parent directory exists
            parent = os.path.dirname(expanded)
            if parent:
                os.makedirs(parent, exist_ok=True)
            # Write back expanded path so dictConfig uses it
            h["filename"] = expanded
    return config


class MasterLogger:
    _logger = None
    _is_master = None
    _initialized = False

    @classmethod
    def _initialize(cls):
        if not cls._initialized:
            cls._is_master = True # TODO debugging, remove
            # cls._is_master = int(os.getenv("RANK", 0)) == 0
            if cls._is_master:
                try:
                    project_root = Path(__file__).resolve().parents[1]
                    logging_config_path = project_root / "config" / "logging.yml"

                    with logging_config_path.open('r') as f:
                        config = yaml.safe_load(f.read())
                    config = _prepare_logging_paths(config)
                    logging.config.dictConfig(config)
                except Exception as e:
                    _configure_fallback_console()
                    logging.getLogger(__name__).warning(
                        f"[MasterLogger] Falling back to console logging; failed to apply config/logging.yml: {e}"
                    )
            else:
                # Non-master ranks: keep console-only logging to avoid file contention
                _configure_fallback_console()
            cls._logger = logging.getLogger(__name__)
            cls._initialized = True
            cls._logger.setLevel(log_level_from_env())

    @classmethod
    def isDebugEnabled(cls) -> bool:
        return cls._logger.isEnabledFor(logging.DEBUG)

    @classmethod
    def debug(cls, st):
        if cls._is_master:
            cls._logger.debug(st, stacklevel=2)

    @classmethod
    def info(cls, st):
        if cls._is_master:
            cls._logger.info(st, stacklevel=2)

    @classmethod
    def warning(cls, st):
        if cls._is_master:
            cls._logger.warning(st, stacklevel=2)

    @classmethod
    def error(cls, st):
        if cls._is_master:
            cls._logger.error(st, stacklevel=2)

MasterLogger._initialize()