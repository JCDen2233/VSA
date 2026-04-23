from pathlib import Path
from typing import List

from dotenv import load_dotenv

CONFIG_PATH = Path(__file__).parent.parent / ".env"


def load_config() -> dict:
    load_dotenv(CONFIG_PATH)
    return {
        "DB_HOST": get_env("DB_HOST", "localhost"),
        "DB_PORT": int(get_env("DB_PORT", "3306")),
        "DB_NAME": get_env("DB_NAME", "moex"),
        "DB_USER": get_env("DB_USER", "root"),
        "DB_PASSWORD": get_env("DB_PASSWORD", ""),
        "RISK_PER_TRADE": float(get_env("RISK_PER_TRADE", "0.01")),
        "RR_RATIO": float(get_env("RR_RATIO", "2.0")),
        "COMMISSION_PCT": float(get_env("COMMISSION_PCT", "0.001")),
        "SLIPPAGE_PCT": float(get_env("SLIPPAGE_PCT", "0.0005")),
        "MAX_DD_PCT": float(get_env("MAX_DD_PCT", "0.20")),
        "ALLOW_SHORT": get_env("ALLOW_SHORT", "false").lower() == "true",
        "TICKER_LIST": get_env("TICKER_LIST", "SBER").split(","),
        "DEFAULT_TF": get_env("DEFAULT_TF", "D1,H1").split(","),
    }


def get_env(key: str, default: str = "") -> str:
    import os
    return os.getenv(key, default)


def validate_config(config: dict) -> bool:
    required = ["DB_HOST", "DB_NAME", "DB_USER", "RISK_PER_TRADE", "RR_RATIO"]
    for key in required:
        if key not in config or config[key] is None:
            raise ValueError(f"Missing required config: {key}")
    if not 0 < config["RISK_PER_TRADE"] <= 1:
        raise ValueError("RISK_PER_TRADE must be between 0 and 1")
    if config["RR_RATIO"] <= 0:
        raise ValueError("RR_RATIO must be positive")
    return True


class Config:
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self._config = load_config()
            validate_config(self._config)

    def __getitem__(self, key: str):
        return self._config[key]

    def get(self, key: str, default=None):
        return self._config.get(key, default)

    @property
    def db_url(self) -> str:
        return f"mysql+pymysql://{self._config['DB_USER']}:{self._config['DB_PASSWORD']}@{self._config['DB_HOST']}:{self._config['DB_PORT']}/{self._config['DB_NAME']}"


config = Config()