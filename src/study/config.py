"""project config"""
import os
import sys
from logging.config import dictConfig

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    # pylint: disable=too-few-public-methods
    """default config"""
    # token
    OPENAI_KEY = os.getenv("OPENAI_KEY", "")

    # logging
    LOG_LEVEL = "WARNING"
    LOG_LINE_FORMAT = "%(asctime)s %(levelname)-5s %(threadName)s: %(message)s"
    LOG_DATETIME_FORMAT = "%Y/%m/%d %H:%M:%S"

    TOKEN_BPE = os.path.join(basedir, 'tokenizer/vocab.bpe')
    TOKEN_ID = os.path.join(basedir, 'tokenizer/encoder.json')
    TOKEN_ALPHABET = os.path.join(basedir, 'tokenizer/alphabet_utf8.json')

    @classmethod
    def configure_logger(cls, root_module_name):
        """configure logging"""
        dictConfig({
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "stdout_formatter": {
                    "format": cls.LOG_LINE_FORMAT,
                    "datefmt": cls.LOG_DATETIME_FORMAT,
                },
            },
            "handlers": {
                "stdout_handler": {
                    "level": cls.LOG_LEVEL,
                    "formatter": "stdout_formatter",
                    "class": "logging.StreamHandler",
                    "stream": sys.stdout,
                },
            },
            "loggers": {
                root_module_name: {
                    "handlers": ["stdout_handler"],
                    "level": cls.LOG_LEVEL,
                    "propagate": True,
                },
            },
        })


class TestConfig(Config):
    # pylint: disable=too-few-public-methods
    """testing config"""
    LOG_LEVEL = "DEBUG"
