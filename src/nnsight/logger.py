import logging
import os
from logging.handlers import RotatingFileHandler

PATH = os.path.dirname(os.path.abspath(__file__))

logging_handler = RotatingFileHandler(
    os.path.join(PATH, f"nnsight.log"),
    mode="a",
    maxBytes=5 * 1024 * 1024,
    backupCount=2,
    encoding=None,
    delay=0,
)
logging_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s"
    )
)
logging_handler.setLevel(logging.INFO)

logger = logging.getLogger("nnsight")
logger.addHandler(logging_handler)
logger.setLevel(logging.INFO)
