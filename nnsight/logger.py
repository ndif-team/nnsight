import logging
import os

PATH = os.path.dirname(os.path.abspath(__file__))
logging_handler = logging.FileHandler(os.path.join(PATH, f"nnsight.log"), "a")
logging_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s"
    )
)
logging_handler.setLevel(logging.DEBUG)
logger = logging.getLogger("nnsight")
logger.addHandler(logging_handler)
logger.setLevel(logging.DEBUG)
