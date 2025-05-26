import logging
import os
from logging.handlers import RotatingFileHandler

# Work out the log directory:
# – Look for an environment variable called NNSIGHT_LOG_PATH (stripped of whitespace);
# – if it’s unset or empty, fall back to the default.
# Then make sure that directory actually exists, creating it if necessary.
_DEFAULT_PATH = os.path.dirname(os.path.abspath(__file__))
_env_path = os.getenv("NNSIGHT_LOG_PATH", "").strip()
PATH = _env_path or _DEFAULT_PATH
os.makedirs(PATH, exist_ok=True)

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

# set up a std out logger
remote_logger = logging.getLogger("nnsight_remote")
remote_handler = logging.StreamHandler()
remote_handler.setFormatter(
    logging.Formatter("%(asctime)s %(message)s")
)
remote_handler.setLevel(logging.INFO)
remote_logger.addHandler(remote_handler)
remote_logger.setLevel(logging.INFO)