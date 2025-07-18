import logging

# set up a std out logger
remote_logger = logging.getLogger("nnsight_remote")
remote_handler = logging.StreamHandler()
remote_handler.setFormatter(
    logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
)
remote_handler.setLevel(logging.INFO)
remote_logger.addHandler(remote_handler)
remote_logger.setLevel(logging.INFO)