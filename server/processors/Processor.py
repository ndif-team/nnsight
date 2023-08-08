import logging
import os
import queue
from abc import ABC, abstractmethod
from multiprocessing import Process, Queue
from typing import Any

from .. import CONFIG


class Processor(Process, ABC):
    """
    A Processor is an abstract class representing a handler for some functionality to be carried out on a
    seperate process, and wait on a multiprocessing.Queue for data.

    Attributes
    ----------
        timeout : float
            how many seconds to wait for data on the queue before continuing (to run maintenance).
        queue : multiprocessing.Queue
            queue to wait on for incoming data to process.
    """

    def __init__(self, queue: Queue, timeout: float = 30):
        self.timeout = timeout
        self.queue = queue
        self.logging_handler = None
        self.logger = None

        super().__init__()

    def init_logging(self):
        self.logging_handler = logging.FileHandler(
            os.path.join(CONFIG["LOG_PATH"], f"{self.name}.log"), "a"
        )
        self.logging_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s"
            )
        )
        self.logging_handler.setLevel(logging.DEBUG)
        self.logger = logging.getLogger("NDIF")
        self.logger.addHandler(self.logging_handler)
        self.logger.setLevel(logging.DEBUG)
    
    def initialize(self) -> None:
        """Called on process start."""
        pass

    def maintenance(self) -> None:
        """Called on end of processing loop."""
        pass

    @abstractmethod
    def process(self, data: Any) -> None:
        """
        Abstract method required to be implemented. Perform work on data pulled from queue.

        Parameters
        ----------
            data : Any
        """
        raise NotImplementedError()

    def run(self) -> None:
        """Processing loop of Processor."""

        self.init_logging()

        self.logger.info("Initializing...")

        try:
            self.initialize()

        except:
            self.logger.exception(f"Critical exception encountered in initialization.")
        else:
            self.logger.info("Initialized.")

        while True:
            try:
                data = self.queue.get(timeout=self.timeout)

                self.process(data)

            except queue.Empty:
                pass
            except Exception as error:
                self.logger.exception(f"Exception encountered in processing.")
            finally:
                self.maintenance()
