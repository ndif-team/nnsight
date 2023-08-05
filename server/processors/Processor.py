import logging
import queue
from abc import ABC, abstractmethod
from multiprocessing import Process, Queue
from typing import Any


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

        super().__init__()

    @abstractmethod
    def initialize(self) -> None:
        """Abstract method to be optionally implemented. Called on process start."""
        pass

    @abstractmethod
    def maintenance(self) -> None:
        """Abstract method to be optionally implemented. Called on end of processing loop."""
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
        try:
            self.initialize()

        except:
            logging.exception(
                f"Critical exception encountered in initialization for processor ({self})"
            )

        while True:
            try:
                data = self.queue.get(timeout=self.timeout)

                self.process(data)

            except queue.Empty:
                pass
            except Exception as error:
                logging.exception(
                    f"Exception encountered in processing for processor ({self})"
                )
            finally:
                self.maintenance()
