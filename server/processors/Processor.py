import queue
from abc import abstractmethod
from multiprocessing import Process, Queue
from typing import Any


class Processor(Process):
    def __init__(self, queue: Queue, timeout: float = 30):
        self.timeout = timeout
        self.queue = queue

        super().__init__()

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def maintenance(self) -> None:
        pass

    @abstractmethod
    def process(self, data: Any) -> None:
        raise NotImplementedError()

    def run(self) -> None:
        self.initialize()

        while True:
            try:
                data = self.queue.get(timeout=self.timeout)

                self.process(data)

            except queue.Empty:
                pass
            except Exception as error:
                raise error
            finally:
                self.maintenance()
