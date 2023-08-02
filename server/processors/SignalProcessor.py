from typing import List

import socketio

from . import Processor


class SignalProcessor(Processor):
    def __init__(self, url:str, *args, **kwargs):

        self.url = url

        super().__init__(*args, **kwargs)

    def initialize(self) -> None:
        self.sio = socketio.Client()
        self.sio.connect(self.url)

        super().initialize()

    def maintenance(self) -> None:
        if not self.sio.connected:
            self.sio.connect(self.url)

    def process(self, signal: List):
        self.sio.emit(signal[0], signal[1:])
