
import torch.futures
import torch.fx
from .Promise import Proxy
from typing import List

class Intervention:


    def __init__(self, proxy:Proxy) -> None:
    
        self.proxy = proxy