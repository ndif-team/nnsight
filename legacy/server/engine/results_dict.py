import json
import os
from threading import Semaphore

from engine.models.result import Result


def aquire(function):
    def wrapper(self, *args,**kwargs):

        self.semaphore.acquire()
        try:
            result = function(self, *args,**kwargs)
        except Exception as error:

            self.semaphore.release()

            raise error

        self.semaphore.release()

        return result
    
    return wrapper

class ResultsDict(dict):

    def __init__(self, 
                results_path:str, 
                semaphore:Semaphore):

        self.results_path = results_path
        self.semaphore = semaphore

        os.makedirs(self.results_path, exist_ok=True)

        super().__init__()

    @aquire
    def __setitem__(self, key:str, item:Result):

        item.log()

        path = os.path.join(self.results_path, key)

        if not os.path.exists(path):

            os.makedirs(path)

        path = os.path.join(path, 'results.json')

        with open(path, 'w') as file:

            file.write(item.json())

    @aquire
    def __getitem__(self, key:str) -> Result:

        path = os.path.join(self.results_path, key, 'results.json')

        if not os.path.exists(path):

            raise KeyError(path)
        
        with open(path, 'r') as file:
            result = Result(**json.load(file))
        
        return result

    def __len__(self):
        return len(os.listdir(self.results_path))

    @aquire
    def __delitem__(self, key):

        path = os.path.join(self.results_path, key)

        if not os.path.exists(path):

            raise KeyError(path)
        
        file_path = os.path.join(path, 'results.json')
        
        if os.path.exists(file_path):

            os.remove(file_path)

        os.rmdir(path)

    def clear(self):
        for key in self.keys(): self.__delitem__(key)

    #TODO
    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

    def keys(self):
        return os.listdir(self.results_path)

    #TODO
    def values(self):
        return self.__dict__.values()

    #TODO
    def items(self):
        return self.__dict__.items()

    #TODO
    def pop(self, *args):
        return self.__dict__.pop(*args)

    #TODO
    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)

    def __contains__(self, key):
        
        path = os.path.join(self.results_path, key)

        return os.path.exists(path)

    #TODO
    def __iter__(self):
        return iter(self.__dict__)
