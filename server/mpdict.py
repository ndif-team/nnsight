import json
import os
from threading import Semaphore

def aquire(function):
    def wrapper(self, *args,**kwargs):

        self.semaphore.acquire()
        result = function(self, *args,**kwargs)
        self.semaphore.release()

        return result
    
    return wrapper

class MPDict(dict):


    def __init__(self, results_path:str, semaphore:Semaphore):

        self.results_path = results_path
        self.semaphore = semaphore

        os.makedirs(self.results_path, exist_ok=True)

        super().__init__()

    @aquire
    def __setitem__(self, key, item):

        path = os.path.join(self.results_path, key)

        if not os.path.exists(path):

            os.makedirs(path)

        path = os.path.join(path, 'results.json')

        with open(path, 'w') as file:

            json.dump(item, file)

    @aquire
    def __getitem__(self, key):

        path = os.path.join(self.results_path, key, 'results.json')

        if not os.path.exists(path):

            raise KeyError(path)
        
        with open(path, 'r') as file:
            result = json.load(file)
        
        return result

    #TODO
    def __repr__(self):
        return repr(self.__dict__)

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

    #TODO
    def clear(self):
        return self.__dict__.clear()

    #TODO
    def copy(self):
        return self.__dict__.copy()

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

    #TODO
    def __contains__(self, key):
        
        path = os.path.join(self.results_path, key)

        return os.path.exists(path)

    #TODO
    def __iter__(self):
        return iter(self.__dict__)

    #TODO
    def __unicode__(self):
        return unicode(repr(self.__dict__))