from flask import request
def internal(function):
    def wrapper(self, *args, **kwargs):
        
        result = function(self, *args, **kwargs)
  
    return wrapper
