from sys import gettrace, settrace
from types import FrameType

# local trace function which returns itself 
def my_tracer(frame:FrameType, event, arg = None): 
    # extracts frame code 
    code = frame.f_code 
  
    # extracts calling function name 
    func_name = code.co_consts 
  
    # extracts the line number 
    line_no = frame.f_lineno 
  
    print(event) 
  
    return my_tracer 
  
  
# global trace function is invoked here and 
# local trace function is set for fun() 
def fun():
    
    zzz = list()
    
    zzz.append(1)
    
    if len(zzz) > 2:
        return 1 
    return "GFG"
  
  
# global trace function is invoked here and 
# local trace function is set for check() 
def check(): 
    return fun() 
  
  
# returns reference to local 
# trace function (my_tracer) 
settrace(my_tracer) 
  
check() 