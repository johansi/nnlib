from pynvml import *

class GPU_MEM_INFO():
    def __init__(self, gpu_id):
        nvmlInit()
        self.__h = nvmlDeviceGetHandleByIndex(gpu_id)
    
    def __info(self):
        return nvmlDeviceGetMemoryInfo(self.__h)
    
    def total(self):
        return self.__byte_to_mib(self.__info().total)
    
    def free(self):
        return self.__byte_to_mib(self.__info().free)
    
    def used(self):
        return self.__byte_to_mib(self.__info().used)
    
    def __byte_to_mib(self,inp):
        return round(inp/(1024**2))
        
        