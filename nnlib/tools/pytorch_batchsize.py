import torch
import pdb
from .gpu_memory_tools import *
import numpy as np

class Batch_Size_Estimator:
    
    def __init__(self, net, opt, loss_func, dataset, gpu_id=0):
        self.__gpu_info = GPU_MEM_INFO(gpu_id)
        self.__device = torch.device("cuda:"+str(gpu_id))
        self.__net = net
        self.__loss_func = loss_func
        self.__opt = opt
        self.__dataset = dataset

    def __loss_batch(self,batches):
        xdata = self.__xdata_orig.repeat(batches,1,1,1).to(self.__device)
        ydata = self.__ydata_orig.repeat(batches,1,1,1).to(self.__device)
        out = self.__net(xdata)
        loss = self.__loss_func(out, ydata)
        loss.backward()
        self.__opt.step()
        self.__opt.zero_grad()
        loss_v = float(loss.detach().cpu())
        max_mem = round(torch.cuda.max_memory_allocated(0)/1000/1000/1.049)
        del xdata
        del ydata
        del out
        del loss
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        return max_mem        
    
            
    def find_max_bs(self):
        
        mems_used = []
        self.__xdata_orig,self.__ydata_orig = self.__dataset[0]
        for i in range(3):           
            mems_used.append(self.__loss_batch(i+1))
        mems_used = np.array(mems_used[1:])    
        mebi_per_set = mems_used[1]-mems_used[0]
        mebi_per_set += mebi_per_set*0.07
                    
        return int(round((self.__gpu_info.total() -  self.__gpu_info.used())/mebi_per_set))