import torch

class MAX_BATCH_SIZE_ESTIMATER:
    
    def __init__(self, net, opt, loss_func, gpu_id=0, input_features=1, input_size_x=256, input_size_y=256):
        self.__gpu_id = gpu_id
        self.__net = net
        #self.__opt = opt
        self.__loss_func = loss_func
        self.__input_features = input_features
        self.__input_size_x = input_size_x
        self.__input_size_y = input_size_y        
        
    def check_bs(self,bs):        
        input_data = None
        out = None
        label_data = None
        loss = None
        try:
            input_data = torch.cuda.FloatTensor(bs,self.__input_features,self.__input_size_y,self.__input_size_x, 
                                            device="cuda:"+str(self.__gpu_id))
            out = self.__net(input_data)                
            label_data = torch.cuda.FloatTensor(out.size(), device=out.device)         
            loss = self.__loss_func(out,label_data)
            loss.backward()
            result = True 
        except RuntimeError as e:
            result = str(e).find("CUDA out of memory") == -1
        
        if input_data is not None:
            del input_data
        if out is not None:
            del out
        if label_data is not None:
            del label_data
        if loss is not None:
            del loss            
        
        torch.cuda.empty_cache()
        
        return result
            
    def find_max_bs(self):        
        find_top = True
        bss = [2]
        final_bs = 1
        max_iter = 50
        c_iter = 0
        while True:
            c_iter += 1
            if c_iter == max_iter:
                break
                
            result = self.check_bs(bss[-1])
            if result:
                if find_top:
                    bss.append(bss[-1]*2)
                else:
                    bss.append(int(bss[-1]+((abs(bss[-1] - bss[-2]))/2)))                   
               
                if abs(bss[-1]-bss[-2]) < 2:
                    final_bs = bss[-1]
                    break
            else:
                if find_top:
                    find_top=False
                bss.append(int(bss[-1]-((abs(bss[-1] - bss[-2]))/2)))                    
                
                if abs(bss[-1]-bss[-2]) == 0:
                    final_bs = bss[-1]-1
                    break
      
        
        if final_bs > 5:
            final_bs = round(final_bs-(final_bs*0.05))
            
        return final_bs