import torch
from fastprogress.fastprogress import master_bar, progress_bar
import numpy as np
from time import time
from time import strftime, gmtime

__all__ = ["LearnerCallback", "Learner"]

class LearnerCallback(): 
    
    def get_metric_names(self):
        return
    
    def on_train_begin(self):
        return

    def on_train_end(self):
        return
    
    def on_epoch_begin(self):
        return

    def on_epoch_end(self):
        return
    
    def on_batch_begin(self, features, target, train):
        return
    
    def on_batch_end(self, output, target, train):
        return
        

class Learner:
    
    def __init__(self, model, train_dl, valid_dl, loss_func, optimizer, learner_callback=None, gpu_id=0, predict_smaple_func=None):        
        self.device = torch.device("cuda:"+str(gpu_id))
        self.predict_smaple_func = predict_smaple_func if predict_smaple_func is not None else lambda : None        
        self.model = model
        self.train_dl = train_dl
        self.valid_dl = valid_dl        
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.learner_callback = learner_callback if learner_callback is not None else LearnerCallback()
        self.metric_names = self.learner_callback.get_metric_names()
        self.metrics = None
        self.train_losses = None
        self.valid_losses = None
                            
    def loss_batch(self, x_data, y_data, is_train=True):
        "Calculate loss and metrics for a batch."    
        
        self.learner_callback.on_batch_begin(x_data, y_data, train=is_train)
        out = self.model(x_data)                
        loss = self.loss_func(out, y_data)
        if is_train:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        self.learner_callback.on_batch_end(out.detach().cpu(), y_data.detach().cpu(), train=is_train)
        return float(loss.detach().cpu())
    
    def validate(self,dl,parrent_bar):
        "Calculate `loss_func` of `model` on `dl` in evaluation mode."
        self.model.eval()
        with torch.no_grad():
            val_losses = []
            
            for x_data,y_data in progress_bar(self.valid_dl, parent=parrent_bar):                
                x_data = x_data.to(self.device)
                y_data = y_data.to(self.device)                                              
                val_loss = self.loss_batch(x_data, y_data, is_train=False)
                parrent_bar.child.comment = str(round(val_loss,4))
                val_losses.append(val_loss)

        return np.array(val_losses).mean()                        
    
    def get_losses(self, train=True, valid=True):
        assert (train or valid), "train or valid must be True"
        losses = []        
        if train:
            losses.append(np.array(self.train_losses))
        
        if valid:
            losses.append(np.array(self.valid_losses))
            
        return losses[0] if len(losses) == 1 else losses                        
            
    def get_metrics(self):
        return np.array(self.metrics)

    def fit(self, epochs):
        self.train_losses = []
        self.valid_losses = []
        self.metrics = []
        pbar = master_bar(range(epochs))
        pbar.write(["Epoch","train_loss","valid_loss"] + self.metric_names + ["time"], table=True)
        self.learner_callback.on_train_begin()
        for epoch in pbar:
            t1 = time()
            self.model.train()
            self.learner_callback.on_epoch_begin()
            
            for x_data,y_data in progress_bar(self.train_dl, parent=pbar):                
                x_data = x_data.to(self.device)
                y_data = y_data.to(self.device)                
                train_loss = self.loss_batch(x_data, y_data, is_train=True)                
                pbar.child.comment = str(round(train_loss,4))
                                            
            valid_loss = self.validate(self.valid_dl, parrent_bar=pbar)
            
            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)
            self.predict_smaple_func()
            met = self.learner_callback.on_epoch_end()
            
            t2 = time()            
            metrics_table = []
            if met is not None:
                self.metrics.append(met)
                for name,value in zip(self.metric_names, met):
                    metrics_table.append(f'{round(value,4):.4f}')                    
                    
            metrics_table.append(strftime('%M:%S', gmtime(round(t2-t1))))
                                                                        
            graphs = [[np.arange(len(self.train_losses)),np.array(self.train_losses)], [np.arange(len(self.valid_losses)),np.array(self.valid_losses)]]
            pbar.update_graph(graphs, [0,epochs], [0,np.array([self.train_losses,self.valid_losses]).max()])
            pbar.write([f'{epoch:04d}',f'{round(train_loss,4):.4f}',f'{round(valid_loss,4):.4f}'] + metrics_table, table=True)
            
        self.learner_callback.on_train_end()
        
        