from fastai.vision import *
import numpy as np
from .unet import UNet
from pathlib import Path
from random import sample
import pdb
import math
import pickle
import cv2
from nnlib.tools.pytorch_batchsize import * 

'''
params["paths"]["root_path"]
params["paths"]["heatmap_paths"]
params["paths"]["images_path"]
params["params"]["size"]
params["params"]["bs"]
params["params"]["norm_stats"]
params["params"]["transforms"]
params["files"]["file_filters"]
params["files"]["find_valid_imgs_func"]
params["metric"]["custom_metrics"]
params["metric"]["batch_end_func"]
params["metric"]["epoch_begin_func"]
params["metric"]["epoch_end_func"]
params["metric"]["metric_counter"]
params["misc"]["show_func"]
params["nn"]["unet_init_features"]
'''
numeric_metric = 1
accuracy_metric = 2

class NN_UNET_HEATMAP:
    def __init__(self, params):
        
        #static variables (they are fix)
        self.__heatmap_paths = params["paths"]["heatmap_paths"]
        for idx in range(len(self.__heatmap_paths)):
            self.__heatmap_paths[idx] = Path(self.__heatmap_paths[idx])
        self.__size = (params["params"]["size"],params["params"]["size"])
        self.__root_path = Path(params["paths"]["root_path"])
        self.__images_path = Path(params["paths"]["images_path"])
        self.__sample_results_path = Path(params["paths"]["sample_results_path"])
        self.__bs = params["params"]["bs"]
        self.__items_count = params["params"]["items_count"]
        self.__gpu_id = params["params"]["gpu_id"]
        self.__norm_stats = params["params"]["norm_stats"]
        self.__transforms = params["params"]["transforms"]
        self.__preload = params["params"]["preload"]
        self.__show_func = params["misc"]["show_func"]        
        self.__file_filters = np.array(params["files"]["file_filters"]) if params["files"]["file_filters"] is not None else None
        self.__unet_init_features = params["nn"]["unet_init_features"]        
        self.__batch_end_func = params["metric"]["batch_end_func"]
        self.__valid_images_file = params["files"]["valid_images_file"]
        self.__image_convert_mode = params["params"]["image_convert_mode"]
        self.__metric_counter = params["metric"]["metric_counter"]
        self.__sample_img = params["metric"]["sample_img"]
        self.__true_positive_threshold = params["metric"]["true_positive_threshold"]
        self.__type = params["nn"]["type"]
        
        # dynamic variables (they can change during learning)
        self.__lr = params["nn"]["learning_rate"]                
        self.__epochs = 0
        self.__train_losses = None
        self.__valid_losses = None
        self.__metrics = None
        self.__metric_names = None
        self.learner = None
                
    def initialize(self):
        torch.cuda.set_device(self.__gpu_id)
        self.__all_image_files = np.array(get_image_files(self.__root_path/self.__images_path))
        self.__all_image_files = self.__all_image_files if self.__items_count == -1 else self.__all_image_files[:self.__items_count]        
        self.__filter_files()
        self.__unet_in_channels = 1 if self.__image_convert_mode == "L" else 3
        self.__unet_out_channls = len(self.__heatmap_paths)        
        self.__valid_images = self.__find_valid_imgs_func()        
        self.__create_databunch()        
        self.__create_learner()
    
    def __extract_losses(self, epochs):
        idx = -1        
        step = int(len(self.learner.recorder.losses)/epochs)
        indxs = []
        while True:
            idx += step
            if idx >= len(self.learner.recorder.losses)-1:
                indxs.append(len(self.learner.recorder.losses)-1)
                break
            indxs.append(idx)
        train_losses = np.array(self.learner.recorder.losses)[np.array(indxs)]
        valid_losses = np.array(self.learner.recorder.val_losses)
        return train_losses,valid_losses
    
    def save_losses(self, filename=None, train=True,valid=True):
        if filename is None:
            filename = str(self.__sample_results_path)
                
        if train:
            np.save(self.__root_path/(filename+"_losses_train.npy"), self.__train_losses)
        if valid:
            np.save(self.__root_path/(filename+"_losses_valid.npy"), self.__valid_losses)
            
    def get_metric_names(self):
        if self.__metric_names is None:
            raise("Call `fit` Before metric names can be retrived")            
        return self.__metric_names
    
    def save_metrics(self,filename=None):
        if (self.__metric_names is None) and (self.__metrics is None):
            raise("Call `fit` Before metrics can be retrived")            
            
        if filename is None:
            filename = str(self.__sample_results_path)

        data = {"names":self.__metric_names,"metrics":self.__metrics}
        pickle.dump(data, open(self.__root_path/(filename+"_metrics.pkl"),"wb"))

    
    def get_metrics(self, specific_metric_names=None):        
        
        if (self.__metric_names is None) and (self.__metrics is None):
            raise("Call `fit` Before metrics can be retrived")            
        
        if specific_metric_names is None:
            return self.__metrics
        
        specific_metric_names = np.array(specific_metric_names)
        metric_idxs = []
        for metric in specific_metric_names:
            condition = np.where(self.__metric_names == metric)[0]
            if len(condition) != 1:
                continue
            metric_idxs.append(condition[0])
            
        metric_idxs = np.array(metric_idxs)
        if len(metric_idxs) == 0:
            raise("There are no matching `specific_metric_names`. Check orthography.")
                
        return np.array(self.__metrics)[:,metric_idxs]                            
        
    
    def get_losses(self, train=True, valid=True):
        result = []
        if train:
            result.append(self.__train_losses)
        
        if valid:
            result.append(self.__valid_losses)
            
        if len(result) == 1:
            return result[0]
        elif len(result) == 0:
            return None
        else:           
            return result
                
    def fit(self,epochs):
                
        if self.learner is None:
            print("learner not defined. Call initialize")
        
        self.learner.callback_fns[1].keywords["start_epoch"] = self.__epochs
        self.learner.fit_one_cycle(epochs, slice(self.__lr))
        self.__epochs = self.__epochs + epochs
        
        train_loss, valid_loss = self.__extract_losses(epochs)        
        if self.__train_losses is None:
            self.__train_losses = train_loss
        else:
            self.__train_losses = np.concatenate((self.__train_losses, train_loss))

        if self.__valid_losses is None:
            self.__valid_losses = valid_loss
        else:
            self.__valid_losses = np.concatenate((self.__valid_losses, valid_loss))                        
            
        metrics = np.array(self.learner.recorder.metrics)
        self.__metrics = np.concatenate((self.__metrics, metrics)) if self.__metrics is not None else metrics
        self.__metric_names = np.array(self.learner.recorder.metrics_names)
        
    def load_from_file(self,filename=None):
        if filename is None:
            filename = str(self.__sample_results_path)        
        
        self.learner = self.learner.load(filename+"_model")                
        data = pickle.load(open(self.__root_path/"models"/(filename+"_vars.pkl"),"rb")) 
        self.__epochs = data["epochs"]
        self.__train_losses = data["train_losses"]
        self.__valid_losses = data["valid_losses"]
        self.__metrics = data["metrics"]
        self.__metric_names = data["metric_names"]
    
    def save_to_file(self,filename=None):
        (self.__root_path/"models").mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            filename = str(self.__sample_results_path)
        
        self.learner.save(filename+"_model")
        pickle.dump({"epochs":self.__epochs,
                     "train_losses":self.__train_losses,
                     "valid_losses":self.__valid_losses,
                     "metrics":self.__metrics,
                     "metric_names":self.__metric_names}, 
                    open(self.__root_path/"models"/(filename+"_vars.pkl"),"wb"))
        
    def export_to_onnx(self, filename=None, batch_size=4):
        
        input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
        output_names = [ "output1" ]
        
        onnx_export = (self.__root_path/"onnx_export")
        onnx_export.mkdir(parents=True, exist_ok=True)
        
        batch = torch.zeros((batch_size,self.__unet_in_channels,self.__size[0],self.__size[0]), 
                           requires_grad=True, device="cuda:"+str(self.__gpu_id))
        
        for i in range(batch_size):
            batch[i] = self.learner.data.train_ds[i][0].data
            
        batch = normalize(batch,tensor(self.__norm_stats[0]).cuda(self.__gpu_id), 
                             tensor(self.__norm_stats[1]).cuda(self.__gpu_id))
        
        if filename is None:
            filename = onnx_export/(str(self.__sample_results_path)+".onnx")
        else:
            filename = onnx_export/(filename+".onnx")
        
        torch_out = self.learner.model(batch)
        torch.onnx.export(self.learner.model, batch,filename, 
                          verbose=True, input_names=input_names, output_names=output_names)
        
        def test_onnx():            
            import onnxruntime
            ort_session = onnxruntime.InferenceSession(str(filename))

            def to_numpy(tensor):
                return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

            # compute ONNX Runtime output prediction
            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(batch)}
            ort_outs = ort_session.run(None, ort_inputs)
            
            # compare ONNX Runtime and PyTorch results
            numpy_out = to_numpy(torch_out)            
            np.testing.assert_allclose(numpy_out, ort_outs[0], rtol=1e-02, atol=1e-02)

            print("Exported model has been tested with ONNXRuntime, and the result looks good!")        
        
        test_onnx()
            
    def set_lr(self,lr):
        self.__lr = lr
                
    def get_image_fname(self,fname):
        return open_image(self.__root_path/self.__images_path/fname ,convert_mode=self.__image_convert_mode)
    
    def get_train_label(self,idx):
        return self.learner.data.train_ds[idx][1]        

    def get_valid_label(self,idx):
        return self.learner.data.valid_ds[idx][1]        
    
    def get_train_image_idx(self,idx):
        return self.learner.data.train_ds[idx][0]        

    def get_valid_image_idx(self,idx):
        return self.learner.data.valid_ds[idx][0]            
    
    def get_train_ds_len(self):
        return len(self.learner.data.train_ds)

    def get_valid_ds_len(self):
        return len(self.learner.data.valid_ds)    
    
    def predict(self,image): 
        return self.learner.predict(image)[1]        
        
    def __find_valid_imgs_func(self):
        if not (self.__root_path/self.__valid_images_file).is_file():
            valid_images = [img.stem for img in sample(list(self.__all_image_files), int(len(self.__all_image_files)*0.2))]
            np.save(self.__root_path/self.__valid_images_file, valid_images)
        return list(np.load(self.__root_path/self.__valid_images_file))            
        
    def __filter_files(self):
        if self.__file_filters is None:
            return
        
        new_filenames = []
        for filename in self.__all_image_files:
            for ffilter in self.__file_filters:
                if filename.name.find((ffilter)) != -1:
                    new_filenames.append(filename)
                    break
        
        self.__all_image_files = np.array(new_filenames)        
        
        
    def __get_y_data(self,x):
        if self.__preload:
            return self.__get_y_data_preload(x)
        else:
            return self.__get_y_data_on_demand(x)
    
    def __get_y_data_preload(self,x):
        y_data = {"heatmaps":torch.zeros((len(self.__heatmap_paths), self.__size[0], self.__size[1]), dtype=torch.float32),
                  "keys":self.__heatmap_paths}
        
        for idx,heat_path in enumerate(self.__heatmap_paths): 
            heatmap = open_mask(self.__root_path/heat_path/x[0].name).resize(self.__size[0]).data
            if heatmap.max() == 0:
                heatmap = torch.zeros(heatmap.size(),dtype=torch.float32)
            else:                          
                heatmap = torch.from_numpy(np.interp(heatmap, (heatmap.min(), heatmap.max()), (0.0, 1.0)).astype(np.float32))

            y_data["heatmaps"][idx,:,:] = heatmap
        return y_data

    def __get_y_data_on_demand(self,x):
        y_data = {"heatmaps":[],
                  "keys":self.__heatmap_paths}        
        for idx,heat_path in enumerate(self.__heatmap_paths): 
            y_data["heatmaps"].append((self.__root_path/heat_path/x.name,self.__size[0]))
        return y_data

    
    def __valid_set_splitter(self,input):
        input = input[0] if self.__preload else input
        return input.stem in self.__valid_images    

    def __create_learner(self):
        
        sample_img = None
        if self.__sample_img is not None:
            img  = open_image(self.__root_path/self.__images_path/self.__sample_img, convert_mode=self.__image_convert_mode)
            masks = []
            for idx in range(len(self.__heatmap_paths)):                
                masks.append(open_mask(self.__root_path/self.__heatmap_paths[idx]/self.__sample_img).data/255)
            sample_img = (img,masks)
            
        callback_fns=[partial(heatmap_metric, heatmap_paths=self.__heatmap_paths,
                              batch_end_func=self.__batch_end_func, metric_counter=self.__metric_counter, 
                              true_positive_threshold = self.__true_positive_threshold, sample_img = sample_img,
                              sample_results_path = self.__root_path/"sample_results"/self.__sample_results_path, start_epoch=0)]
        
        if self.__type == "res_unet++":
             net = UNet(in_channels = self.__unet_in_channels, out_channels = self.__unet_out_channls,
                        init_features = self.__unet_init_features, resblock=True, squeeze_excite=True, 
                        aspp=True, attention=True, bn_relu_at_first=True, bn_relu_at_end=False)
        elif self.__type == "res_unet_bn_relu_end":
             net = UNet(in_channels = self.__unet_in_channels, out_channels = self.__unet_out_channls,
                        init_features = self.__unet_init_features, resblock=True, squeeze_excite=False, 
                        aspp=False, attention=False, bn_relu_at_first=False, bn_relu_at_end=True)            
        elif self.__type == "attention_unet":
             net = UNet(in_channels = self.__unet_in_channels, out_channels = self.__unet_out_channls,
                        init_features = self.__unet_init_features, resblock=False, squeeze_excite=False, 
                        aspp=False, attention=True, bn_relu_at_first=False, bn_relu_at_end=True)            
        elif self.__type == "aspp_unet":
             net = UNet(in_channels = self.__unet_in_channels, out_channels = self.__unet_out_channls,
                        init_features = self.__unet_init_features, resblock=False, squeeze_excite=False, 
                        aspp=True, attention=False, bn_relu_at_first=False, bn_relu_at_end=True)            
        elif self.__type == "squeeze_excite_unet":
             net = UNet(in_channels = self.__unet_in_channels, out_channels = self.__unet_out_channls,
                        init_features = self.__unet_init_features, resblock=False, squeeze_excite=True, 
                        aspp=False, attention=False, bn_relu_at_first=False, bn_relu_at_end=True)            
        elif self.__type == "res_unet_bn_relu_first":
             net = UNet(in_channels = self.__unet_in_channels, out_channels = self.__unet_out_channls,
                        init_features = self.__unet_init_features, resblock=True, squeeze_excite=False, 
                        aspp=False, attention=False, bn_relu_at_first=True, bn_relu_at_end=False)            
        elif self.__type == "unet":
            net = UNet(in_channels = self.__unet_in_channels, out_channels = self.__unet_out_channls,
                        init_features = self.__unet_init_features, resblock=False, squeeze_excite=False, 
                        aspp=False, attention=False, bn_relu_at_first=False, bn_relu_at_end=True)
        else:
            raise("Net type ´"+ self.__type+"´ is not implemented!")
                                       
        self.learner =  Learner(self.__bunch, net, path=self.__root_path, callback_fns=callback_fns)          
                
        if self.__bs == -1:            
            max_batch_estimator = MAX_BATCH_SIZE_ESTIMATER(net=net, opt=self.learner.opt_func.func,
                                                           loss_func=self.learner.loss_func, gpu_id=self.__gpu_id, 
                                                           input_features=self.__unet_in_channels, input_size_x=self.__size[1],
                                                           input_size_y=self.__size[0])
            self.learner.data.batch_size = max_batch_estimator.find_max_bs()
                
    
    
    def __create_databunch(self):        
        if self.__preload:
            image_list_input = []
            for img_file in self.__all_image_files:
                image_list_input.append((img_file,open_image(img_file, convert_mode=self.__image_convert_mode)))            
                        
        else:
            image_list_input = self.__all_image_files
            
                
        data = HeatsImageList.from_list(image_list_input,path=self.__images_path, 
                                        convert_mode=self.__image_convert_mode, preload=self.__preload)        
        
        data = data.split_by_valid_func(self.__valid_set_splitter)        
        data = data.label_from_func(func=self.__get_y_data, label_cls=None, show_func = self.__show_func, 
                                    preload=self.__preload, gpu_id=self.__gpu_id)
        data = data.transform(self.__transforms, size=self.__size, tfm_y=True)
        bs = 1 if self.__bs == -1 else self.__bs        
        self.__bunch = data.databunch(bs=bs).normalize(self.__norm_stats, do_y=False)        
        
        
class heatmap_metric(LearnerCallback):
    _order=-20 # Needs to run before the recorder
    def __init__(self, learn, heatmap_paths, batch_end_func, metric_counter, true_positive_threshold, sample_img, sample_results_path, start_epoch=0):        
        super().__init__(learn)
        
        self.__counter_epoch = start_epoch
        self.__heatmap_paths = heatmap_paths
        self.__metric_counter = metric_counter
        self.__true_positive_threshold = true_positive_threshold
        self.__custom_metrics = {"metrics":[],"types":[]}
        self.__sample_img = sample_img
        self.__sample_results_path = sample_results_path
        
        for item in self.__heatmap_paths:
            item = str(item)
            self.__custom_metrics["metrics"].append(item+"_pos_train")
            self.__custom_metrics["types"].append(numeric_metric)
            
            self.__custom_metrics["metrics"].append(item+"_pos_valid")
            self.__custom_metrics["types"].append(numeric_metric)
            
            self.__custom_metrics["metrics"].append(item+"_accuracy_train")
            self.__custom_metrics["types"].append(accuracy_metric)
            
            self.__custom_metrics["metrics"].append(item+"_accuracy_valid")
            self.__custom_metrics["types"].append(accuracy_metric)
                    
        self.__batch_end_func = batch_end_func

    def on_train_begin(self, **kwargs):    
        
        self.learn.recorder.add_metric_names(self.__custom_metrics["metrics"])
    
    def on_batch_end(self, last_output, last_target, **kwargs):
                 
        if self.__counter_epoch % self.__metric_counter == 0:            
            last_target = last_target.detach().cpu().numpy()
            last_output = last_output.detach().cpu().numpy()

            for target_batch,output_batch in zip(last_target, last_output):
                self.metrics_values = self.__batch_end_func(target_batch,output_batch, self.__heatmap_paths, 
                                                            self.metrics_values, kwargs["train"], self.__true_positive_threshold)
    
    def on_epoch_begin(self, **kwargs):                               
        if self.__counter_epoch % self.__metric_counter == 0:            
            self.metrics_values = {}
            for idx,metric in enumerate(self.__custom_metrics["metrics"]):
                if self.__custom_metrics["types"][idx] == numeric_metric:             
                    self.metrics_values[metric] = []
                else:
                    self.metrics_values[metric] = {"total_targets":0,"total_true_positives":0}                            
  

    def predict_sample(self):
        
        sample_img = self.__sample_img
        if sample_img is not None:
            
            channels = sample_img[0].data.shape[0]
            shape = sample_img[0].data.shape[1]
            img = np.zeros((shape*2,shape+(shape*len(sample_img[1])),channels), dtype=np.float)        
            img[:shape,:shape,:] = sample_img[0].data.permute(1,2,0)
            for idx in range(len(sample_img[1])):
                img[:shape,shape*(idx+1):shape*(idx+2),:] = sample_img[1][idx].repeat(channels,1,1).permute(1,2,0)
                
            res = self.predict(sample_img[0])[0].data.detach().cpu()

            for idx in range(res.shape[0]):
                img[shape:shape*2,shape*(idx+1):shape*(idx+2),:] = res[idx][None].repeat(channels,1,1).permute(1,2,0)
        
            img = np.round(np.interp(img, (img.min(), img.max()), (0, 255))).astype(np.uint8)            
            if channels == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
            sample_results_path = self.__sample_results_path
            counter_epoch = self.__counter_epoch
            
            sample_results_path.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(sample_results_path/("epoch_"+str(counter_epoch)+".png")), img)

    
    def on_epoch_end(self, last_metrics, **kwargs):        
        self.predict_sample()                            
        metrics = list(np.zeros(len(self.__custom_metrics["metrics"]), dtype=np.float32))
        try:
            if self.__counter_epoch % self.__metric_counter == 0:
                for idx,metric in enumerate(self.__custom_metrics["metrics"]):
                    if self.__custom_metrics["types"][idx] == numeric_metric:
                        if len(self.metrics_values[metric]) == 0:
                            metrics[idx] = 0
                        else:
                            metrics[idx] = np.array(self.metrics_values[metric]).mean()
                    else:                
                        metrics[idx] = self.metrics_values[metric]["total_true_positives"] / self.metrics_values[metric]["total_targets"]   
        except:            
            print("X")
                    
        self.__counter_epoch += 1
                
        return add_metrics(last_metrics, (metrics))        

class HeatLoss(nn.Module):
    def __init__(self, gpu_id):
        super().__init__()
        self._device = "cuda:"+str(gpu_id)
    
    def forward(self, input, target):                         
        if target.shape[1] > 1:
            diffs = torch.zeros(target.shape[1]*3, dtype=torch.float32, device=torch.device(self._device))
        else:
            diffs = torch.zeros(target.shape[1]*2, dtype=torch.float32, device=torch.device(self._device))
                   
        for i in range(target.shape[1]):            
            entry_t = target[:,i,:,:]            
            
            if target.shape[1] > 1:
                other_mask = torch.ones(target.shape[1], dtype=torch.bool, device=torch.device(self._device))
                other_mask[i] = False
                entry_t_other = target[:,other_mask,:,:]
                entry_t_m_other = entry_t_other > 0 
                entry_t_m_other = torch.any(entry_t_m_other, axis=1)
                                    
            entry_t_m = entry_t > 0
            entry_t_m_back = torch.logical_not(entry_t_m)
            
            entry_i = input[:,i,:,:]

            diff_entry = torch.abs(entry_i[entry_t_m] - entry_t[entry_t_m]) 
            diff_entry_back = torch.abs(entry_i[entry_t_m_back] - entry_t[entry_t_m_back]) 
            
            if target.shape[1] > 1:
                diff_entry_other = torch.abs(entry_i[entry_t_m_other] - entry_t[entry_t_m_other]) 

            if len(diff_entry) > 0:                
                diffs[i] = torch.mean(diff_entry)
                
            if len(diff_entry_back) > 0:
                diffs[i+target.shape[1]] = torch.mean(diff_entry_back)
            
            if target.shape[1] > 1:
                if len(diff_entry_other) > 0:
                    diffs[i+(target.shape[1]*2)] = torch.mean(diff_entry_other)
        
        
        if target.shape[1] > 1:
            return diffs.sum()/(target.shape[1]*3)
        else:
            return diffs.sum()/(target.shape[1]*2)


class HeatMaps(ItemBase):    
        
    def __init__(self, y_data, show_func):       
        self._y_data = y_data
        self._show_func = show_func
                                
    def clone(self):
        new_y_data = {"heatmaps":self._y_data["heatmaps"].clone(),"keys":self._y_data["keys"]}
        return self.__class__(new_y_data, self._show_func)

    @property
    def data(self):
        return self._y_data["heatmaps"]
    
    @property
    def obj(self):
        return self._y_data["heatmaps"]
    
    def prepare_for_visual(self):        
        return Image(torch.sum(self._y_data["heatmaps"], dim=0).repeat(3,1,1))
                
    def apply_tfms(self, tfms, **kwargs): 
        
        new_heat = self.clone()                        
        
        new_tfms = []
        for tfm in tfms:
            if type(tfm.tfm) != TfmLighting:
                new_tfms.append(tfm)
                        
        for i in range(len(new_heat._y_data["heatmaps"])): 
            new_heat._y_data["heatmaps"][i] = Image(new_heat._y_data["heatmaps"][i].repeat(3,1,1)).apply_tfms(new_tfms, do_resolve=False, size=kwargs["size"]).data[0,:,:]
                                   
        return new_heat   
    
    def show(self, ax:plt.Axes, **kwargs):        
        self._show_func(ax,self._y_data["heatmaps"], self._y_data["keys"])
                                
    def __repr__(self): 
        return f'{self.__class__.__name__} {tuple(self._y_data["heatmaps"].shape)}'

class HeatsProcessor(PreProcessor):
    "`PreProcessor` that stores the number of targets (heatmaps)."
    def __init__(self, ds:ItemList):                    
        self.c = len(ds.items[0]["keys"])
    def process(self, ds:ItemList):  
        ds.c = self.c

class HeatsLabelList(ItemList):
    "`ItemList` for heat maps."
    _processor = HeatsProcessor
    
    def __init__(self, items:Iterator, **kwargs):           
        show_func =  kwargs["show_func"]
        preload =  kwargs["preload"]
        gpu_id = kwargs["gpu_id"]
        del kwargs["show_func"]
        del kwargs["preload"]
        del kwargs["gpu_id"]
        super().__init__(items, **kwargs)
        self._keys = items[0]["keys"]
        self.loss_func = HeatLoss(gpu_id)
        self._show_func = show_func
        self._preload = preload
        self._gpu_id = gpu_id
        self.copy_new += ['_preload','_show_func', '_gpu_id']
        
    def get_preload(self, i):        
        y_data = super().get(i)
        return HeatMaps(y_data, self._show_func)    
    
    def get(self,i):
        #pdb.set_trace()
        if self._preload:
            return self.get_preload(i)
        else:
            return self.get_on_demand(i)
    
    def get_on_demand(self, i):           
        y_data = super().get(i)
        heatmaps = torch.zeros((len(y_data["heatmaps"]), y_data["heatmaps"][0][1], y_data["heatmaps"][0][1]), dtype=torch.float32)
        idx = 0
        for fname,size in y_data["heatmaps"]:
            heatmap = open_mask(fname).resize(size).data
            if heatmap.max() == 0:
                heatmap = torch.zeros(heatmap.size(),dtype=torch.float32)
            else:                          
                heatmap = torch.from_numpy(np.interp(heatmap, (heatmap.min(), heatmap.max()), (0.0, 1.0)).astype(np.float32))
            
            heatmaps[idx,:,:] = heatmap    
            idx += 1
        new_y_data = {"heatmaps":heatmaps,"keys":self._keys}
        
        return HeatMaps(new_y_data, self._show_func)

    def analyze_pred(self, pred, thresh:float=0.5):              
        return pred
                
    def reconstruct(self, t, x):
        y_data ={"heatmaps":t,"keys":self._keys}
        return HeatMaps(y_data, self._show_func)
    
    @classmethod
    def from_list(cls, labellist, **kwargs):        
        return cls(imagelist, **kwargs)
        
class HeatsImageList(ImageList):    
    _label_cls = HeatsLabelList

    def __init__(self, items:Iterator, **kwargs):                
        self.preload =  kwargs["preload"]
        del kwargs["preload"]
        super().__init__(items, **kwargs) 
        self.copy_new += ['preload']
            
    def get(self,i):
        
        item = ItemList.get(self,i)
        if self.preload:
            image = item[1]
        else:
            image = self.open(item)
        
        return image
    
    def open(self, fn):        
        return open_image(fn, convert_mode=self.convert_mode, after_open=self.after_open)    
    
    def show_xys(self, xs, ys, figsize:Tuple[int,int]=(9,10), **kwargs):
        
        "Show the `xs` and `ys` on a figure of `figsize`. `kwargs` are passed to the show method."
        rows = int(math.sqrt(len(xs)))
        fig, axs = plt.subplots(rows,rows,figsize=figsize)
        for i, ax in enumerate(axs.flatten() if rows > 1 else [axs]):
            xs[i].show(ax=ax, y=ys[i], **kwargs)
        plt.tight_layout()

    def show_xyzs(self, xs, ys, zs, figsize:Tuple[int,int]=None, **kwargs):
        """Show `xs` (inputs), `ys` (targets) and `zs` (predictions) on a figure of `figsize`. 
        `kwargs` are passed to the show method."""
        #pdb.set_trace()
        figsize = ifnone(figsize, (6,3*len(xs)))
        fig,axs = plt.subplots(len(xs), 2, figsize=figsize)
        fig.suptitle('Ground truth / Predictions', weight='bold', size=14)
        for i,(x,y,z) in enumerate(zip(xs,ys,zs)):
            x.show(ax=axs[i,0], y=y, **kwargs)
            x.show(ax=axs[i,1], y=z, **kwargs)
    
    @classmethod
    def from_list(cls, imagelist, path=".", **kwargs):    
        #pdb.set_trace()
        return cls(imagelist, path=Path(path), **kwargs)    