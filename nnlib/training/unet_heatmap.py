
import numpy as np
#from .unet import UNet
from pathlib import Path
from random import sample
import pdb
import math
from math import ceil
import pickle
import cv2
from ..tools.pytorch_batchsize import *
from ..tools.heatmap_to_points import *
from ..tools.helper import *
from ..tools.image_tools import *
from .basic import *
from .unet import *
from PIL import Image
import glob
import sys
from fastprogress.fastprogress import master_bar, progress_bar
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
#from .basic import *
from torch import nn
import random
import platform
import matplotlib.pyplot as plt
import pickle
import os
import sys

__all__ = ["DataAugmentation", "HeatmapLearner", "HeatLoss"]

class CustomHeatmapDataset(Dataset):
    "CustomImageDataset with `image_files`,`y_func`, `convert_mode` and `size(height, width)`"
    def __init__(self, data, size=(512,512), grayscale=False, normalize_mean=None, normalize_std=None, data_aug=None, 
                 is_valid=False, do_normalize=True, clahe=True):
        
        self.data = data        
        self.size = size
        
        if normalize_mean is None:
            if grayscale:
                normalize_mean = [0.131]
            else:
                normalize_mean = [0.485, 0.456, 0.406]

        if normalize_std is None:
            if grayscale:
                normalize_std = [0.308]
            else:
                normalize_std = [0.229, 0.224, 0.225]
                                
        self.normalize = transforms.Normalize(normalize_mean,normalize_std)
        self.convert_mode = "L" if grayscale else "RGB"        
        self.data_aug = data_aug
        self.to_t = transforms.ToTensor()
        self.is_valid = is_valid
        self.do_normalize = do_normalize
        
        self.clahe = cv2.createCLAHE(clipLimit=20.0,tileGridSize=(30,30)) if clahe else None
    
    def __len__(self):
        return len(self.data)

    def load_labels(self,idx):
        heatmaps = []        
        for fname in self.data[idx][1]:
            heatmaps.append(load_heatmap(fname, size=self.size))        
        return heatmaps
    
    def apply_clahe(self,img):
        if self.clahe == None:
            return img
        img = np.asarray(img)
        img = self.clahe.apply(img)
        return Image.fromarray(img)
                
    def __getitem__(self, idx):          
        
        image = load_image(self.data[idx][0], size=self.size, convert_mode=self.convert_mode, to_numpy=False)        
        labels = self.load_labels(idx)
        
        image = self.apply_clahe(image)
        
        if (not self.is_valid) and (self.data_aug is not None):
            image, labels = self.data_aug.transform(image, labels)        
                
        labels = self.to_t(np.stack(labels, axis=2))
        image = self.to_t(image)
        
        if self.do_normalize:
            image = self.normalize(image)
                
        return image,labels    
        
class RandomRotationImageTarget(transforms.RandomRotation):
    def __call__(self, img, targets):    
        angle = self.get_params(self.degrees)
        img = transforms.functional.rotate(img, angle, self.resample, self.expand, self.center)
        for idx in range(len(targets)):
            targets[idx] = transforms.functional.rotate(targets[idx], angle, self.resample, self.expand, self.center)
        return img, targets
    
class RandomHorizontalFlipImageTarget(transforms.RandomHorizontalFlip):
    def __call__(self, img, targets):    
        if random.random() < self.p:
            img = transforms.functional.hflip(img)
            for idx in range(len(targets)):
                targets[idx] = transforms.functional.hflip(targets[idx])
        return img,targets

class RandomVerticalFlipImageTarget(transforms.RandomVerticalFlip):
    def __call__(self, img, targets):    
        if random.random() < self.p:
            img = transforms.functional.vflip(img)
            for idx in range(len(targets)):
                targets[idx] = transforms.functional.vflip(targets[idx])
        return img,targets

class RandomPerspectiveImageTarget(transforms.RandomPerspective):
    def __call__(self, img, targets): 
        if not transforms.functional._is_pil_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        if random.random() < self.p:
            width, height = img.size
            startpoints, endpoints = self.get_params(width, height, self.distortion_scale)
            img = transforms.functional.perspective(img, startpoints, endpoints, self.interpolation)

            for idx in range(len(targets)):
                targets[idx] = transforms.functional.perspective(targets[idx], startpoints, endpoints, self.interpolation)

        return img,targets

class ComposeImageTarget(transforms.Compose):
    def __call__(self, img, targets): 
            for t in self.transforms:
                img,targets = t(img, targets)
            return img,targets 
        
        
class DataAugmentation:
    "DataAugmentation class with `size(height,width)`"
    def __init__(self, rotation=10,horizontal_flip=0.5,
                 vertical_flip=0.5,warp=0.1, zoom=0.8, brightness=0.5, contrast=0.5, GaussianBlur=1):
                 
        
        self.lightning_transforms = transforms.Compose([transforms.ColorJitter(brightness=brightness,contrast=contrast),
                                                        #transforms.GaussianBlur(kernel_size=GaussianBlur)
                                                       ])

        self.affine_transforms = ComposeImageTarget([                                                     
                                                     RandomRotationImageTarget((-rotation,rotation)),
                                                     RandomHorizontalFlipImageTarget(horizontal_flip),
                                                     RandomVerticalFlipImageTarget(vertical_flip),
                                                     RandomPerspectiveImageTarget(distortion_scale=warp),
                                                     #transforms.RandomResizedCrop(size=size,scale=(zoom,1.0),ratio=(1.0,1.0))
                                                    ])        
        
    def transform(self,features,labels):
                
        #do lighting transforms for features
        features = self.lightning_transforms(features)
        
        #do affine transforms for features and labels
        features,labels = self.affine_transforms(features, labels)            
        
        return features,labels      
    
        
class heatmap_metric(LearnerCallback):    
    def __init__(self, features, true_positive_threshold=10, metric_counter=1):        
                     
        self.__counter_epoch = 0        
        self.__metric_counter = metric_counter        
        self.__custom_metrics = {"metrics":[],"types":[]}
        self.__features = features
        self.__true_positive_threshold = true_positive_threshold
        self.numeric_metric = 1
        self.accuracy_metric = 2

        
        for item in self.__features.keys():            
            self.__custom_metrics["metrics"].append(item+"_pos_train")
            self.__custom_metrics["types"].append(self.numeric_metric)
            
            self.__custom_metrics["metrics"].append(item+"_pos_valid")
            self.__custom_metrics["types"].append(self.numeric_metric)
            
            self.__custom_metrics["metrics"].append(item+"_accuracy_train")
            self.__custom_metrics["types"].append(self.accuracy_metric)
            
            self.__custom_metrics["metrics"].append(item+"_accuracy_valid")
            self.__custom_metrics["types"].append(self.accuracy_metric)
                            

    def get_metric_names(self):
        return self.__custom_metrics["metrics"]
        
    def __calc_metrics(self, targets, outputs, metric_values, train):    
        ext = "train" if train else "valid"    
        for target,output,feature  in zip(targets, outputs, list(self.__features.keys())):                        

            type_of = self.__features[feature]["type"]
            if (type_of == "circle"):

                    points_target = heatmap_to_circle(target)
                    points_output = heatmap_to_circle(output)
                    if (points_target is not None):                            
                        metric_values[feature+"_accuracy_"+ext]["total_targets"] += 1 

                    if (points_target is not None) and (points_output is not None):
                        mid_point_output = np.round(np.sum(points_output, axis=0)/len(points_output)).astype(np.int)
                        mid_point_target = np.round(np.sum(points_target, axis=0)/len(points_target)).astype(np.int)
                        diff_circle_midpoint = np.sqrt(np.sum((mid_point_output - mid_point_target)**2))

                        metric_values[feature+"_pos_"+ext].append(diff_circle_midpoint)                                
                        if diff_circle_midpoint < self.__true_positive_threshold:                                
                            metric_values[feature+"_accuracy_"+ext]["total_true_positives"] += 1 

            elif type_of == "single_point":
                    center_point_target = heatmap_to_max_confidence_point(target)
                    center_point_output = heatmap_to_max_confidence_point(output)
                    if (center_point_target is not None):                        
                        metric_values[feature+"_accuracy_"+ext]["total_targets"] += 1 

                    if (center_point_target is not None) and (center_point_output is not None):
                        diff_center = np.sqrt(np.sum((center_point_output - center_point_target)**2))
                        metric_values[feature+"_pos_"+ext].append(diff_center)
                        if diff_center < self.__true_positive_threshold:                                                        
                            metric_values[feature+"_accuracy_"+ext]["total_true_positives"] += 1 


            elif type_of == "multi_point":                    
                    all_peaks_target = heatmap_to_multiple_points(target)
                    all_peaks_output = heatmap_to_multiple_points(output)

                    if (all_peaks_target is not None):
                        metric_values[feature+"_accuracy_"+ext]["total_targets"] += len(all_peaks_target)

                    if (all_peaks_target is not None) and (all_peaks_output is not None):
                        diffs =  []
                        for peak_target in all_peaks_target:
                            if len(all_peaks_output) == 0:
                                break
                            s = np.argmin(np.sqrt(np.sum((all_peaks_output - peak_target)**2, axis=1)))
                            diffs.append(np.sqrt(np.sum((all_peaks_output[s] - peak_target)**2)))
                            if diffs[-1] < self.__true_positive_threshold:
                                metric_values[feature+"_accuracy_"+ext]["total_true_positives"] += 1

                            all_peaks_output = np.delete(all_peaks_output, s, axis=0)

                        diff_nut_edges = np.array(diffs).mean()
                        metric_values[feature+"_pos_"+ext].append(diff_nut_edges)
            else:
                raise("The Heatmaptype " + type_of + " is not implemented yet.")



        return metric_values       
    
    def on_batch_end(self, last_output, last_target, train):
                         
        if self.__counter_epoch % self.__metric_counter == 0:            
            last_target = last_target.numpy()
            last_output = last_output.numpy()

            for target_batch,output_batch in zip(last_target, last_output):
                self.metrics_values = self.__calc_metrics(target_batch,output_batch, 
                                                            self.metrics_values, train)
    
    def on_epoch_begin(self):                               
        if self.__counter_epoch % self.__metric_counter == 0:            
            self.metrics_values = {}
            for idx,metric in enumerate(self.__custom_metrics["metrics"]):
                if self.__custom_metrics["types"][idx] == self.numeric_metric:             
                    self.metrics_values[metric] = []
                else:
                    self.metrics_values[metric] = {"total_targets":0,"total_true_positives":0}                            
  
    
    def on_epoch_end(self):        
        metrics = list(np.zeros(len(self.__custom_metrics["metrics"]), dtype=np.float32))
        
        if self.__counter_epoch % self.__metric_counter == 0:
            for idx,metric in enumerate(self.__custom_metrics["metrics"]):
                if self.__custom_metrics["types"][idx] == self.numeric_metric:
                    if len(self.metrics_values[metric]) == 0:
                        metrics[idx] = 0
                    else:
                        metrics[idx] = np.array(self.metrics_values[metric]).mean()
                else:                
                    if self.metrics_values[metric]["total_targets"] != 0:
                        metrics[idx] = self.metrics_values[metric]["total_true_positives"] / self.metrics_values[metric]["total_targets"]   
                    else:
                        metrics[idx] = 0
                    
        self.__counter_epoch += 1
                
        return metrics

    
class HeatLoss(nn.Module):
    
    def forward(self, input, target):                         
        if target.shape[1] > 1:
            diffs = torch.zeros(target.shape[1]*3, dtype=torch.float32, device=input.device)
        else:
            diffs = torch.zeros(target.shape[1]*2, dtype=torch.float32, device=input.device)
                   
        for i in range(target.shape[1]):            
            entry_t = target[:,i,:,:]            
            
            if target.shape[1] > 1:
                other_mask = torch.ones(target.shape[1], dtype=torch.bool, device=input.device)
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

class HeatmapLearner:
    def __init__(self, features, root_path, images_path, size=(512,512), bs=-1, items_count=-1, gpu_id=0, 
                 norm_stats=None, data_aug=None, preload=False, sample_results_path="sample_results",
                 unet_init_features=16, valid_images_store="valid_images.npy", image_convert_mode="L", metric_counter=1, 
                 sample_img=None, true_positive_threshold=0.02, ntype="unet", lr=1e-03, file_filters=None, clahe=False):
        
        r"""Class for train an Unet-style Neural Network, for heatmap based image recognition

        Args:
            size       : Size of images to pass through the neural network. The sizes must be a power of two (Heigth, Width).
            norm_stats : Normalize values for images in the form (mean,std).
        Example:

        """        
        
        # check assertions
        assert power_of_2(size), "size must be a power of 2, to work with this class"
        
        #static variables (they are fix)                
        heatmap_paths = list(features.keys())
        for idx in range(len(heatmap_paths)):
            heatmap_paths[idx] = Path(heatmap_paths[idx])
                            
        self.features = features
        self.__size = size
        self.__root_path = Path(root_path)
        self.__images_path = Path(images_path)
        self.__sample_results_path = Path(sample_results_path)  
        (self.__root_path/self.__sample_results_path).mkdir(parents=True, exist_ok=True)
        
        self.__gpu_id = gpu_id
        
        data_aug = DataAugmentation() if data_aug is None else data_aug
        
        if norm_stats is None:
            norm_stats = ([0.131],[0.308]) if image_convert_mode == "L" else ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                        
        if sample_img is None:
            t_img_files = glob.glob(str(self.__root_path/self.__images_path/"*"))
            idx = random.randint(0, len(t_img_files)-1)
            self.__sample_img = Path(t_img_files[idx]).name            
        else:
            self.__sample_img = sample_img
        
        true_positive_threshold = round(true_positive_threshold*self.__size[0])        
        
        # dynamic variables (they can change during learning)        
        self.__epochs = 0
        self.__train_losses = None
        self.__valid_losses = None
        self.__metrics = None        
        
        file_filters = np.array(file_filters) if file_filters is not None else None
        
        self.__create_learner(file_filters=file_filters, valid_images_store=valid_images_store, items_count=items_count,
                              features=features, bs=bs, data_aug=data_aug,image_convert_mode=image_convert_mode, 
                              heatmap_paths=heatmap_paths, true_positive_threshold=true_positive_threshold, metric_counter=metric_counter,
                              lr=lr, clahe=clahe, norm_stats=norm_stats, unet_init_features=unet_init_features, ntype=ntype)
        
    def __create_learner(self, file_filters, valid_images_store, items_count, features, bs, data_aug, 
                         image_convert_mode, heatmap_paths, true_positive_threshold, metric_counter, lr, clahe, norm_stats, 
                         unet_init_features, ntype):
                                
        training_data, valid_data = self.__load_data(features=features,file_filters=file_filters,
                                                     valid_images_store=valid_images_store, items_count=items_count)                       

        torch.cuda.set_device(self.__gpu_id)                        
        self.__unet_in_channels = 1 if image_convert_mode == "L" else 3
        self.__unet_out_channls = len(heatmap_paths)                                                    

        self.train_dataset = CustomHeatmapDataset(data=training_data, grayscale=image_convert_mode == "L", 
                                                  normalize_mean=norm_stats[0],normalize_std=norm_stats[1], data_aug=data_aug, clahe=clahe)

        self.valid_dataset = CustomHeatmapDataset(data=valid_data, grayscale=image_convert_mode == "L",
                                                  normalize_mean=norm_stats[0],normalize_std=norm_stats[1], is_valid=True, clahe=clahe)                                


        sample_img = None
        if self.__sample_img is not None:
            to_t = transforms.ToTensor()
            img  = to_t(load_image(self.__root_path/self.__images_path/self.__sample_img,
                                   convert_mode=image_convert_mode, size=self.__size, to_numpy=False))
            masks = []
            for idx in range(len(heatmap_paths)):                
                heat = to_t(load_heatmap(self.__root_path/heatmap_paths[idx]/self.__sample_img))                
                masks.append(heat)
            sample_img = (img,masks)

        metric = heatmap_metric(features = features, true_positive_threshold = true_positive_threshold, metric_counter = metric_counter)
        
        net = self.__get_net(ntype, unet_init_features).to(torch.device("cuda:"+str(self.__gpu_id)))
        opt = torch.optim.Adam(net.parameters(), lr) 
        loss_func = HeatLoss().to(torch.device("cuda:"+str(self.__gpu_id)))    
        
        if bs == -1:              
            batch_estimator = Batch_Size_Estimator(net=net, opt=opt,loss_func=loss_func, gpu_id=self.__gpu_id,
                                                           dataset = self.train_dataset)
            bs = batch_estimator.find_max_bs()  
                     
            
        train_dl = DataLoader(self.train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers(), pin_memory=True)
        valid_dl = DataLoader(self.valid_dataset, batch_size=bs, shuffle=True, num_workers=num_workers(), pin_memory=True)                                                       

        self.learner = Learner(model=net,train_dl=train_dl, valid_dl=valid_dl,
                               loss_func=loss_func,optimizer=opt, learner_callback= metric)                 
        
   
    def __get_net(self, ntype, unet_init_features):
        if ntype == "res_unet++":
             net = UNet(in_channels = self.__unet_in_channels, out_channels = self.__unet_out_channls,
                        init_features = unet_init_features, resblock=True, squeeze_excite=True, 
                        aspp=True, attention=True, bn_relu_at_first=True, bn_relu_at_end=False)
        elif ntype == "res_unet_bn_relu_end":
             net = UNet(in_channels = self.__unet_in_channels, out_channels = self.__unet_out_channls,
                        init_features = unet_init_features, resblock=True, squeeze_excite=False, 
                        aspp=False, attention=False, bn_relu_at_first=False, bn_relu_at_end=True)            
        elif ntype == "attention_unet":
             net = UNet(in_channels = self.__unet_in_channels, out_channels = self.__unet_out_channls,
                        init_features = unet_init_features, resblock=False, squeeze_excite=False, 
                        aspp=False, attention=True, bn_relu_at_first=False, bn_relu_at_end=True)            
        elif ntype == "aspp_unet":
             net = UNet(in_channels = self.__unet_in_channels, out_channels = self.__unet_out_channls,
                        init_features = unet_init_features, resblock=False, squeeze_excite=False, 
                        aspp=True, attention=False, bn_relu_at_first=False, bn_relu_at_end=True)            
        elif ntype == "squeeze_excite_unet":
             net = UNet(in_channels = self.__unet_in_channels, out_channels = self.__unet_out_channls,
                        init_features = unet_init_features, resblock=False, squeeze_excite=True, 
                        aspp=False, attention=False, bn_relu_at_first=False, bn_relu_at_end=True)            
        elif ntype == "res_unet_bn_relu_first":
             net = UNet(in_channels = self.__unet_in_channels, out_channels = self.__unet_out_channls,
                        init_features = unet_init_features, resblock=True, squeeze_excite=False, 
                        aspp=False, attention=False, bn_relu_at_first=True, bn_relu_at_end=False)            
        elif ntype == "unet":
            net = UNet(in_channels = self.__unet_in_channels, out_channels = self.__unet_out_channls,
                        init_features = unet_init_features, resblock=False, squeeze_excite=False, 
                        aspp=False, attention=False, bn_relu_at_first=False, bn_relu_at_end=True)
        else:
            raise("Net type ´"+ ntype+"´ is not implemented!")     
            
        return net
    
    def __load_data(self, features, file_filters, valid_images_store, items_count):
        
        def find_valid_imgs_func(all_image_files):
            if not (self.__root_path/valid_images_store).is_file():
                valid_images = [img for img in sample(list(all_image_files), int(len(all_image_files)*0.2))]
                np.save(self.__root_path/valid_images_store, valid_images, allow_pickle=True)
            return list(np.load(self.__root_path/valid_images_store, allow_pickle=True))            

        def filter_files(all_image_files):
            if file_filters is None:
                return all_image_files

            new_filenames = []
            for filename in all_image_files:
                for ffilter in file_filters:
                    if filename.name.find((ffilter)) != -1:
                        new_filenames.append(filename)
                        break

            return np.array(new_filenames) 


        all_image_files = list((self.__root_path/self.__images_path).glob("*.png"))
        all_image_files = all_image_files if items_count == -1 else all_image_files[:items_count] 
        
        all_image_files = filter_files(all_image_files)
        valid_images_files = find_valid_imgs_func(all_image_files)
                                                
        training_data = []
        valid_data = []
        
        pbar = progress_bar(range(len(all_image_files)))
        pbar.comment = "loading files"
        for idx in pbar:            
            img_file = all_image_files[idx]              
            heatmap_files = []
            for feat in features.keys():
                heatmap_files.append(self.__root_path/feat/img_file.name)                                        
            
            if img_file in valid_images_files:                
                valid_data.append((img_file,heatmap_files))
            else:
                training_data.append((img_file,heatmap_files))        

        return training_data, valid_data  

    def save_losses(self, filename=None, train=True,valid=True):
        assert ((self.__train_losses is not None) and (self.__valid_losses is not None)), "Call `fit` Before losses can be saved"                         
        
        filename = "losses" if filename is None else filename
                
        if train:
            np.save(self.__root_path/self.__sample_results_path/(filename+"_train.npy"), self.__train_losses)
        if valid:
            np.save(self.__root_path/self.__sample_results_path/(filename+"_valid.npy"), self.__valid_losses)
            
    def get_metric_names(self):        
        return np.array(self.learner.metric_names)
    
    def save_metrics(self,filename=None):
        assert ((self.__metrics is not None)),"Call `fit` Before metrics can be saved"            
            
        filename = "metrics" if filename is None else filename
        
        data = {"names":self.get_metric_names(),"metrics":self.__metrics}
        pickle.dump(data, open(self.__root_path/self.__sample_results_path/(filename+".pkl"),"wb"))

    
    def get_metrics(self, specific_metric_names=None):        
        
        assert ((self.__metrics is not None)), "Call `fit` Before metrics can be retrived"
        
        if specific_metric_names is None:
            return self.__metrics
        
        specific_metric_names = np.array(specific_metric_names)
        metric_idxs = []
        for metric in specific_metric_names:
            condition = np.where(self.get_metric_names() == metric)[0]
            if len(condition) != 1:
                continue
            metric_idxs.append(condition[0])
            
        metric_idxs = np.array(metric_idxs)
        
        if len(metric_idxs) == 0:
            raise("There are no matching `specific_metric_names`. Check orthography.")
                
        return np.array(self.__metrics)[:,metric_idxs]                            
        
    
    def get_losses(self, train=True, valid=True):
        assert (train or valid), "train or valid must be True"
        assert ((self.__train_losses is not None) and (self.__valid_losses is not None)), "Call `fit` Before losses can be retrived"
        
        result = []
        if train:
            result.append(self.__train_losses)
        
        if valid:
            result.append(self.__valid_losses)
            
        return result[0] if len(result) == 1 else result
                
    def fit(self,epochs):                        
                
        self.learner.fit(epochs)
        self.__epochs += epochs
        
        train_loss, valid_loss = self.learner.get_losses()
        
        if self.__train_losses is None:
            self.__train_losses = train_loss
        else:
            self.__train_losses = np.concatenate((self.__train_losses, train_loss))

        if self.__valid_losses is None:
            self.__valid_losses = valid_loss
        else:
            self.__valid_losses = np.concatenate((self.__valid_losses, valid_loss))                        
            
        metrics = self.learner.get_metrics()
        self.__metrics = np.concatenate((self.__metrics, metrics), axis=0) if self.__metrics is not None else metrics        
        
    def load_from_file(self,filename=None):
        
        filename = "model" if filename is None else filename 
        
        checkpoint = torch.load(self.__root_path/"models"/(filename+".pth"))
        self.learner.model.load_state_dict(checkpoint['model'])
        self.learner.optimizer.load_state_dict(checkpoint['optimizer'])                                
        self.__epochs = checkpoint["epochs"]
        self.__train_losses = checkpoint["train_losses"]
        self.__valid_losses = checkpoint["valid_losses"]
        self.__metrics = checkpoint["metrics"]
        
    
    def save_to_file(self,filename=None):
        (self.__root_path/"models").mkdir(parents=True, exist_ok=True)
        
        filename = "model" if filename is None else filename
            
        self.learner.model
        
        torch.save({'epochs': self.__epochs,
                    'train_losses':self.__train_losses,
                    'valid_losses':self.__valid_losses,
                    'metrics':self.__metrics,
                    'model': self.learner.model.state_dict(),
                    'optimizer': self.learner.optimizer.state_dict()}, self.__root_path/"models"/(filename+".pth"))        
                
    def export_to_onnx(self, filename=None, batch_size=1):
        
        input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
        output_names = [ "output1" ]
        
        onnx_export = (self.__root_path/"onnx_export")
        onnx_export.mkdir(parents=True, exist_ok=True)
        
        batch = torch.zeros((batch_size,self.__unet_in_channels,self.__size[0],self.__size[1]), 
                           requires_grad=True, device="cuda:"+str(self.__gpu_id))
        
        for i in range(batch_size):
            batch[i] = self.valid_dataset[i][0]
                    
        filename = "export" if filename is None else filename 
                
        torch_out = self.learner.model(batch)
        torch.onnx.export(self.learner.model, batch,onnx_export/(filename+".onnx"), 
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
        raise("TODO Implement")
                
    def get_image_fname(self,fname):
        raise("TODO Implement")
    
    def get_label(self, idx, train=True):
        raise("TOO Implement")
        
    def get_image(self, idx, train=True):
        raise("TOO Implement")

    def get_ds_len(self, train=True):
        raise("TOO Implement")                            

    def show_data(self, count_img, train=True):    

        def show_feat_func(ax, image):
            if image.shape[0] == 1:
                ax.imshow(image[0,:,:], cmap="gray")
            elif image.shape[0] == 3:
                ax.imshow(image.permute(1,2,0))        

        def show_label_func(ax, heatmaps):   
            ax.set_title("features on image")           
            for heatmap,feature  in zip(heatmaps, list(self.features.keys())):
                type_of = self.features[feature]["type"]
                if (type_of == "circle"):            
                    points = heatmap_to_circle(np.array(heatmap))                                
                    if points is not None:
                        ax.plot(points[:,0], points[:,1], "r.")
                elif type_of == "single_point":            
                    center_point = heatmap_to_max_confidence_point(heatmap)
                    if center_point is not None:
                        ax.plot(center_point[0], center_point[1], 'bo') 
                elif type_of == "multi_point":
                    all_peaks = heatmap_to_multiple_points(np.array(heatmap))
                    if all_peaks is not None:
                        for peak in all_peaks[:,0:2]:
                            ax.plot(peak[0], peak[1], 'go')    
                else:
                    raise("The Heatmaptype " + type_of + " is not implemented yet.")            
        
        dataset = self.train_dataset if train else self.valid_dataset
        train_dataloader = DataLoader(dataset, batch_size=count_img, shuffle=True, num_workers=num_workers())
        data_iter = iter(train_dataloader)
        images, labels = next(data_iter)    
        nrows = ceil(count_img/3)
        f,axes = plt.subplots(nrows=nrows,ncols=3,figsize=(15,5*nrows))
        axes = axes.flatten()[:count_img]
        for image,label,ax in zip(images, labels, axes.flatten()):
            show_feat_func(ax, image)
            show_label_func(ax, label)                
                
    def predict(self,image): 
        raise("TODO Implement !")
        
    def predict_sample(self):

        sample_img = self.__sample_img
        if (sample_img is not None) and (self.__sample_results_path is not None):

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


            self.__sample_results_path.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.__sample_results_path/("epoch_"+str(self.__counter_epoch)+".png")), img)