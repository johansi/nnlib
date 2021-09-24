
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
import warnings

__all__ = ["DataAugmentation", "HeatmapLearner", "HeatLoss_OldGen_0", "HeatLoss_OldGen_1", "HeatLoss_OldGen_2", "HeatLoss_OldGen_3", "HeatLoss_OldGen_4", "HeatLoss_NextGen_0", "HeatLoss_NextGen_1",
          "HeatLoss_NextGen_2", "HeatLoss_NextGen_3"]

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

class CustomHeatmapDataset(Dataset):
    "CustomImageDataset with `image_files`,`y_func`, `convert_mode` and `size(height, width)`"
    def __init__(self, data, hull_path, size=(512,512), grayscale=False, normalize_mean=None, normalize_std=None, data_aug=None, 
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
        self.un_normalize = UnNormalize(mean=normalize_mean, std=normalize_std)
        self.hull_path = hull_path
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
            mask_file = fname.parent/(fname.stem + "_mask"+ fname.suffix)
            heatmaps.append([load_heatmap(fname, size=self.size),load_heatmap(mask_file, size=self.size)])        
        return heatmaps
    
    def load_hull(self, idx):
        return load_heatmap(self.hull_path/self.data[idx][0].name, size=self.size)        
    
    def apply_clahe(self,img):
        if self.clahe == None:
            return img
        img = np.asarray(img)
        img = self.clahe.apply(img)
        return Image.fromarray(img)                
    
    def __getitem__(self, idx):
        
        
        
        image = load_image(self.data[idx][0], size=self.size, convert_mode=self.convert_mode, to_numpy=False)        
                
        labels = self.load_labels(idx)
        
        hull = self.load_hull(idx)
        
        image = self.apply_clahe(image)
                
        if (not self.is_valid) and (self.data_aug is not None):
            image, labels, hull = self.data_aug.transform(image, labels, hull)        
                        
        hull = torch.squeeze(self.to_t(hull),dim=0).type(torch.bool)        
                        
        labels_extraced = [label[0] for label in labels]
        masks_extraced  = [label[1] for label in labels]
        
        labels_extraced = self.to_t(np.stack(labels_extraced, axis=2))
        masks_extraced = self.to_t(np.stack(masks_extraced, axis=2)).type(torch.bool)
        
        image = self.to_t(image)
        
        if self.do_normalize:
            image = self.normalize(image)
                        
        return self.data[idx][0].stem, image, labels_extraced, masks_extraced, hull    
        
        
class RandomRotationImageTarget(transforms.RandomRotation):
    def __call__(self, img, targets, hull):    
        
        
        angle = self.get_params(self.degrees)
        img = transforms.functional.rotate(img, angle, self.resample, self.expand, self.center)
        hull = transforms.functional.rotate(hull, angle, self.resample, self.expand, self.center)
        for idx in range(len(targets)):
            targets[idx][0] = transforms.functional.rotate(targets[idx][0], angle, self.resample, self.expand, self.center)
            targets[idx][1] = transforms.functional.rotate(targets[idx][1], angle, self.resample, self.expand, self.center)
        
        return img, targets, hull
    
class RandomHorizontalFlipImageTarget(transforms.RandomHorizontalFlip):
    def __call__(self, img, targets, hull):   
        
        
        if random.random() < self.p:
            img = transforms.functional.hflip(img)
            hull = transforms.functional.hflip(hull)
            for idx in range(len(targets)):
                targets[idx][0] = transforms.functional.hflip(targets[idx][0])
                targets[idx][1] = transforms.functional.hflip(targets[idx][1])
        
        return img,targets,hull

class RandomVerticalFlipImageTarget(transforms.RandomVerticalFlip):
    def __call__(self, img, targets, hull):    
        
        
        if random.random() < self.p:
            img = transforms.functional.vflip(img)
            hull = transforms.functional.vflip(hull)
            for idx in range(len(targets)):
                targets[idx][0] = transforms.functional.vflip(targets[idx][0])
                targets[idx][1] = transforms.functional.vflip(targets[idx][1])
        
        return img,targets,hull

class RandomPerspectiveImageTarget(transforms.RandomPerspective):
    def __call__(self, img, targets, hull): 
        
        
        if not transforms.functional._is_pil_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
        
        if random.random() < self.p:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="torch.lstsq")            
                width, height = img.size
                startpoints, endpoints = self.get_params(width, height, self.distortion_scale)
                img = transforms.functional.perspective(img, startpoints, endpoints, self.interpolation, self.fill)
                hull = transforms.functional.perspective(hull, startpoints, endpoints, self.interpolation, self.fill)

                for idx in range(len(targets)):
                    targets[idx][0] = transforms.functional.perspective(targets[idx][0], startpoints, endpoints, self.interpolation, self.fill)
                    targets[idx][1] = transforms.functional.perspective(targets[idx][1], startpoints, endpoints, self.interpolation, self.fill)

        return img,targets,hull

class ComposeImageTarget(transforms.Compose):
    def __call__(self, img, targets, hull): 
            for t in self.transforms:
                img,targets,hull = t(img, targets, hull)
            return img,targets,hull 
        
        
class DataAugmentation:
    "DataAugmentation class with `size(height,width)`"
    def __init__(self, rotation=10,horizontal_flip_p=0.5,
                 vertical_flip_p=0.5,warp=0.1,warp_p=0.5, zoom=0.8, 
                 brightness=0.5, contrast=0.5, GaussianBlur=1):
                 
        
        self.lightning_transforms = transforms.Compose([transforms.ColorJitter(brightness=brightness,contrast=contrast),
                                                        #transforms.GaussianBlur(kernel_size=GaussianBlur)
                                                       ])

        self.affine_transforms = ComposeImageTarget([                                                     
                                                     RandomRotationImageTarget(degrees=(-rotation,rotation)),
                                                     RandomHorizontalFlipImageTarget(p=horizontal_flip_p),
                                                     RandomVerticalFlipImageTarget(p=vertical_flip_p),
                                                     RandomPerspectiveImageTarget(distortion_scale=warp, p=warp_p),
                                                     #transforms.RandomResizedCrop(size=size,scale=(zoom,1.0),ratio=(1.0,1.0))
                                                    ])        
        
    def transform(self,features,labels, hull):
                
        
        #do lighting transforms for features
        features = self.lightning_transforms(features)
        
        
        
        #do affine transforms for features and labels        
        features,labels,hull = self.affine_transforms(features, labels, hull)            
        
        
        
        return features,labels,hull     
    
        
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

class HeatLoss_OldGen_0(nn.Module):
    def __init__(self):
        super().__init__()
        r"""Class for HeatLoss calculation. This variant includes no masks, simple Mean absolute error over all pixles:                        
        """           
    def forward(self, input, target, masks, hull):                
        return torch.mean(torch.abs(input - target))

class HeatLoss_OldGen_1(nn.Module):
    def __init__(self, print_out_losses=False):
        super().__init__()
        r"""Class for HeatLoss calculation. This variant includes the masks of following objects:            
            - specific features
        """       
        self.print_out_losses = print_out_losses

    def forward(self, input, target, masks, hull):
        m1 = (target > 0.0)
        ret1 = torch.abs(input[m1] - target[m1])
        mean1 = torch.mean(ret1)
        
        if self.print_out_losses:
            print("specific features:",mean1.item(), end="\r")

        return mean1    
        
class HeatLoss_OldGen_2(nn.Module):
    def __init__(self, print_out_losses=False):
        
        r"""Class for HeatLoss calculation. This variant includes the masks of following objects:
            - Background (no heat at all)
            - specific features
        """             
        super().__init__()
        self.print_out_losses = print_out_losses

    def forward(self, input, target, masks, hull):
        m1 = (target > 0.0)        
        m2 = torch.logical_not(m1)

        ret1 = torch.abs(input[m1] - target[m1])
        ret2 = torch.abs(input[m2] - target[m2])        

        mean1 = torch.mean(ret1)
        mean2 = torch.mean(ret2)
        
        if self.print_out_losses:
            print("specific features:",mean1.item(), "background:",mean2.item(), end="\r")

        return (mean1+mean2)/2

class HeatLoss_OldGen_3(nn.Module):
    def __init__(self, print_out_losses=False):
        super().__init__()
        r"""Class for HeatLoss calculation. This variant includes the masks of following objects:            
            - specific feature
            - all features in a image
        """     
        self.print_out_losses = print_out_losses

    def forward(self, input, target, masks, hull):
        m1 = (target > 0.0)
        m2 = torch.zeros(m1.shape, dtype=torch.bool, device=input.device)        

        for dset in range(input.shape[0]):
            logor = torch.zeros((input.shape[2], input.shape[3]), dtype=torch.bool, device=input.device)
            
            for i in range(input.shape[1]):
                logor = logor | m1[dset,i,:,:]
            
            for i in range(input.shape[1]):
                m2[dset,i,:,:] = logor                

        ret1 = torch.abs(input[m1] - target[m1])
        ret2 = torch.abs(input[m2] - target[m2])        

        mean1 = torch.mean(ret1)
        mean2 = torch.mean(ret2)
        if  self.print_out_losses:
            print("specific feature:",mean1.item(), "all features:",mean2.item(), end="\r")

        return (mean1+mean2)/2

class HeatLoss_OldGen_4(nn.Module):
    def __init__(self, print_out_losses=False):
        super().__init__()
        r"""Class for HeatLoss calculation. This variant includes the masks of following objects:            
            - specific feature
            - all features in a image
            - background
        """   
        self.print_out_losses = print_out_losses

    def forward(self, input, target, masks, hull):
        m1 = (target > 0.0)
        m2 = torch.zeros(m1.shape, dtype=torch.bool, device=input.device)
        m3 = torch.logical_not(m1)

        for dset in range(input.shape[0]):
            logor = torch.zeros((input.shape[2], input.shape[3]), dtype=torch.bool, device=input.device)
            
            for i in range(input.shape[1]):
                logor = logor | m1[dset,i,:,:]
            
            for i in range(input.shape[1]):
                m2[dset,i,:,:] = logor                

        ret1 = torch.abs(input[m1] - target[m1])
        ret2 = torch.abs(input[m2] - target[m2])
        ret3 = torch.abs(input[m3] - target[m3])

        mean1 = torch.mean(ret1)
        mean2 = torch.mean(ret2)
        mean3 = torch.mean(ret3)
        if self.print_out_losses:
            print("specific feature:",mean1.item(), "all features:",mean2.item(), "background:",mean3.item(), end="\r")

        return (mean1+mean2+mean3)/3  

class HeatLoss_NextGen_0(nn.Module):
    def __init__(self, print_out_losses=False):
        super().__init__()
        r"""Class for Next Generation HeatLoss calculation. This variant includes offline generated masks of following objects:            
            - specific feature with mask dilation (single loss calculation for every feature)
            - convex hull of all featureswith maks dilation
            - background
        """   
        self.print_out_losses = print_out_losses
        
    def forward(self, input, target, masks, hull):                
        
        hull_not = torch.logical_not(hull)
        feature_count = target.shape[1]
        loss_items = torch.zeros(feature_count, dtype=torch.float32, device=input.device)
        for idx in range(feature_count):
            diff = torch.abs(input[:,idx,:,:][masks[:,idx,:,:]] - target[:,idx,:,:][masks[:,idx,:,:]])
            if len(diff) > 0:
                loss_items[idx] = torch.mean(diff)
            
        loss_hull = torch.mean(torch.abs(input[hull] - target[hull]))        
        loss_backgrond = torch.mean(torch.abs(input[hull_not] - target[hull_not]))
        
        if self.print_out_losses:
            # print loss begin
            out_str = ""
            print_items_loss = []
            sum_items_loss = torch.zeros(1, dtype=torch.float32, device=input.device)
            for idx in range(len(loss_items)):
                out_str = out_str + "loss_item_"+str(idx) + " {:.4f} "
                print_items_loss.append(round(loss_items[idx].item(),4))
                sum_items_loss += loss_items[idx]

            print_items_loss.append(round(loss_hull.item(),4))
            print_items_loss.append(round(loss_backgrond.item(),4))
            print((out_str+" loss_hull {:.4f} loss_backgrond {:.4f}").format(*print_items_loss), end="\r")
            # print loss end        
        
        return (sum_items_loss+loss_hull+loss_backgrond)/(feature_count+2)

    
class HeatLoss_NextGen_1(nn.Module):
    def __init__(self):
        super().__init__(print_out_losses=False)
        r"""Class for Next Generation HeatLoss calculation. This variant includes offline generated masks of following objects:            
            - specific feature with mask dilation (calculation of feature loss all the same)
            - convex hull of all featureswith maks dilation
            - background
        """   
        self.print_out_losses = print_out_losses
        
    def forward(self, input, target, masks, hull):                
        
        hull_not = torch.logical_not(hull)
        
        loss_features = torch.mean(torch.abs(input[masks] - target[masks]))
        loss_hull = torch.mean(torch.abs(input[hull] - target[hull]))        
        loss_backgrond = torch.mean(torch.abs(input[hull_not] - target[hull_not]))        
        
        if self.print_out_losses:
            print(("loss_features {:.4f} loss_hull {:.4f} loss_backgrond {:.4f}").format(loss_features,loss_hull,loss_backgrond), end="\r")               
        
        return (loss_features+loss_hull+loss_backgrond)/3      
    
class HeatLoss_NextGen_2(nn.Module):
    def __init__(self, print_out_losses=False):
        super().__init__()
        r"""Class for Next Generation HeatLoss calculation. This variant includes offline generated masks of following objects:            
            - specific feature with mask dilation (calculation of feature loss all the same)
            - all features in a image  (calculation of feature loss all the same)    
            - background (calculation of feature loss all the same)
        """   
        self.print_out_losses = print_out_losses
        
    def forward(self, input, target, masks, hull):                
                
        all_mask = torch.any(masks,dim=1)[:,None].repeat(1,target.shape[1],1,1)        
        mask_not = torch.logical_not(masks)        
        
        loss_features = torch.mean(torch.abs(input[masks] - target[masks]))
        loss_all_features = torch.mean(torch.abs(input[all_mask] - target[all_mask]))        
        loss_backgrond = torch.mean(torch.abs(input[mask_not] - target[mask_not]))        
        
        if self.print_out_losses:
            print(("loss_features {:.4f} loss_all_features {:.4f} loss_backgrond {:.4f}").format(loss_features.item(),loss_all_features.item(),loss_backgrond.item()), end="\r")               
        
        return (loss_features+loss_all_features+loss_backgrond)/3

    
class HeatLoss_NextGen_3(nn.Module):
    def __init__(self, print_out_losses=False):
        super().__init__()
        r"""Class for Next Generation HeatLoss calculation. This variant includes offline generated masks of following objects:            
            - specific feature with mask dilation (single loss calculation for every feature)
            - all features in a image   (single loss calculation for every feature) 
            - background (single loss calculation for every feature)
        """   
        self.print_out_losses = print_out_losses
        
    def forward(self, input, target, masks, hull):                
                
        feature_count = target.shape[1]
        mask_not = torch.logical_not(masks)
        all_mask = torch.any(masks,dim=1)[:,None].repeat(1,target.shape[1],1,1)
        
        loss_features = torch.zeros(feature_count, dtype=torch.float32, device=input.device)
        loss_backgrond = torch.zeros(feature_count, dtype=torch.float32, device=input.device)
        loss_all_features = torch.zeros(feature_count, dtype=torch.float32, device=input.device)
        
        for idx in range(feature_count):
            diff = torch.abs(input[:,idx,:,:][masks[:,idx,:,:]] - target[:,idx,:,:][masks[:,idx,:,:]])
            diff_not = torch.abs(input[:,idx,:,:][mask_not[:,idx,:,:]] - target[:,idx,:,:][mask_not[:,idx,:,:]])
            diff_all = torch.abs(input[:,idx,:,:][all_mask[:,idx,:,:]] - target[:,idx,:,:][all_mask[:,idx,:,:]])
            if len(diff) > 0:
                loss_features[idx] = torch.mean(diff)
            if len(diff_not) > 0:
                loss_backgrond[idx] = torch.mean(diff_not)
            if len(diff_all) > 0:
                loss_all_features[idx] = torch.mean(diff_all)
            
        loss_features = torch.mean(loss_features)
        loss_backgrond = torch.mean(loss_backgrond)
        loss_all_features = torch.mean(loss_all_features)
        
        if self.print_out_losses:
            print(("loss_features {:.4f} loss_all_features {:.4f} loss_backgrond {:.4f}").format(loss_features.item(),loss_all_features.item(),loss_backgrond.item()), end="\r")                          
        
        return (loss_features+loss_all_features+loss_backgrond)/3    
    
class HeatmapLearner:
    def __init__(self, features, root_path, images_path, hull_path, size=(512,512), bs=-1, items_count=-1, gpu_id=0, 
                 norm_stats=None, data_aug=None, preload=False, sample_results_path="sample_results",
                 unet_init_features=16, valid_images_store="valid_images.npy", image_convert_mode="L", metric_counter=1, 
                 sample_img=None, true_positive_threshold=0.02, ntype="unet", lr=1e-03, file_filters_include=None, file_filters_exclude=None, clahe=False,
                 disable_metrics=False, file_prefix="", loss_func=None, weight_decay=0, num_load_workers=None):
        
        r"""Class for train an Unet-style Neural Network, for heatmap based image recognition

        Args:
            features     :  Heatmap features for the neural net. This must be a dict. The Keys must be the folder names for the heatmap features.
                            Every entry is a dict with the feature types: single_point, multi_point, circle
                            Example:
                            {"feature_1":{"type":"single_point"},
                             "feature_2":{"type":"multi_point"},
                             "feature_3":{"type":"circle"}}  
            root_path    :  The root path, where image files and label files are located
            images_path  :  The path, where the images are located, in relation to the root_path            
            size         :  Size of images to pass through the neural network. The sizes must be a power of two with the dimensions of (Heigth, Width).
            bs           :  The Batch Size
            norm_stats   :  Normalize values for images in the form (mean,std).
            file_filters_incluce :  incluce file filter in images_path, must be a list with include search strings
            file_filters_exclude :  exclude file filter in images_path, must be a list with exclude search strings
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
        self.__num_load_workers = num_load_workers
        self.__root_path = Path(root_path)
        self.__images_path = Path(images_path)
        self.__hull_path = self.__root_path/Path(hull_path)
        self.__sample_results_path = Path(sample_results_path)  
        (self.__root_path/self.__sample_results_path).mkdir(parents=True, exist_ok=True)
        
        self.__gpu_id = gpu_id
        self.__file_prefix = file_prefix
        data_aug = DataAugmentation() if data_aug is None else data_aug
        
        if norm_stats is None:
            norm_stats = ([0.131],[0.308]) if image_convert_mode == "L" else ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                        
        if sample_img is None:
            t_img_files = glob.glob(str(self.__root_path/self.__images_path/"*"))
            idx = random.randint(0, len(t_img_files)-1)
            self.__sample_img = self.__root_path/self.__images_path/Path(t_img_files[idx]).name
        else:
            self.__sample_img = self.__root_path/self.__images_path/sample_img
        
        true_positive_threshold = round(true_positive_threshold*self.__size[0])        
        
        # dynamic variables (they can change during learning)        
        self.__epochs = 0
        self.__train_losses = None
        self.__valid_losses = None
        self.__metrics = None        
        
        file_filters_include = np.array(file_filters_include) if file_filters_include is not None else None
        file_filters_exclude = np.array(file_filters_exclude) if file_filters_exclude is not None else None
        
        
        self.__create_learner(file_filters_include=file_filters_include, file_filters_exclude=file_filters_exclude, valid_images_store=valid_images_store, items_count=items_count,
                              features=features, bs=bs, data_aug=data_aug,image_convert_mode=image_convert_mode, 
                              heatmap_paths=heatmap_paths, true_positive_threshold=true_positive_threshold, metric_counter=metric_counter,
                              lr=lr, clahe=clahe, norm_stats=norm_stats, unet_init_features=unet_init_features, ntype=ntype, disable_metrics=disable_metrics, loss_func=loss_func,
                              weight_decay = weight_decay)
        
    def __create_learner(self, file_filters_include, file_filters_exclude, valid_images_store, items_count, features, bs, data_aug, 
                         image_convert_mode, heatmap_paths, true_positive_threshold, metric_counter, lr, clahe, norm_stats, 
                         unet_init_features, ntype, disable_metrics, loss_func, weight_decay):
                                
        training_data, valid_data = self.__load_data(features=features,file_filters_include=file_filters_include,
                                                     file_filters_exclude = file_filters_exclude, valid_images_store=valid_images_store, 
                                                     items_count=items_count)                       
                                      
        self.__unet_in_channels = 1 if image_convert_mode == "L" else 3
        self.__unet_out_channls = len(heatmap_paths)                                                    
                
        heatmap_files_sample = []
        for feat in features.keys():
                heatmap_files_sample.append(self.__root_path/feat/self.__sample_img.name)                                        
        
        self.sample_dataset = CustomHeatmapDataset(data=[[self.__sample_img,heatmap_files_sample]], hull_path=self.__hull_path, grayscale=image_convert_mode == "L", 
                                                  normalize_mean=norm_stats[0],normalize_std=norm_stats[1], is_valid=True, 
                                                  clahe=clahe, size=self.__size) 
        
        self.train_dataset = CustomHeatmapDataset(data=training_data, hull_path=self.__hull_path, grayscale=image_convert_mode == "L", 
                                                  normalize_mean=norm_stats[0],normalize_std=norm_stats[1], data_aug=data_aug, clahe=clahe,
                                                  size=self.__size)

        self.valid_dataset = CustomHeatmapDataset(data=valid_data, hull_path=self.__hull_path, grayscale=image_convert_mode == "L",
                                                  normalize_mean=norm_stats[0],normalize_std=norm_stats[1], is_valid=True, clahe=clahe,
                                                  size=self.__size)                                
        
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

        metric = None if disable_metrics else heatmap_metric(features = features, true_positive_threshold = true_positive_threshold, metric_counter = metric_counter)
        net = self.__get_net(ntype, unet_init_features).to(torch.device("cuda:"+str(self.__gpu_id)))
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)        
        loss_func = HeatLoss_NextGen_1() if loss_func is None else loss_func       
        loss_func = loss_func.to(torch.device("cuda:"+str(self.__gpu_id)))                
        if bs == -1:      
            
            batch_estimator = Batch_Size_Estimator(net=net, opt=opt,
                                                   loss_func=loss_func, 
                                                   gpu_id=self.__gpu_id, dataset = self.train_dataset)
            bs = batch_estimator.find_max_bs()
                                         
        train_dl = DataLoader(self.train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers() if self.__num_load_workers is None else self.__num_load_workers, pin_memory=False)
        valid_dl = DataLoader(self.valid_dataset, batch_size=bs, shuffle=True, num_workers=num_workers() if self.__num_load_workers is None else self.__num_load_workers, pin_memory=False)
        
        self.learner = Learner(model=net,loss_func=loss_func, train_dl=train_dl, valid_dl=valid_dl,
                               optimizer=opt, learner_callback= metric,gpu_id= self.__gpu_id,
                               predict_smaple_func=self.predict_sample)                         
           
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
        elif ntype == "res34":
                    net = UNet(in_channels = self.__unet_in_channels, out_channels = self.__unet_out_channls,
                               init_features = unet_init_features, resblock=True, squeeze_excite=False, 
                               aspp=False, attention=False, bn_relu_at_first=True, bn_relu_at_end=False,
                               block_sizes_down = res34_downsample, downsample_method=downsample_stride,
                               blocksize_bottleneck = 2, block_sizes_up=[2,2,2,2])            
        else:
            raise("Net type ´"+ ntype+"´ is not implemented!")     
            
        return net
    
    def __load_data(self, features, file_filters_include, file_filters_exclude, valid_images_store, items_count):
        
        def find_valid_imgs_func(all_image_files):
            if not (self.__root_path/valid_images_store).is_file():
                valid_images = [img for img in sample(list(all_image_files), int(len(all_image_files)*0.2))]
                np.save(self.__root_path/valid_images_store, valid_images, allow_pickle=True)
            return list(np.load(self.__root_path/valid_images_store, allow_pickle=True))            

        def filter_files(all_image_files):
            
            if file_filters_include is not None:
                new_filenames = []
                for filename in all_image_files:
                    for ffilter in file_filters_include:
                        if filename.find((ffilter)) != -1:
                            new_filenames.append(filename)
                            break
                all_image_files = np.array(new_filenames) 
            
            if file_filters_exclude is not None:
                new_filenames = []
                for filename in all_image_files:
                    include_in = True
                    
                    for ffilter in file_filters_exclude:
                        if filename.find((ffilter)) != -1:
                            include_in = False                            
                            break
                    
                    if include_in:
                        new_filenames.append(filename)
                
                all_image_files = np.array(new_filenames) 
                
            return all_image_files
        
        all_image_files = [fname.name for fname in list((self.__root_path/self.__images_path).glob("*.png"))]
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
                heatmap_files.append(self.__root_path/feat/img_file)                                                                
            
            add_file = self.__root_path/self.__images_path/img_file
            if img_file in valid_images_files:                
                valid_data.append((add_file,heatmap_files))
            else:
                training_data.append((add_file,heatmap_files))        

        return training_data, valid_data  

    def save_losses(self, filename=None, train=True,valid=True):
        assert ((self.__train_losses is not None) and (self.__valid_losses is not None)), "Call `fit` Before losses can be saved"                         
        
        filename = "losses" if filename is None else filename
                
        if train:
            np.save(self.__root_path/self.__sample_results_path/(self.__file_prefix+filename+"_train.npy"), self.__train_losses)
        if valid:
            np.save(self.__root_path/self.__sample_results_path/(self.__file_prefix+filename+"_valid.npy"), self.__valid_losses)
            
    def get_metric_names(self):        
        return np.array(self.learner.metric_names)
    
    def save_metrics(self,filename=None):
        assert ((self.__metrics is not None)),"Call `fit` Before metrics can be saved"            
            
        filename = "metrics" if filename is None else filename
        
        data = {"names":self.get_metric_names(),"metrics":self.__metrics}
        pickle.dump(data, open(self.__root_path/self.__sample_results_path/(self.__file_prefix+filename+".pkl"),"wb"))

    
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
                
    def fit(self,epochs,one_cylce=True):
        self.learner.fit(epochs, one_cylce)
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
    
    def set_loss_func(self,loss_func):        
        self.learner.set_loss_func(loss_func().to(torch.device("cuda:"+str(self.__gpu_id))))        
        
    def load_from_file(self,filename=None):
        
        filename = "model" if filename is None else filename 
        
        checkpoint = torch.load(self.__root_path/"models"/(self.__file_prefix+filename+".pth"))
        self.learner.model.load_state_dict(checkpoint['model'])
        self.learner.optimizer.load_state_dict(checkpoint['optimizer'])                                
        self.__epochs = checkpoint["epochs"]
        self.__train_losses = checkpoint["train_losses"]
        self.__valid_losses = checkpoint["valid_losses"]
        self.__metrics = checkpoint["metrics"]
        
    
    def save_to_file(self,filename=None):
        (self.__root_path/"models").mkdir(parents=True, exist_ok=True)
        
        filename = "model" if filename is None else filename                    
        
        torch.save({'epochs': self.__epochs,
                    'train_losses':self.__train_losses,
                    'valid_losses':self.__valid_losses,
                    'metrics':self.__metrics,
                    'model': self.learner.model.state_dict(),
                    'optimizer': self.learner.optimizer.state_dict()}, self.__root_path/"models"/(self.__file_prefix+filename+".pth"))        
                
    def export_to_onnx(self, filename=None, batch_size=1):
        
        input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
        output_names = [ "output1" ]
        #pdb.set_trace()
        onnx_export = (self.__root_path/"onnx_export")
        onnx_export.mkdir(parents=True, exist_ok=True)
        
        batch = torch.zeros((batch_size,self.__unet_in_channels,self.__size[0],self.__size[1]), 
                           requires_grad=True, device="cuda:"+str(self.__gpu_id))
        
        for i in range(batch_size):
            batch[i] = self.valid_dataset[i][1]
                    
        filename = "export" if filename is None else filename 
                
        torch_out = self.learner.model(batch)
        torch.onnx.export(self.learner.model, batch,onnx_export/(self.__file_prefix+filename+".onnx"), 
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

        def show_label_func(ax, heatmaps, name):   
            ax.set_title(name)           
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
        
        train_dataloader = DataLoader(dataset, batch_size=count_img, shuffle=True, num_workers=num_workers() if self.__num_load_workers is None else self.__num_load_workers)
        data_iter = iter(train_dataloader)
        names, images, labels, masks, hulls = next(data_iter)
        for idx in range(len(images)):
            images[idx] = dataset.un_normalize(images[idx])
        nrows = ceil(count_img/3)
        f,axes = plt.subplots(nrows=nrows,ncols=3,figsize=(15,5*nrows))
        axes = axes.flatten()[:count_img]
        for name, image,label,ax in zip(names, images, labels, axes.flatten()):
            show_feat_func(ax, image)
            show_label_func(ax, label, name)                
                
    def predict(self,inp, eval_mode=True):        
        return self.learner.predict(inp[None].to(torch.device("cuda:"+str(self.__gpu_id))), eval_mode=eval_mode).detach().cpu()
                        
    def predict_sample(self, epoch):                  
        name, image, label, mask, hull  = self.sample_dataset[0]
        image_un_norm = self.sample_dataset.un_normalize(image)
        channels = image.data.shape[0]
        shape = (image.data.shape[1], image.data.shape[2])
        img = np.zeros((shape[0]*2,shape[1]+(shape[1]*len(label)),channels), dtype=np.float)        
        img[:shape[0],:shape[1],:] = image_un_norm.data.permute(1,2,0)
        for idx in range(len(label)):
            img[:shape[0],shape[1]*(idx+1):shape[1]*(idx+2),:] = label[idx].repeat(channels,1,1).permute(1,2,0)

        res = torch.squeeze(self.predict(image, eval_mode=False), dim=0)        
        for idx in range(res.shape[0]):
            img[shape[0]:shape[0]*2,shape[1]*(idx+1):shape[1]*(idx+2),:] = res[idx][None].repeat(channels,1,1).permute(1,2,0)

        img = np.round(np.interp(img, (img.min(), img.max()), (0, 255))).astype(np.uint8)            
        if channels == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        self.__sample_results_path.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.__root_path/self.__sample_results_path/(self.__file_prefix+"epoch_"+str(self.__epochs+epoch)+".png")), img)                    