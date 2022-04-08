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
from .resnet import *
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

__all__ = ["DataAugmentation_CNN", "CNNLearner", "RegressionLoss", "accuracy_metric"]

class UnNormalize_CNN(object):
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

class CustomHeatmapDataset_CNN(Dataset):
    "CustomImageDataset with `image_files`,`y_func`, `convert_mode` and `size(height, width)`"
    def __init__(self, data, size=(512,512), classification=True, grayscale=False, normalize_mean=None, normalize_std=None, data_aug=None, 
                 is_valid=False, do_normalize=True, clahe=True):
        
        
        self.data = data        
        self.classification = classification
        self.size = size   
        
        if self.classification:
            self.unique_labels = {}
            for idx,label in enumerate(np.unique(np.array(data)[:,1])):
                self.unique_labels[label] = idx 
        
        
        
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
        self.un_normalize = UnNormalize_CNN(mean=normalize_mean, std=normalize_std)        
        self.convert_mode = "L" if grayscale else "RGB"        
        self.data_aug = data_aug
        self.to_t = transforms.ToTensor()
        self.is_valid = is_valid
        self.do_normalize = do_normalize        
        
        self.clahe = cv2.createCLAHE(clipLimit=20.0,tileGridSize=(30,30)) if clahe else None
    
    def __len__(self):
        return len(self.data)

    def load_labels(self,idx):
        data = self.data[idx][1]
        if self.classification:
            data = self.unique_labels[data]        
        return data
    
    def apply_clahe(self,img):
        if self.clahe == None:
            return img
        img = np.asarray(img)
        img = self.clahe.apply(img)
        return Image.fromarray(img)                
    
    def classes(self):
        return self.unique_labels
        
    def __getitem__(self, idx):          
        
        image = load_image(self.data[idx][0], size=self.size, convert_mode=self.convert_mode, to_numpy=False)        
                
        labels = self.load_labels(idx)
        
        image = self.apply_clahe(image)
                
        if (not self.is_valid) and (self.data_aug is not None):
            image = self.data_aug.transform(image)        
                                
        image = self.to_t(image)
        
        if self.do_normalize:
            image = self.normalize(image)
                        
        return self.data[idx][0].stem, image, labels
        
        
class RandomRotationImageTarget_CNN(transforms.RandomRotation):
    def __call__(self, img):    
        angle = self.get_params(self.degrees)
        img = transforms.functional.rotate(img, angle, self.resample, self.expand, self.center)
        return img
    
class RandomHorizontalFlipImageTarget_CNN(transforms.RandomHorizontalFlip):
    def __call__(self, img):    
        if random.random() < self.p:
            img = transforms.functional.hflip(img)            

        return img

class RandomVerticalFlipImageTarget_CNN(transforms.RandomVerticalFlip):
    def __call__(self, img):    
        if random.random() < self.p:
            img = transforms.functional.vflip(img)            

        return img

class RandomPerspectiveImageTarget_CNN(transforms.RandomPerspective):
    def __call__(self, img): 
        if not transforms.functional._is_pil_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        if random.random() < self.p:
            width, height = img.size
            startpoints, endpoints = self.get_params(width, height, self.distortion_scale)
            img = transforms.functional.perspective(img, startpoints, endpoints, self.interpolation)            

        return img

class ComposeImageTarget_CNN(transforms.Compose):
    def __call__(self, img): 
            for t in self.transforms:
                img = t(img)
            return img 
        
        
class DataAugmentation_CNN:
    "DataAugmentation class with `size(height,width)`"
    def __init__(self, rotation=10,horizontal_flip=0.5,
                 vertical_flip=0.5,warp=0.1, zoom=0.8, brightness=0.5, contrast=0.5, GaussianBlur=1):
                 
        
        self.lightning_transforms = transforms.Compose([transforms.ColorJitter(brightness=brightness,contrast=contrast),
                                                        #transforms.GaussianBlur(kernel_size=GaussianBlur)
                                                       ])

        self.affine_transforms = ComposeImageTarget_CNN([                                                     
                                                     RandomRotationImageTarget_CNN((-rotation,rotation)),
                                                     RandomHorizontalFlipImageTarget_CNN(horizontal_flip),
                                                     RandomVerticalFlipImageTarget_CNN(vertical_flip),
                                                     RandomPerspectiveImageTarget_CNN(distortion_scale=warp),
                                                     #transforms.RandomResizedCrop(size=size,scale=(zoom,1.0),ratio=(1.0,1.0))
                                                    ])        
        
    def transform(self,features):
                
        #do lighting transforms for features
        features = self.lightning_transforms(features)
        
        #do affine transforms for features and labels        
        features = self.affine_transforms(features)            
        
        return features     
    
        
class accuracy_metric(LearnerCallback):      

    def get_metric_names(self):
        return ["accuracy_train", "accuracy_valid"]
        
    def __calc_metrics(self, targets, outputs, train):                    
        n = targets.shape[0]
        outputs = outputs.argmax(dim=-1).view(n,-1)
        targets = targets.view(n,-1)
        res = (outputs==targets).float().mean()
        if train:            
            self.metrics_values_train.append(res)
        else:
            self.metrics_values_valid.append(res)
        
    def on_batch_end(self, last_output, last_target, train):
        self.__calc_metrics(last_target,last_output, train)
    
    def on_epoch_begin(self):
        self.metrics_values_train = []
        self.metrics_values_valid = []
    
    def on_epoch_end(self):        
        return [np.array(self.metrics_values_train).mean(), np.array(self.metrics_values_valid).mean()]

class RegressionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        r"""Class for PointLoss calculation.
        """           
    def forward(self, input, target, masks, hulls):                
        return torch.mean(torch.abs(input - target))

    
class CNNLearner:
    def __init__(self, root_path, get_y_data, images_path, num_classes, size=(512,512), bs=-1, items_count=-1, gpu_id=0, 
                 norm_stats=None, data_aug=None, preload=False, sample_results_path="sample_results",
                 init_features=16, valid_images_store="valid_images.npy", image_convert_mode="L", metric_counter=1, 
                 lr=1e-03, file_filters_include=None, file_filters_exclude=None, clahe=False, metric=None,use_softmax=False,
                 disable_metrics=False, file_prefix="", loss_func=None, weight_decay=0, ntype="resnet18", pretrained=True,
                 freeze_net=True, show_label_func=None, early_stopping_metric=None):
        
        r"""Class for train an CNN-style Neural Network

        Args:
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
        #assert power_of_2(size), "size must be a power of 2, to work with this class"
        
        #static variables (they are fix)                
        self.__size = size
        self.__root_path = Path(root_path)
        self.__images_path = Path(images_path)
        self.__sample_results_path = Path(sample_results_path)  
        (self.__root_path/self.__sample_results_path).mkdir(parents=True, exist_ok=True)
        self.__num_classes = num_classes   
        self.__show_label_func = show_label_func if show_label_func is not None else lambda a,b,c: None        
        self.__gpu_id = gpu_id
        self.__file_prefix = file_prefix
        self.pretrained =pretrained
        self.freeze_net = freeze_net
        self.init_features = init_features
        data_aug = DataAugmentation_CNN() if data_aug is None else data_aug
        
        if norm_stats is None:
            norm_stats = ([0.131],[0.308]) if image_convert_mode == "L" else ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

        
        # dynamic variables (they can change during learning)        
        self.__epochs = 0
        self.__train_losses = None
        self.__valid_losses = None
        self.__metrics = None        
        
        file_filters_include = np.array(file_filters_include) if file_filters_include is not None else None
        file_filters_exclude = np.array(file_filters_exclude) if file_filters_exclude is not None else None
        
        
        self.__create_learner(get_y_data=get_y_data, file_filters_include=file_filters_include, file_filters_exclude=file_filters_exclude, 
                              valid_images_store=valid_images_store, items_count=items_count, bs=bs, metric=metric,
                              data_aug=data_aug,image_convert_mode=image_convert_mode, metric_counter=metric_counter,use_softmax=use_softmax,
                              lr=lr, clahe=clahe, norm_stats=norm_stats, init_features=init_features, disable_metrics=disable_metrics, loss_func=loss_func,
                              weight_decay = weight_decay, ntype=ntype, num_classes=num_classes, early_stopping_metric=early_stopping_metric)
        
    def __create_learner(self, get_y_data, file_filters_include, file_filters_exclude, valid_images_store, items_count, bs, data_aug, 
                         image_convert_mode, metric_counter, lr, clahe, norm_stats, metric,use_softmax,
                         init_features, disable_metrics, loss_func, weight_decay, ntype, num_classes, early_stopping_metric):
                                
        training_data, valid_data = self.__load_data(get_y_data=get_y_data,file_filters_include=file_filters_include,
                                                     file_filters_exclude = file_filters_exclude, valid_images_store=valid_images_store, 
                                                     items_count=items_count)                       
                                      
        self.__in_channels = 1 if image_convert_mode == "L" else 3                                                                             

        
        self.train_dataset = CustomHeatmapDataset_CNN(data=training_data, grayscale=image_convert_mode == "L", 
                                                  normalize_mean=norm_stats[0],normalize_std=norm_stats[1], data_aug=data_aug, clahe=clahe,
                                                  size=self.__size)

        self.valid_dataset = CustomHeatmapDataset_CNN(data=valid_data, grayscale=image_convert_mode == "L",
                                                  normalize_mean=norm_stats[0],normalize_std=norm_stats[1], is_valid=True, clahe=clahe,
                                                  size=self.__size)                                

        #TODO Implement Metric
        metric = None if disable_metrics else metric()
        
        net = self.__get_net(ntype, init_features, num_classes, use_softmax).to(torch.device("cuda:"+str(self.__gpu_id)))
        
        #opt = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=weight_decay)
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        
        loss_func = torch.nn.CrossEntropyLoss() if loss_func is None else loss_func               
        loss_func = loss_func.to(torch.device("cuda:"+str(self.__gpu_id)))        
        
        if bs == -1:      
            
            batch_estimator = Batch_Size_Estimator(net=net, opt=opt,
                                                   loss_func=loss_func, 
                                                   gpu_id=self.__gpu_id, dataset = self.train_dataset)
            bs = batch_estimator.find_max_bs()
                                 
        
        train_dl = DataLoader(self.train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers(), pin_memory=True)
        valid_dl = DataLoader(self.valid_dataset, batch_size=bs, shuffle=True, num_workers=num_workers(), pin_memory=True)                                                       
        
        self.learner = Learner(model=net,loss_func=loss_func, train_dl=train_dl, valid_dl=valid_dl,
                               optimizer=opt, learner_callback= metric,gpu_id= self.__gpu_id, masks_and_hulls=False,
                               early_stopping_metric = early_stopping_metric)                 
           
    def __get_net(self, ntype, init_features, num_classes, use_softmax):
        if ntype == "resnet18":
             net = resnet18(init_features=init_features,num_classes=num_classes, freeze_net=self.freeze_net, pretrained=self.pretrained, use_softmax=use_softmax)
        elif ntype == "resnet34":
             net = resnet34(init_features=init_features,num_classes=num_classes, freeze_net=self.freeze_net, pretrained=self.pretrained, use_softmax=use_softmax)                      
        else:
            raise("Net type ´"+ ntype+"´ is not implemented!")     
            
            
        return net
    
    def __load_data(self, get_y_data, file_filters_include, file_filters_exclude, valid_images_store, items_count):
        
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
                        if filename.name.find((ffilter)) != -1:
                            new_filenames.append(filename)
                            break
                all_image_files = np.array(new_filenames) 
            
            if file_filters_exclude is not None:
                new_filenames = []
                for filename in all_image_files:
                    include_in = True
                    
                    for ffilter in file_filters_exclude:
                        if filename.name.find((ffilter)) != -1:
                            include_in = False                            
                            break
                    
                    if include_in:
                        new_filenames.append(filename)
                
                all_image_files = np.array(new_filenames) 
                
            return all_image_files
        
        all_image_files = list((self.__root_path/self.__images_path).glob("*.png"))+list((self.__root_path/self.__images_path).glob("*.jpg"))
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
            
            y_data = get_y_data(img_file.name)                                      
            
            if img_file in valid_images_files:                
                valid_data.append((img_file,y_data))
            else:
                training_data.append((img_file,y_data))        

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
                
    def export_to_onnx(self, filename=None, out_folder=None, batch_size=1):
        
        input_names = ["input"]
        output_names = ["output"]  
        if out_folder is None:
            onnx_export = (self.__root_path/"onnx_export")
        else:
            onnx_export = (self.__root_path/out_folder)
            
        onnx_export.mkdir(parents=True, exist_ok=True)
        
        batch = torch.zeros((batch_size,3,self.__size[0],self.__size[1]), 
                           requires_grad=True, device="cuda:"+str(self.__gpu_id))
                
        
        for i in range(batch_size):
            batch[i] = self.valid_dataset[i][1]
                    
        filename = "export" if filename is None else filename 
                
        torch_out = self.learner.model(batch)
        torch.onnx.export(self.learner.model, batch,onnx_export/(filename+".onnx"), 
                          verbose=False, input_names=input_names, output_names=output_names)
        
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
        
        #test_onnx()
                
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
          
        
        dataset = self.train_dataset if train else self.valid_dataset
        
        train_dataloader = DataLoader(dataset, batch_size=count_img, shuffle=True, num_workers=num_workers())
        data_iter = iter(train_dataloader)
        names, images, labels, masks, hulls = next(data_iter)
        for idx in range(len(images)):
            images[idx] = dataset.un_normalize(images[idx])
        nrows = ceil(count_img/3)
        f,axes = plt.subplots(nrows=nrows,ncols=3,figsize=(15,5*nrows))
        axes = axes.flatten()[:count_img]
        for name, image,label,ax in zip(names, images, labels, axes.flatten()):
            show_feat_func(ax, image)
            self.__show_label_func(ax, label, name)                
                
    def predict(self,inp, eval_mode=True):        
        return self.learner.predict(inp[None].to(torch.device("cuda:"+str(self.__gpu_id))), eval_mode=eval_mode).detach().cpu()                