import json
import numpy as np
from skimage.draw import line
from pathlib import Path
import pdb
import cv2
import math
import matplotlib.pyplot as plt
import os
import shutil
from time import sleep
import multiprocessing
import time
import pdb
from .helper import get_labels, replace_values
from fastprogress.fastprogress import progress_bar
from skimage.morphology.convex_hull import convex_hull_image

class LABELS_TO_HEATMAPS_COMVERTER:
    
    def __init__(self, root_path, img_path, json_path, features, target_size=512, image_ext="png", 
                 scales=[0.0],crop_margin=0, file_name_function=None, final_images_path="final_images", 
                 painted_images_path="painted_images", convex_hull_features=[],hull_path="hull",
                 replacements=None, process_output_image=None, mask_dilation=1):
        self.__root_path=root_path
        self.__convex_hull_features = convex_hull_features
        self.__img_path=img_path
        self.__hull_path = hull_path
        self.mask_dilation = mask_dilation
        self.__json_path=json_path
        self.__features=features            
        self.__final_images_path=final_images_path
        self.__painted_images_path=painted_images_path        
        self.__replacements = replacements
        self.__image_ext=image_ext
        self.__target_size=target_size
        self.__scales = scales
        self.__crop_margin=crop_margin
        self.__process_output_image = process_output_image
        self.__file_name_function = file_name_function if file_name_function is not None else lambda label, image_ext: label["Labeled Data"]+ "."+ image_ext 
    def process(self):
        labels = json.load(open(self.__root_path/self.__json_path,"r"))
        self.manage_folders()
        replace_values(labels, self.__replacements)
        self.new_process(labels)
        
    
    def process_multithread(self):
        raise("`process_multithread` is not working now")
        starttime = time.time()
        labels = json.load(open(self.__json_path,"r"))
        self.manage_folders()
        replace_values(labels, self.__replacements)
        cpu_count = round(multiprocessing.cpu_count()/2)            
        imgs_pre_thread = round(len(labels) / cpu_count)    

        idx = 1
        processes = []
        labels_for_thread = []
        for label in labels:

            labels_for_thread.append(label)        

            if idx%imgs_pre_thread == 0:        
                p = multiprocessing.Process(target=worker.new_process, args=(labels_for_thread,))
                processes.append(p)        
                labels_for_thread = []
                idx=1
            else:
                idx+=1

        if len(labels_for_thread)< imgs_pre_thread:
            p = multiprocessing.Process(target=worker.new_process, args=(labels_for_thread,))
            processes.append(p)

        for process in processes:
            process.start()  

        for process in processes:        
            process.join()        

        print('That took {} seconds'.format(time.time() - starttime))    
            
            
    def manage_folders(self):
        if os.path.exists(self.__root_path/self.__final_images_path):
            shutil.rmtree(self.__root_path/self.__final_images_path)
            sleep(0.1)
        os.mkdir(self.__root_path/self.__final_images_path)

        if self.__painted_images_path is not None:
            if os.path.exists(self.__root_path/self.__painted_images_path):
                shutil.rmtree(self.__root_path/self.__painted_images_path)
                sleep(0.1)
            os.mkdir(self.__root_path/self.__painted_images_path)
        
        if os.path.exists(self.__root_path/self.__hull_path):
            shutil.rmtree(self.__root_path/self.__hull_path)
            sleep(0.1)
        os.mkdir(self.__root_path/self.__hull_path)                    
        
        for key in self.__features.keys():        
            if os.path.exists(self.__root_path/key):
                shutil.rmtree(self.__root_path/key)
                sleep(0.1)
            os.mkdir(self.__root_path/key)

            
    def get_all_points(self,all_pts):
        all_points = np.array([[0,0]])
        for key in all_pts.keys():
            if len(all_pts[key]) > 0:
                all_points = np.concatenate((all_points,all_pts[key]))

        return np.delete(all_points,0,axis=0).astype(np.int)

    def adjust_labels_zoomed(self,label_data, x_diff, y_diff):
        new_label_data = {}
        for key in label_data.keys():
            new_label_data[key] = []
            for point in label_data[key]:
                new_label_data[key].append((point[0] - x_diff,point[1] - y_diff))
            new_label_data[key] = np.array(new_label_data[key])

        return new_label_data

    def adjust_labels_squared(self,label_data, value, orientation):
        new_label_data = {}
        for key in label_data.keys():
            new_label_data[key] = []
            for point in label_data[key]:
                new_x = point[0] - value if orientation == "landscape" else point[0]
                new_y = point[1] if orientation == "landscape" else point[1] - value
                new_label_data[key].append((new_x,new_y))
            new_label_data[key] = np.array(new_label_data[key])

        return new_label_data


    def adjust_labels_sized(self,label_data, old_square_shape, new_square_shape):
        new_label_data = {}
        for key in label_data.keys():
            new_label_data[key] = []
            for point in label_data[key]:            
                new_x = round(new_square_shape*(point[0]/old_square_shape))            
                new_y = round(new_square_shape*(point[1]/old_square_shape))
                new_label_data[key].append((new_x,new_y))
            new_label_data[key] = np.array(new_label_data[key])

        return new_label_data

    def zoom_image(self,label_data, img, scale):        
        all_points = self.get_all_points(label_data)
        if len(all_points) == 0:
            return label_data, img
        bb = cv2.boundingRect(all_points)
        x_from = round(bb[0]*scale)
        x_to = round(img.shape[1]-((img.shape[1]-(bb[0]+bb[2]))*scale))
        y_from = round(bb[1]*scale)
        y_to = round(img.shape[0]-((img.shape[0]-(bb[1]+bb[3]))*scale))
        img_zoomed = img[y_from:y_to,x_from:x_to,:]
        zoomed_label_data = self.adjust_labels_zoomed(label_data,x_from, y_from)
        return zoomed_label_data, img_zoomed

    def square_image(self,zoomed_label_data, img_zoomed):
        all_points = self.get_all_points(zoomed_label_data)
        if len(all_points) == 0:
            return zoomed_label_data, img_zoomed
        orientation = "landscape" if img_zoomed.shape[1] > img_zoomed.shape[0] else "portrait"

        crop_v = abs(img_zoomed.shape[1] - img_zoomed.shape[0])        

        if crop_v < 2:
            return zoomed_label_data, img_zoomed

        min_v = all_points[:,0].min() if orientation == "landscape" else all_points[:,1].min()
        max_v = all_points[:,0].max() if orientation == "landscape" else all_points[:,1].max()

        if min_v < ((crop_v/2)+self.__crop_margin):
            # cut from HIGH
            img_squared = img_zoomed[:,:-crop_v,:] if orientation == "landscape" else img_zoomed[:-crop_v,:,:]        
            squared_label_data = self.adjust_labels_squared(zoomed_label_data,0,orientation)    

        elif max_v > img_zoomed.shape[1] - ((crop_v/2)+self.__crop_margin):
            # cut LOW
            img_squared = img_zoomed[:,crop_v:,:] if orientation == "landscape" else  img_zoomed[crop_v:,:,:]
            squared_label_data = self.adjust_labels_squared(zoomed_label_data,crop_v,orientation)    
        else:
            # cut from LOW AND HIGH
            img_squared = img_zoomed[:,int(crop_v/2):-int(crop_v/2),:] if orientation == "landscape" else img_zoomed[int(crop_v/2):-int(crop_v/2),:,:]
            squared_label_data = self.adjust_labels_squared(zoomed_label_data, int(crop_v/2),orientation)

        return squared_label_data,img_squared

    def size_image(self,squared_label_data, img_squared):
        all_points = self.get_all_points(squared_label_data)
        if len(all_points) == 0:
            img_sized = cv2.resize(img_squared, (self.__target_size, self.__target_size), interpolation = cv2.INTER_CUBIC)
            return squared_label_data, img_sized

        img_sized = cv2.resize(img_squared, (self.__target_size, self.__target_size), interpolation = cv2.INTER_CUBIC)
        sized_label_data = self.adjust_labels_sized(squared_label_data, img_squared.shape[1],self.__target_size)
        return sized_label_data,img_sized

    def make_point_heat(self,pt, shape, h, initial_itensity = None):

        if type(shape) == int:
            shape = [shape, shape]

        #DEFINE GRID SIZE AND RADIUS(h)
        grid_size=1  

        #FUNCTION TO CALCULATE INTENSITY WITH QUARTIC KERNEL
        def kde_quartic(d,h):
            dn=d/h
            P=(15/16)*(1-dn**2)**2
            return P

        def kde_linear(d,h):     
            return 1-(d/h)


        #PROCESSING
        if initial_itensity is None:
            itensity = np.zeros(shape, dtype=np.float32)
        else:
            itensity = initial_itensity

        try:
            if pt[1]-h < 0:
                h = h - abs(pt[1]-h)

            if pt[0]-h < 0:
                h = h - abs(pt[0]-h)

            if pt[1]+h > shape[0]:
                h = h - ((pt[1]+h)-shape[0])

            if pt[0]+h > shape[1]:
                h = h - ((pt[0]+h)-shape[1])

            for x_p in range(pt[1]-h, pt[1]+h):
                for y_p in range(pt[0]-h, pt[0]+h):                                        
                    d=math.sqrt((x_p-pt[1])**2+(y_p-pt[0])**2)            
                    if d<=h:
                        itensity[x_p,y_p] = kde_linear(d,h)
        except:
            pdb.set_trace()
            print("exception")


        return itensity

    def get_orth_vektor_from_points(self,P,Q):   
        return np.array([Q[1] - P[1],P[0] - Q[0]])

    def check_neighbors_of_zero_and_interpolate(self,mask,p):
        # check neighbors of point in mask of zeroes. Using 4er neighborhood. neighbors with no black pixels interpolate
        try:
            kernel = np.array([[-1,0],[0,-1],[0,1],[1,0]])
            pixels = kernel + p    
            if len(np.where(mask[pixels[:,0],pixels[:,1]] == 0)[0]) < 2:

                    mask[p[0],p[1]] = np.mean(mask[pixels[:,0],pixels[:,1]]).round(2)
        except:
            pass

    def fill_holes(self,orth_lines, mask):
        # get boundaries of line segment

        if len(orth_lines) == 0:
            return
        
        boundaries = np.where(orth_lines[:,2] == 0)[0]
        
        if len(boundaries) < 4:
            return
        
        boundaries = boundaries[[0,1,-2,-1]]
        x_min = orth_lines[boundaries][:,0].astype(np.int).min()
        x_max = orth_lines[boundaries][:,0].astype(np.int).max()
        y_min = orth_lines[boundaries][:,1].astype(np.int).min()
        y_max = orth_lines[boundaries][:,1].astype(np.int).max()

        x_min = x_min if x_min >0 else 1
        x_max = x_max if x_max<mask.shape[1]-2 else mask.shape[1]-3
        y_min = y_min if y_min >0 else 1
        y_max = y_max if y_max<mask.shape[0]-2 else mask.shape[0]-3
                
        # get black pixels in segment
        mask_t = np.copy(mask[y_min-1:y_max+2,x_min-1:x_max+2])
        mask_t[[0,-1],:] = 1
        mask_t[:,[0,-1]] = 1

        # get black pixels
        black_pixels = np.array(np.where(mask_t == 0)).T

        # check wether pixel is surrounded of non black pixels and interpolate 
        # TODO MAKE FAST
        for black_pixel in black_pixels:    
            self.check_neighbors_of_zero_and_interpolate(mask[y_min-1:y_max+2,x_min-1:x_max+2],black_pixel) 

    def make_line_heat(self,p1,p2, mask, heat_radius):
        #try:
        discrete_line = np.array(list(zip(*line(*p1, *p2))))
        orth_vector = self.get_orth_vektor_from_points(p1,p2)
        if np.sqrt(np.sum(orth_vector**2)) == 0.0:
            return mask
        orth_vector_new_length = ((heat_radius/np.sqrt(np.sum(orth_vector**2)))*orth_vector).round().astype(np.int)
        discrete_line, orth_vector_new_length

        orth_lines = []

        # TODO MAKE FAST
        for line_p in discrete_line:
            orth_line = np.array(list(zip(*line(*line_p, *(line_p+orth_vector_new_length)))))        
            orth_lines.append(np.concatenate((orth_line,np.linspace(1,0,len(orth_line))[:,None]),axis=1))
            orth_line = np.array(list(zip(*line(*line_p, *(line_p-orth_vector_new_length)))))
            orth_lines.append(np.concatenate((orth_line,np.linspace(1,0,len(orth_line))[:,None]),axis=1))

        orth_lines = np.vstack(orth_lines)

        aa = orth_lines[:,0:2] > -1
        bb = orth_lines[:,0:2] < mask.shape[0]
        valid_idx = np.logical_and(np.logical_and(aa[:,0], aa[:,1]), np.logical_and(bb[:,0], bb[:,1]))
        
        mask[orth_lines[valid_idx,1].astype(np.int),orth_lines[valid_idx,0].astype(np.int)] = orth_lines[valid_idx,2]

        kernel = np.ones((3,3),np.uint8)
        dilation = cv2.dilate(mask,kernel,iterations = 3)
        close = cv2.erode(dilation,kernel,iterations = 3)
        
        return close
        #self.fill_holes(orth_lines[valid_idx], mask)
        #except:
            #pdb.set_trace()


    def make_polygonal_heatmap(self,points, close, heat_radius):
        heatmap = np.zeros((self.__target_size,self.__target_size), dtype=np.float)

        if len(points) == 0:
            return heatmap.astype(np.uint8)

        for s in np.arange(1,len(points)):
            p1 = np.array([points[s-1][0], points[s-1][1]]).round().astype(np.int)
            p2 = np.array([points[s][0], points[s][1]]).round().astype(np.int)

            '''
            if (p1[0]+(heat_radius*2)>heatmap.shape[1]-1) or (p1[1]+(heat_radius*2)>heatmap.shape[0]-1) or (p2[0]+(heat_radius*2)>heatmap.shape[1]-1) or (p2[1]+(heat_radius*2)>heatmap.shape[0]-1):
                continue

            if (p1[0]-(heat_radius*2)<0) or (p1[1]-(heat_radius*2)<0) or (p2[0]-(heat_radius*2)<0) or (p2[1]-(heat_radius*2)<0):        
                continue
            '''

            heatmap = self.make_line_heat(p1,p2,heatmap, heat_radius)

        if close:
            p1 = np.array([points[-1][0], points[-1][1]]).round().astype(np.int)
            p2 = np.array([points[0][0], points[0][1]]).round().astype(np.int)

            '''
            if (p1[0]+(heat_radius*2)>heatmap.shape[1]-1) or (p1[1]+(heat_radius*2)>heatmap.shape[0]-1) or (p2[0]+(heat_radius*2)>heatmap.shape[1]-1) or (p2[1]+(heat_radius*2)>heatmap.shape[0]-1):
                if heatmap.max() == 0.0:
                    return heatmap.astype(np.uint8)
                else:
                    return np.round(np.interp(heatmap, (heatmap.min(), heatmap.max()), (0, 255))).astype(np.uint8)

            if (p1[0]-(heat_radius*2)<0) or (p1[1]-(heat_radius*2)<0) or (p2[0]-(heat_radius*2)<0) or (p2[1]-(heat_radius*2)<0):
                if heatmap.max() == 0.0:
                    return heatmap.astype(np.uint8)
                else:
                    return np.round(np.interp(heatmap, (heatmap.min(), heatmap.max()), (0, 255))).astype(np.uint8)
            '''

            heatmap = self.make_line_heat(p1,p2,heatmap, heat_radius)

        if heatmap.max() == 0.0:
            return heatmap.astype(np.uint8)

        return np.round(np.interp(heatmap, (heatmap.min(), heatmap.max()), (0, 255))).astype(np.uint8)

    def make_points_heatmap(self,points, heat_radius):
        heatmap = np.zeros((self.__target_size,self.__target_size), dtype=np.float32)

        if len(points) == 0:
            return heatmap.astype(np.uint8)

        for pt in points:
            heatmap = self.make_point_heat([pt[0], pt[1]],self.__target_size, heat_radius, heatmap)

        return np.round(np.interp(heatmap, (heatmap.min(), heatmap.max()), (0, 255))).astype(np.uint8)   

    
    
    def write_image(self,img,path,file, is_gray=False):
        if not is_gray:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return cv2.imwrite(str(path/file), img)


    def new_process(self,labels):    
        dilate_kernel = np.ones((5,5),np.uint8)
        count = len(labels)
        pbar = progress_bar(range(len(labels)))
        for idx in pbar:
            label = labels[idx]
            # get labels
            
            label_data,label_types, heat_radiuses = get_labels(label, self.__features)

            # get file name TODO: set file name to extern function
            file = self.__file_name_function(label, self.__image_ext)

            #if file == "2020-11-11_15-44-39.584.png":
                #pdb.set_trace()

            # load image and convert to rgb 
            
            filen = self.__root_path/self.__img_path/file
            
            if not filen.exists():
                continue
            
            #pdb.set_trace()
            img = cv2.imread(str(filen))
            
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            
            for scale in self.__scales:
                sized_image_file = file[:file.find("."+self.__image_ext)]+"_sized_"+str(scale)+".png"

                #zoom images
                zoomed_label_data, img_zoomed = self.zoom_image(label_data,img, scale)

                # square images        
                squared_label_data,img_squared = self.square_image(zoomed_label_data, img_zoomed)

                # size images
                sized_label_data,img_sized = self.size_image(squared_label_data, img_squared)

                # make heatmaps
                heatmaps = []
                for asset_key in sized_label_data.keys():
                    points = sized_label_data[asset_key]
                    
                    #if (filen.stem == "10IF56K7_top") and (asset_key == "T11"):
                        #pdb.set_trace()
                    if len(points) > 0:                        
                        if (label_types[asset_key] == "polygon"):
                            heatmap = self.make_polygonal_heatmap(points,True, heat_radiuses[asset_key])
                        elif (label_types[asset_key] == "line"):
                            heatmap = self.make_polygonal_heatmap(points,False, heat_radiuses[asset_key])                        
                        elif label_types[asset_key] == "point":
                            heatmap = self.make_points_heatmap(points, heat_radiuses[asset_key])                        
                        else:
                            raise("label_type `"+label_types[asset_key]+"` not implemented yet.")
                    else:
                        heatmap = np.zeros((self.__target_size,self.__target_size), dtype=np.uint8)
                        
                    if heatmap.min() == 255:
                        heatmap = np.zeros((self.__target_size,self.__target_size), dtype=np.uint8)                        

                    self.write_image(heatmap,self.__root_path/asset_key, sized_image_file, is_gray=True)
                    
                    # make mask
                    mask = heatmap>0
                    if mask.max() == False:
                        interpolated = np.zeros((mask.shape[0],mask.shape[1]), dtype=np.uint8)
                    else:
                        dilated = cv2.dilate(mask.astype(np.uint8),dilate_kernel,iterations = self.mask_dilation)                        
                        interpolated = np.round(np.interp(dilated, (dilated.min(), dilated.max()), (0, 255))).astype(np.uint8)
                        
                    self.write_image(interpolated,self.__root_path/asset_key, Path(sized_image_file).stem + "_mask.png", is_gray=True)                                        
                    
                    if asset_key in self.__convex_hull_features:
                        heatmaps.append(heatmap)
                
                # make convex hull
                if len(heatmaps) > 0:
                    #if filen.stem == "10IF56K7_top":
                        #pdb.set_trace()

                    combined_mask = np.logical_or.reduce(np.stack(heatmaps))
                    if combined_mask.max() == False:                        
                        interpolated = np.zeros((combined_mask.shape[0],combined_mask.shape[1]), dtype=np.uint8)
                    else:                        
                        hull = convex_hull_image(combined_mask)                    
                        dilated = cv2.dilate(hull.astype(np.uint8),dilate_kernel,iterations = 3)
                        interpolated = np.round(np.interp(dilated, (dilated.min(), dilated.max()), (0, 255))).astype(np.uint8)
                    self.write_image(interpolated,self.__root_path/self.__hull_path, sized_image_file, is_gray=True)

                # paint images            
                if self.__painted_images_path is not None:
                    img_sized_paint = img_sized.copy()
                    for key in sized_label_data.keys():
                        points = sized_label_data[key]
                        for pt in points:                    
                            img_sized_paint = cv2.circle(img_sized_paint, tuple(pt),1,(255,0,0),2)

                    self.write_image(img_sized_paint,self.__root_path/self.__painted_images_path, sized_image_file, is_gray=False)            

                # save final image
                if self.__process_output_image is not None:
                    img_sized = self.__process_output_image(img_sized, sized_image_file)
                    self.write_image(img_sized, self.__root_path/self.__final_images_path, sized_image_file, is_gray=True)
                else:
                    self.write_image(img_sized, self.__root_path/self.__final_images_path, sized_image_file, is_gray=False)
                                    

                pbar.comment = sized_image_file
                #print("saved heatmaps and final image", idx+1,"from", count, sized_image_file,  end="\r")


