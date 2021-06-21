from numba import jit, njit
import numba
from PIL import Image
import numpy as np
from .heatmap_to_points import *
try:
    import cv2
except ModuleNotFoundError:
    printing("UNALBE TO IMPORT OpenCV", print_types.WARNING)  

__all__ = ["square_image","load_image","load_heatmap","batch_and_normalize","show_image_cv","get_image_daheng_camera",
           "get_mean_point_of_activations", "get_image_points", "image_enhance", "draw_middle_lines", "draw_keypoints"]


gray_norm_stats = ([0.131],[0.308])
rgb_norm_stats = ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

def square_image(image):
    image = square_image_njit(image)
    if image.shape[0] != image.shape[1]:
        image = cv2.resize(image, (image.shape[0],image.shape[0]))
    return image
    
@njit
def square_image_njit(image):
    crop = int(abs(image.shape[1] - image.shape[0])/2)    
    if (crop == 0):
        return image
    else:    
        return image[:,crop:-crop,...]

def load_image(img_file, convert_mode="RGB", size=(512,512), to_numpy=True, after_open=None):
    "Load PIL.Image from `fn` with `size(height, width)`"
    img = Image.open(img_file).resize((size[1],size[0])).convert(convert_mode)    
    if  to_numpy:
        img = np.asarray(img)
        
    if after_open is not None:
        img = after_open(img)
        
    return img

def load_heatmap(fn, size=(512,512)):
    "Load mask in PIL.Image from `fn` with `size(height, width)`"
    return load_image(fn, size=size, convert_mode="L", to_numpy=False)

def batch_and_normalize(img, mean, std):
    data = ((img/255)-mean) / std
    data = data if len(data.shape) == 3 else data[None]
    return data[None]

def show_image_cv(image, window):
    cv2.imshow(window, image)

def get_image_daheng_camera(cam):
    try:
        # get image from daheng camera
        raw_image = cam.data_stream[0].get_image()
        np_image = raw_image.get_numpy_array()         
        return square_image(np_image)
    except:        
        print("NO IMAGE FROM CAMERA")
        return None

def get_mean_point_of_activations(mask, thres=0.5):
    return np.flip(np.mean(np.array(np.where(mask>thres)).T, axis=0).astype(np.int))


def get_image_points(predictions, heatmap_types):
    out = {}
    for i in range(len(heatmap_types)):
        if heatmap_types[i][1] == "max_confidence_point":
            out[heatmap_types[i][0]] = heatmap_to_max_confidence_point(predictions[i])
        elif heatmap_types[i][1] == "circle_points":
            out[heatmap_types[i][0]] = heatmap_to_circle(predictions[i])
        elif heatmap_types[i][1] == "multiple_points":
            out[heatmap_types[i][0]] = heatmap_to_multiple_points(predictions[i])
        else:
            raise("heatmap type '" + heatmap_types[i][1]+"' is not implemented!")
    return out

kernel_sharpen = np.array([[-1,-1,-1,-1,-1],
                             [-1,2,2,2,-1],
                             [-1,2,8,2,-1],
                             [-1,2,2,2,-1],
                             [-1,-1,-1,-1,-1]]) / 8.0

clahe = cv2.createCLAHE(clipLimit=20.0,tileGridSize=(30,30))

def image_enhance(img):
    #return cv2.filter2D(clahe.apply(img), -1, kernel_sharpen)
    return clahe.apply(img)

def draw_middle_lines(image, X=True,Y=True, near_dist=None, line_thickness=1, convert_to_RGB=True):    
    
    if (convert_to_RGB == True) and (len(image.shape) == 2): 
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if X:
        x_middle = int(image.shape[1]/2)
        image = cv2.line(image, (x_middle, 0), (x_middle, image.shape[0]-1), (0,0,255), line_thickness)
        if near_dist is not None:
            near_dist_abs = int(image.shape[1] * near_dist)
            image = cv2.line(image, (x_middle-near_dist_abs, 0), (x_middle-near_dist_abs, image.shape[0]-1), (0,255,0), line_thickness)
            image = cv2.line(image, (x_middle+near_dist_abs, 0), (x_middle+near_dist_abs, image.shape[0]-1), (0,255,0), line_thickness)

    if Y:
        y_middle = int(image.shape[0]/2)
        image = cv2.line(image, (0, y_middle), (image.shape[0]-1, y_middle), (0,0,255), line_thickness)
        if near_dist is not None:
            near_dist_abs = int(image.shape[0] * near_dist)
            image = cv2.line(image, (0, y_middle-near_dist_abs), ( image.shape[0]-1, y_middle-near_dist_abs), (0,255,0), line_thickness)
            image = cv2.line(image, (0, y_middle+near_dist_abs), ( image.shape[0]-1, y_middle+near_dist_abs), (0,255,0), line_thickness)

    return image

#centerpoint, nutedges, outer_circlepoints, inner_circlepoints, x_middle=None, y_middle=None
def draw_keypoints(image, img_points, heatmap_types, convert_to_RGB=True):
    #inner_circle_radius = None
    #outer_circle_mean = None
    #nut_mean = None
    #x_circle_diff = None
    #y_circle_diff = None
    #x_center_diff = None
    #y_center_diff = None
    
    if (convert_to_RGB == True)  and (len(image.shape) == 2):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    for i in range(len(heatmap_types)):
        if heatmap_types[i][1] == "max_confidence_point":
            image = cv2.circle(image, tuple(img_points[heatmap_types[i][0]].astype(np.int)), 3, heatmap_types[i][2], 2)                    
        elif (heatmap_types[i][1] == "circle_points") or heatmap_types[i][1] == "multiple_points":
            multiple_points = img_points[heatmap_types[i][0]]
            circle_mean = np.mean(multiple_points, axis=0)
            image = cv2.circle(image, tuple(circle_mean.astype(np.int)), 3, heatmap_types[i][2], 2)
            for point in multiple_points:
                image = cv2.circle(image, tuple(point.astype(np.int)), 2, heatmap_types[i][2], 3)                
        else:
            raise("heatmap type '" + heatmap_types[i][1]+"' is not implemented for drawing!")
    return image




