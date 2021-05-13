import cv2
import numpy as np
from numba import njit
import numba
import pdb
from scipy.interpolate import splprep, splev

def heatmap_to_multiple_points(pred, thres=0.5, max_points=100):    
    mask = (pred > thres).astype(np.uint8)
    if int(cv2.__version__[0]) < 4:
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if (len(contours) == 0) or (len(contours) > max_points):
        return None
    nut_points = np.zeros((len(contours),2))
    for i in range(len(contours)):
        nut_points[i] = np.mean(contours[i][:,0,:], axis=0)
    return nut_points

def heatmap_to_max_confidence_point(heatmap, thres=0.5):
    center_point = None
    if heatmap.max() > thres:
        center_point = np.flip(np.unravel_index(heatmap.argmax(), heatmap.shape)) 
    return center_point


def heatmap_to_circle_HoughCircles(mask, cp):            
    radius = get_radius(mask,cp[0],cp[1], add_factor=0.0)
    if not is_valid_radius(radius,cp,mask.shape):
        return None
    
    radius_range = 40
    mind_circle_center_dist = 100
    param1 = 100
    param2 = 30
    
    circle = np.squeeze(cv2.HoughCircles(image=np.round(mask*255).astype(np.uint8), method=cv2.HOUGH_GRADIENT, dp=1, 
                                minDist=mind_circle_center_dist,param1=param1,
                                param2=param2,minRadius=radius-radius_range,
                                maxRadius=radius+radius_range).astype(np.int))
    
    return circle

@njit
def check_validity(circle_points):
    zero_counter = 0
    len_c = int(len(circle_points)*0.3)
    for idx in range(len(circle_points)):        
        if (circle_points[idx,0] == 0) and (circle_points[idx,1] == 0):
            zero_counter += 1            
            if zero_counter == len_c:
                return False
        else:
            zero_counter = 0
    
    return True                     
        

def heatmap_to_circle(mask, cp=None):    
    #pdb.set_trace()
    #radius = get_radius(mask,cp[0],cp[1])
    radius = mask.shape[0]
    #if not is_valid_radius(radius,cp,mask.shape):
        #return None
    if cp is None:    
        cp = np.flip(np.mean(np.array(np.where(mask>0.5)).T, axis=0).astype(np.int))
        
    scan_points = get_scan_points(10,cp[0],cp[1],mask.shape[0],radius)
    circle_points = get_circle_points(scan_points,cp[0],cp[1],mask)
    
    if not check_validity(circle_points):
        return None
    
    len_points = circle_points        
    circle_filter = circle_points[np.logical_and(circle_points[:,0] > 0, circle_points[:,1] > 0)]                

    try:
        tck, u = splprep([circle_filter[:,0], circle_filter[:,1]], s=0)
        new_points = splev(np.linspace(0,1,len(len_points)), tck)
        new_points = np.array(new_points).T
    except:
        return None

    return new_points

@njit
def get_radius(mask,p0,p1, add_factor=0.1):
    radiuss = np.zeros(4,dtype=numba.int64)
    # detect circle radius        
    m_north = np.flip(mask[:p1+1,p0])
    m_east = mask[p1,p0:]    
    m_south = mask[p1:,p0]    
    m_west = np.flip(mask[p1,:p0+1])
    
    radiuss[0] = np.argsort(m_north)[-1:][0]
    radiuss[1] = np.argsort(m_east)[-1:][0]
    radiuss[2] = np.argsort(m_south)[-1:][0]
    radiuss[3] = np.argsort(m_west)[-1:][0]
    
    radius = np.median(radiuss)
    return int(radius + round(radius*add_factor))

def is_valid_radius(radius,cp,shape):
    return (((cp[0] + radius) < shape[1]) and ((cp[1] + radius) < shape[0]) and ((cp[0] - radius) >= 0) and ((cp[1] - radius) >= 0))

@njit
def get_scan_points(step,cp0,cp1,shape,radius):
    angles = np.arange(0,360,step)
    scan_points = np.zeros((len(angles),2), dtype=numba.int64)
    
    for i in range(len(angles)):        
        x = round(radius*np.sin(np.deg2rad(angles[i]))+cp0)
        y = round(radius*np.sin(np.deg2rad(90-angles[i]))+cp1)

        scan_points[i,0] = x
        scan_points[i,1] = y
        
    return scan_points

@njit
def line(r0, c0, r1, c1):
    steep = 0
    r = r0
    c = c0
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr=0
    sc=0
    d=0
    i=0

    rr = np.zeros(max(dc, dr) + 1, dtype=np.intp)
    cc = np.zeros(max(dc, dr) + 1, dtype=np.intp)
    
    if (c1 - c) > 0:
        sc = 1
    else:
        sc = -1
        
    if (r1 - r) > 0:
        sr = 1
    else:
        sr = -1
    
    if dr > dc:
        steep = 1
        c, r = r, c
        dc, dr = dr, dc
        sc, sr = sr, sc
    
    d = (2 * dr) - dc

    
    for i in range(dc):
        if steep:
            rr[i] = c
            cc[i] = r
        else:
            rr[i] = r
            cc[i] = c
        while d >= 0:
            r = r + sr
            d = d - (2 * dc)
            
        c = c + sc
        d = d + (2 * dr)

    rr[dc] = r1
    cc[dc] = c1

    return rr, cc


@njit
def get_circle_points(scan_points,cp0,cp1,mask):
    
    
    circle_points = np.zeros((len(scan_points),2))
    point_diffs = np.zeros(len(scan_points)-1)
    shape = mask.shape[0]
    p_idx = 0
    for i in range(len(scan_points)):
        #pdb.set_trace()
        p = scan_points[i]
        l = line(cp0,cp1,p[0],p[1])
        discrete_line = np.zeros((len(l[0]),2),dtype=np.int64)
        discrete_line[:,0] = l[0]
        discrete_line[:,1] = l[1]
        
        x_cond = np.where(np.logical_or(discrete_line[:,0] < 0, discrete_line[:,0] > shape-1))[0]
        y_cond = np.where(np.logical_or(discrete_line[:,1] < 0, discrete_line[:,1] > shape-1))[0]
        
        idx_x = len(discrete_line) if len(x_cond) == 0 else x_cond[0]
        idx_y = len(discrete_line) if len(y_cond) == 0 else y_cond[0]
        
        
        discrete_line = discrete_line[:min(idx_x,idx_y)]
        
        intens = np.zeros(len(discrete_line))
        for lp in range(len(discrete_line)):
            intens[lp] = mask[discrete_line[lp,1],discrete_line[lp,0]]

        circle_point_idx = np.argsort(intens)[-1]
        circle_point = discrete_line[circle_point_idx]        
        # return None detected circle if confidence for circle is below 0.3
        
        if mask[circle_point[1], circle_point[0]] < 0.3:               
            #return None        
            continue
        # return None detected circle if detected circle point is 3x far away, as mean of all circle points
        #pdb.set_trace()
        if i > 0:
            if (circle_points[i-1][0] != 0) and (circle_points[i-1][1] != 0):
                point_diff = np.sqrt(np.sum((circle_points[i-1] - circle_point)**2))
                if p_idx > 0:
                    if (point_diffs[0:p_idx].mean()*3) < point_diff:                        
                        #return None
                        continue                
                point_diffs[p_idx] = point_diff        
                p_idx += 1
                
        circle_points[i] = circle_point    
    return circle_points

