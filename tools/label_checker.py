import json
from .helper import get_labels, replace_values
import cv2
from pathlib import Path
import numpy as np
import pdb
import os

def check_labels(params):
    json_path=params["json_path"]
    image_ext=params["image_ext"]
    file_name_function = params["file_name_function"]
    features = params["features"]
    root_path = params["root_path"]
    img_path = params["img_path"]
    paint_color = params["paint_color"]
    replacements = params["replacements"]
    declined_file_name = params["declined_file_name"]
    if os.path.isfile(declined_file_name):
        declined_labels = list(np.load(declined_file_name))
        start_idx = int(declined_labels[0])
    else:
        declined_labels = [0]
        start_idx = 0
        

    labels = json.load(open(json_path,"r"))
    replace_values(labels, replacements)
    count = len(labels)
    
    for idx,label in enumerate(labels):
        if idx < start_idx:
            continue
        # get labels
        label_data,label_types, heat_radiuses = get_labels(label, features)
        # get file name
        file = file_name_function(label, image_ext)
        # load img file
        img = cv2.imread(str(root_path/img_path/file))
        for key in label_data.keys():
            points = label_data[key]
            for pt in points:                    
                img = cv2.circle(img, tuple(pt.astype(np.int)),1,paint_color[key],10)
                
        frame = cv2.putText(img, str(idx+1) +" von " + str(count),(1,120), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0),4)
        
        img = cv2.resize(img, (800,800), interpolation = cv2.INTER_CUBIC)
        cv2.imshow("Image", img)
        while True:
            keyCode = cv2.waitKey(0)
        
            if keyCode == 97:
                # keycode A - accepted
                declined_labels[0] = idx+1
                print("accepted")
                break
            if keyCode == 115:
                # keyCode S - declined
                print("declined")
                declined_labels[0] = idx+1
                declined_labels.append(file)
                break
            if keyCode == 27:
                # keycode esc - terminate
                return
            
        np.save(declined_file_name, np.array(declined_labels))
            
          