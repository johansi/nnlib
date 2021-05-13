import numpy as np
import os
import platform


__all__ = ["print_types","printing","get_label","get_labels","replace_values","num_workers","power_of_2"]

class print_types:
    HEADER = ('\033[95m', "HEADER")
    INFO = ('\033[94m', "INFO")
    OKCYAN = ('\033[96m', "HEADER")
    OKGREEN = ('\033[92m', "HEADER")
    WARNING = ('\033[93m', "WARNING")
    FAIL = ('\033[91m', "ERROR")
    ENDC = ('\033[0m', "HEADER")
    BOLD = ('\033[1m', "HEADER")
    UNDERLINE = ('\033[4m', "HEADER")

def printing(txt,type,new_line=False,end="\n"):
        new_line = "\n" if new_line else ""   
        print(new_line+ print_types.BOLD[0]+type[0]+"["+type[1]+"]"+print_types.ENDC[0]+ " " + txt, end=end)

def get_label(label, label_asset, label_type):
    results = []

    if "objects" in label["Label"].keys():
        for label_obj in label["Label"]["objects"]:
            if label_obj["value"] == label_asset:
                if label_type not in label_obj.keys():                
                    continue

                asset = label_obj[label_type]
                asset = [asset] if type(asset) == dict else asset            
                for item in asset:
                    results.append((item["x"], item["y"]))     

    return np.array(results)

def get_labels(label, features):        
    out = {}
    label_types = {}
    heat_radiuses = {}
    for feat in features.keys():
        heat_radiuses[feat] = features[feat]["heat_radius"]
        for l_type in features[feat]["label_type"]:                
            out[feat] = get_label(label, feat, l_type)
            if len(out[feat]) > 0:
                label_types[feat] = l_type
                break

    return out, label_types, heat_radiuses

def replace_values(labels, replacements):
    if replacements is None:
        return
    for img in labels:
        if "objects" not in img["Label"].keys():
            continue
        for label in img["Label"]["objects"]: 
            for replace in replacements:
                if label["value"] == replace[0]:
                    label["value"] = replace[1]
                    break

def num_workers():
    "Get number of workers"
    
    if  platform.system() == "Windows":
        return 0
    
    try:                   
        return len(os.sched_getaffinity(0))
    except AttributeError: 
        return os.cpu_count()                    
    
def power_of_2(n):
    if type(n) == int:
        n = [n]    
    if (type(n) == tuple) or (type(n) == list):
        for i in n:
            if not ((i & (i-1) == 0) and i != 0):
                return False
    else:
        return False
    
    return True