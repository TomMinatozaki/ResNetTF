
import numpy as np
import pickle
import re
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from setting import *
from inputs import *
from functions import *
from xml.etree.ElementTree import Element, SubElement, dump, ElementTree, parse


os.chdir(DATA_DIR)
img_name_list=os.listdir('./JPEGImages/')

data_raw={}
for img_name in img_name_list:
    
    img=cv2.imread('./JPEGImages/'+img_name)
    xml_name=img_name[0:-4]+'.xml'
    annots=load_xml('./Annotations/'+xml_name)
    the_list={}
    the_list['class']=[]
    the_list['box_coords']=[]
    for j in range(len(annots[0])):
            coord=[item[j] for item in annots][:-1]
            coord=np.array(coord)
            cls=[item[j] for item in annots][-1]
            the_list['box_coords'].append(coord)
            the_list['class'].append(cls)
    data_raw[img_name]=the_list
  
data_raw_sd={}

for key in data_raw.keys():
    d=data_raw[key]
    the_list={}
    the_list['class']=[]
    the_list['box_coords']=[]
    for k in range(len(d['class'])):
        if d['class'][k]=='sd':
            coord=d['box_coords'][k]
            coord=np.array(coord)
            the_list['box_coords'].append(coord)
            the_list['class'].append('sd')
    if len(the_list['class']) > 0:
        data_raw_sd[key]=the_list
        
IOU_THRESH = 0.55

data_prep = {}
c=1
for img_name in data_raw.keys():
    y_true_conf, y_true_loc, match_counter=find_gt_boxes(data_raw,img_name)
    if match_counter==0:
        continue
    if match_counter > 0:
        data_prep[img_name] = {'y_true_conf': y_true_conf, 'y_true_loc': y_true_loc}
    #print(match_counter)
    c+=1
    if c>800:
        break
        
np.save('data_prep_sd.npy',data_prep)