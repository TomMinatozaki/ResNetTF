import numpy as np
import os
from setting import *
import cv2
import math
import matplotlib.pyplot as plt
import pickle
from xml.etree.ElementTree import Element, SubElement, dump, ElementTree, parse




def nms(y_pred_conf, y_pred_loc, prob):
        # Keep track of boxes for each class
        #class_boxes = [[],[],[]]  # class -> [(x1, y1, x2, y2, prob), (...), ...]
        class_boxes = []  # class -> [(x1, y1, x2, y2, prob), (...), ...]
        for h in range(NUM_CLASSES):
            class_boxes.append([])
        

        # Go through all possible boxes and perform class-based greedy NMS (greedy based on class prediction confidence)
        y_idx = 0
        for fm_size in FM_SIZES:
            fm_h, fm_w = fm_size  # feature map height and width
            for row in range(fm_h):
                for col in range(fm_w):
                    for db in DEFAULT_BOXES:
                        # Only perform calculations if class confidence > CONF_THRESH and not background class
                        if prob[y_idx] > CONF_THRESH and y_pred_conf[y_idx] > 0.:
                            # Calculate absolute coordinates of predicted bounding box
                            xc, yc = col + 0.5, row + 0.5  # center of current feature map cell
                            center_coords = np.array([xc, yc, xc, yc])
                            abs_box_coords = center_coords + y_pred_loc[y_idx*4 : y_idx*4 + 4]  # predictions are offsets to center of fm cell

                            # Calculate predicted box coordinates in actual image
                            scale = np.array([float(IMG_W)/fm_w, float(IMG_H)/fm_h, float(IMG_W)/fm_w, float(IMG_H)/fm_h])
                            box_coords = abs_box_coords * scale
                            box_coords = [int(round(x)) for x in box_coords]

                            # Compare this box to all previous boxes of this class
                            cls = int(y_pred_conf[y_idx])
                            cls_prob = prob[y_idx]
                            box = (box_coords, cls, cls_prob)
                            #print (box[-1])
                            if len(class_boxes[cls]) == 0:
                                class_boxes[cls].append(box)
                            else:
                                suppressed = False  # did this box suppress other box(es)?
                                overlapped = False  # did this box overlap with other box(es)?
                                for other_box in class_boxes[cls]:
                                    #print(box)
                                    #print(other_box)
                                    #iou = calc_iou(box[:4], other_box[:4])
                                    iou = calc_iou(box[0], other_box[0])
                                    if iou > NMS_IOU_THRESH:
                                        
                                        overlapped = True
                                        
                                        if box[-1] > other_box[-1]:
                                            class_boxes[cls].remove(other_box)
                                            suppressed = True
                                if suppressed or not overlapped:
                                    class_boxes[cls].append(box)

                        y_idx += 1

        # Gather all the pruned boxes and return them
        boxes = []
        for cls in range(len(class_boxes)):
            for class_box in class_boxes[cls]:
                boxes.append(class_box)
        boxes = np.array(boxes)

        return boxes

    
def calc_iou(box_a, box_b):
        """
        Calculate the Intersection Over Union of two boxes
        Each box specified by upper left corner and lower right corner:
        (x1, y1, x2, y2), where 1 denotes upper left corner, 2 denotes lower right corner
        Returns IOU value
        """
        # Calculate intersection, i.e. area of overlap between the 2 boxes (could be 0)
        # http://math.stackexchange.com/a/99576
        x_overlap = max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
        y_overlap = max(0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
        intersection = x_overlap * y_overlap

        # Calculate union
        area_box_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_box_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_box_a + area_box_b - intersection

        if union ==0:
            return 0
        
        iou = float(intersection) / union
        return iou
    
    
    
def next_batch(X, y_conf, y_loc, batch_size):
        """
        Next batch generator
        Arguments:
            * X: List of image file names
            * y_conf: List of ground-truth vectors for class labels
            * y_loc: List of ground-truth vectors for localization
            * batch_size: Batch size
        Yields:
            * images: Batch numpy array representation of batch of images
            * y_true_conf: Batch numpy array of ground-truth class labels
            * y_true_loc: Batch numpy array of ground-truth localization
            * conf_loss_mask: Loss mask for confidence loss, to set NEG_POS_RATIO
        """
        start_idx = 0
        while True:
            images = X[start_idx : start_idx + batch_size]
            y_true_conf = np.array(y_conf[start_idx : start_idx + batch_size])
            y_true_loc  = np.array(y_loc[start_idx : start_idx + batch_size])

            # Read images from image_files
            #images = []
            #for image_file in image_files:
            #	image = Image.open('resized_images_%sx%s/%s' % (IMG_W, IMG_H, image_file))
            #	image = np.asarray(image)
            #	images.append(image)

            #images = np.array(images, dtype='float32')

            #images=X
            # Grayscale images have array shape (H, W), but we want shape (H, W, 1)
            #if NUM_CHANNELS == 1:
            #	images = np.expand_dims(images, axis=-1)

            # For y_true_conf, calculate how many negative examples we need to satisfy NEG_POS_RATIO
            num_pos = np.where(y_true_conf > 0)[0].shape[0]
            num_neg = NEG_POS_RATIO * num_pos
            y_true_conf_size = np.sum(y_true_conf.shape)

            # Create confidence loss mask to satisfy NEG_POS_RATIO
            if num_pos + num_neg < y_true_conf_size:
                conf_loss_mask = np.copy(y_true_conf)
                conf_loss_mask[np.where(conf_loss_mask > 0)] = 1.

                # Find all (i,j) tuples where y_true_conf[i][j]==0
                zero_indices = np.where(conf_loss_mask == 0.)  # ([i1, i2, ...], [j1, j2, ...])
                zero_indices = np.transpose(zero_indices)  # [[i1, j1], [i2, j2], ...]

                # Randomly choose num_neg rows from zero_indices, w/o replacement
                chosen_zero_indices = zero_indices[np.random.choice(zero_indices.shape[0], int(num_neg), False)]

                # "Enable" chosen negative examples, specified by chosen_zero_indices
                for zero_idx in chosen_zero_indices:
                    i, j = zero_idx
                    conf_loss_mask[i][j] = 1.

            else:
                # If we have so many positive examples such that num_pos+num_neg >= y_true_conf_size,
                # no need to prune negative data
                conf_loss_mask = np.ones_like(y_true_conf)

            yield (images, y_true_conf, y_true_loc, conf_loss_mask)

            # Update start index for the next batch
            start_idx += batch_size
            if start_idx >= X.shape[0]:
                start_idx = 0



            
def nms_gt(y_pred_conf, y_pred_loc):
        # Keep track of boxes for each class
        class_boxes = [[],[],[]]  # class -> [(x1, y1, x2, y2, prob), (...), ...]


        # Go through all possible boxes and perform class-based greedy NMS (greedy based on class prediction confidence)
        y_idx = 0
        for fm_size in FM_SIZES:
            fm_h, fm_w = fm_size  # feature map height and width
            for row in range(fm_h):
                for col in range(fm_w):
                    for db in DEFAULT_BOXES:
                        # Only perform calculations if class confidence > CONF_THRESH and not background class
                        if  y_pred_conf[y_idx] > 0.:
                            # Calculate absolute coordinates of predicted bounding box
                            xc, yc = col + 0.5, row + 0.5  # center of current feature map cell
                            center_coords = np.array([xc, yc, xc, yc])
                            abs_box_coords = center_coords + y_pred_loc[y_idx*4 : y_idx*4 + 4]  # predictions are offsets to center of fm cell

                            # Calculate predicted box coordinates in actual image
                            scale = np.array([float(IMG_W)/fm_w, float(IMG_H)/fm_h, float(IMG_W)/fm_w, float(IMG_H)/fm_h])
                            box_coords = abs_box_coords * scale
                            box_coords = [int(round(x)) for x in box_coords]

                            # Compare this box to all previous boxes of this class
                            cls = int(y_pred_conf[y_idx])
                            box = (box_coords, cls)
                            #if len(class_boxes[cls]) == 0:
                            class_boxes[cls].append(box)
                            
                        y_idx += 1

        # Gather all the pruned boxes and return them
        boxes = []
        for cls in range(len(class_boxes)):
            for class_box in class_boxes[cls]:
                boxes.append(class_box)
        boxes = np.array(boxes)

        return boxes
    
def load_xml(xml_name):
    tree = parse(xml_name)
    root = tree.getroot()

    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    class_names = []

    for obj in root.iter("object"):
        for cls_name in obj.iter("name"):
            class_names.append(cls_name.text)
        for bndbox in obj.iter("bndbox"):
            for val in bndbox.iter("xmin"):
                value=int(float(val.text))
                xmins.append(value)
            for val in bndbox.iter("ymin"):
                value=int(float(val.text))
                ymins.append(value)
            for val in bndbox.iter("xmax"):
                value=int(float(val.text))
                xmaxs.append(value)
            for val in bndbox.iter("ymax"):
                value=int(float(val.text))
                ymaxs.append(value)
    return xmins, ymins, xmaxs, ymaxs, class_names


def find_gt_boxes(data_raw,img_name):
    signs_class=[]
    signs_box_coords=[]
    signs_data=data_raw[img_name]
    sign_data=signs_data['class']
    if(len(signs_data['box_coords'])==0):
            return 0,0,0
    abs_box_coords = signs_data['box_coords']
    #print(signs_data['box_coords'])
    global_scale = np.array([IMG_W, IMG_H, IMG_W, IMG_H])
    
    #box_coords = np.array(abs_box_coords).astype(float) / scale
    signs_class.append(sign_data)
    #signs_box_coords.append(box_coords)


    y_true_len=0
    for fm_size in FM_SIZES:
            y_true_len += fm_size[0] * fm_size[1] * NUM_DEFAULT_BOXES
    y_true_conf = np.zeros(y_true_len)
    y_true_loc = np.zeros(y_true_len * 4)

    match_counter=0
    #for i,gt_box_coords in enumerate(signs_box_coords):
    for j in range(len(abs_box_coords)):
        abs_box_coord=abs_box_coords[j]
        gt_box_coord = np.array(abs_box_coord).astype(float) / global_scale
        is_matched=0
        #print('abs_coord')
        #print(abs_box_coord)
        
        y_true_idx=0
        
        for fm_size in FM_SIZES:
        
            fm_h,fm_w=fm_size
            scale=np.array([fm_w,fm_h,fm_w,fm_h])
            for row in range(int(fm_h)):
                for col in range(int(fm_w)):
                    for db in DEFAULT_BOXES:
                        x1,y1,x2,y2=db
                        abs_db_box_coord=np.array([max(0,col+x1),max(0,row+y1),min(fm_w,col+x2),min(fm_h,row+y2)])
                        db_box_coord=(abs_db_box_coord.astype(float)) / scale
                        
                        iou=calc_iou(gt_box_coord,db_box_coord)
                        
                        
                        if iou>=IOU_THRESH:
                            
                            #print(signs_class)
                            y_true_conf[y_true_idx]=1
                            match_counter+=1
                            is_matched=1

                            abs_box_center=np.array([col+0.5,row+0.5])
                            #abs_box_center=np.array([col+1,row+1])
                            abs_gt_box_coord=gt_box_coord*scale.astype(float)
                            
                            norm_box_coord = abs_db_box_coord - np.concatenate((abs_box_center, abs_box_center))
                            y_true_loc[y_true_idx*4 : y_true_idx*4 + 4] = norm_box_coord
                           
        
                        y_true_idx+=1
            
      
    return y_true_conf,y_true_loc,match_counter



'''  
def split4(img):
    n= np.shape(img)[0]
    n = n / 2
    overlap = 30
    
    the_list={}
    the_list['0']=(img[0:n+overlap, 0:n])
    the_list['1']=(img[0:n+overlap, n:2*n])
    the_list['2']=(img[n-overlap:2*n, 0:n])
    the_list['3']=(img[n-overlap:2*n, n:2*n])
    
    return the_list

def merge4(imgs,boxes):
    n=512
    overlap=30
    merged=np.zeros((1024,1024,3))
    merged[0:n+overlap, 0:n]    = imgs['0']
    merged[0:n+overlap, n:2*n]  = imgs['1']
    merged[n-overlap:2*n, 0:n]   = imgs['2']
    merged[n-overlap:2*n, n:2*n] = imgs['3']
    merged_box=[]
    
    for item in boxes['0']:
        new_box=item
        merged_box.append(new_box)
    for item in boxes['1']:
        new_box=(item[0],item[1]+n-overlap,item[2],item[3]+n-overlap)
        merged_box.append(new_box)
    for item in boxes['2']:
        new_box=(item[0]+n-overlap,item[1],item[2]+n-overlap,item[3])
        merged_box.append(new_box)
    for item in boxes['3']:
        new_box=(item[0]+n-overlap,item[1]+n-overlap,item[2]+n-overlap,item[3]+n-overlap)
        merged_box.append(new_box)
    
    return merged,merged_box
'''

def split4(img):
    n= np.shape(img)[0]
    n = n / 2
    overlap = 30
    
    the_list={}
    the_list['0']=(img[0:n+overlap, 0:n])
    the_list['1']=(img[0:n+overlap, n:2*n])
    the_list['2']=(img[n-overlap:2*n, 0:n])
    the_list['3']=(img[n-overlap:2*n, n:2*n])
    
    return the_list

def merge4(imgs):
    n=512
    overlap=30
    merged=np.zeros((1024,1024,3))
    merged[0:n+overlap, 0:n]    = imgs[0]
    merged[0:n+overlap, n:2*n]  = imgs[1]
    merged[n-overlap:2*n, 0:n]   = imgs[2]
    merged[n-overlap:2*n, n:2*n] = imgs[3]
    merged_box=[]
    
    return merged
    
    
    
    
    
    
    