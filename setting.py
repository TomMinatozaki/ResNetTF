import tensorflow as tf

class_dict={
 'aeroplane': 18,
 'bicycle': 13,
 'bird': 20,
 'boat': 11,
 'bottle': 15,
 'bus': 3,
 'car': 10,
 'cat': 12,
 'chair': 2,
 'cow': 1,
 'diningtable': 19,
 'dog': 4,
 'foot': 8,
 'hand': 7,
 'head': 6,
 'horse': 17,
 'motorbike': 21,
 'person': 5,
 'pottedplant': 9,
 'sheep': 23,
 'sofa': 16,
 'train': 14,
 'tvmonitor': 22}

# Default boxes
# DEFAULT_BOXES = ((x1_offset, y1_offset, x2_offset, y2_offset), (...), ...)
# Offset is relative to upper-left-corner and lower-right-corner of the feature map cell
DEFAULT_BOXES = ((0,0,1,1),(0,0,1,2),(0,0,2,1),(0,0,1,3),
                 (0,0,3,1),(0.5,0.5,1.5,1.5),(0.5,0.5,1.5,2.5),(0.5,0.5,2.5,1.5),
                 (0.5,0.5,1.5,3.5),(0.5,0.5,3.5,1.5),(0.5,0.5,1.5,2.5),(0.5,0.5,2.5,1.5))
#1,1,1,0.53,1.66,0.6,0.54,0.66,0.42
#1,1.73,0.57h,3,0.66h
#DEFAULT_BOXES = ((0,0,1,1),(0,0,1.2,1),(0,0,1.5,1),(0,0,1.75,1),(0,0,2,1),(0,0,2.25,1),(0,0,2.5,1),(0,0,3,1),
#                 (0.5,0.5,1.5,1.5),(0.5,0.5,1.7,1.5),(0.5,0.5,2,1.5),(0.5,0.5,2.25,1.5),(0.5,0.5,2.5,1.5),(0.5,0.5,2.75,1.5),(0.5,0.5,3,1.5),#(0.5,0.5,3.5,1.5) )


NUM_DEFAULT_BOXES = len(DEFAULT_BOXES)

# Constants (TODO: Keep this updated as we go along)
NUM_CLASSES = 24  
NUM_CHANNELS = 3  # grayscale->1, RGB->3
NUM_PRED_CONF = NUM_DEFAULT_BOXES * NUM_CLASSES  # number of class predictions per feature map cell
NUM_PRED_LOC  = NUM_DEFAULT_BOXES * 4  # number of localization regression predictions per feature map cell

# Bounding box parameters
IOU_THRESH = 0.6  # match ground-truth box to default boxes exceeding this IOU threshold, during data prep
NMS_IOU_THRESH = 0.2  # IOU threshold for non-max suppression

# Negatives-to-positives ratio used to filter training data
NEG_POS_RATIO = 40  # negative:positive = NEG_POS_RATIO:1

# Class confidence threshold to count as detection
CONF_THRESH = 0.95

# Model selection and dependent parameters
MODEL = 'AlexNet'  # AlexNet/VGG16/ResNet50
if MODEL == 'AlexNet':
        #IMG_H, IMG_W = 300, 300
        #FM_SIZES = [[36, 36], [17, 17], [9, 9], [5, 5]]  # feature map sizes for SSD hooks via TensorBoard visualization (HxW)

        IMG_H, IMG_W = 300, 300
        #FM_SIZES = [[31, 48], [15, 23], [8, 12], [4, 6]]
        FM_SIZES = [ [25, 25], [13, 13], [7, 7], [4, 4] ,[2 ,2] ]
else:
         raise NotImplementedError('Model not implemented')

# Model hyper-parameters
OPT = tf.train.AdadeltaOptimizer()
REG_SCALE = 1e-3  # L2 regularization strength
LOC_LOSS_WEIGHT = 0.3  # weight of localization loss: loss = conf_loss + LOC_LOSS_WEIGHT * loc_loss

# Training process
RESUME = False  # resume training from previously saved model?
NUM_EPOCH = 200
BATCH_SIZE = 16  # batch size for training (relatively small)
VALIDATION_SIZE = 0.05  # fraction of total training set to use as validation set
SAVE_MODEL = True  # save trained model to disk?
MODEL_SAVE_PATH = './model.ckpt'  # where to save trained model