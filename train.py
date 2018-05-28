
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os

import cv2


import pickle
from PIL import Image
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt
from setting import *
from inputs import *
from functions import *

if does_use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu_num)
def SSDHook(feature_map, hook_id):
        """
        Takes input feature map, output the predictions tensor
        hook_id is for variable_scope unqie string ID
        """
        with tf.variable_scope('ssd_hook_' + hook_id):
            # Note we have linear activation (i.e. no activation function)
            net_conf = slim.conv2d(feature_map, NUM_PRED_CONF, [3, 3], activation_fn=None, scope='conv_conf')
            net_conf = tf.contrib.layers.flatten(net_conf)

            net_loc = slim.conv2d(feature_map, NUM_PRED_LOC, [3, 3], activation_fn=None, scope='conv_loc')
            net_loc = tf.contrib.layers.flatten(net_loc)

        return net_conf, net_loc
def AlexNet():
        
        # Image batch tensor and dropout keep prob placeholders
        x = tf.placeholder(tf.float32, [None, IMG_H, IMG_W, NUM_CHANNELS], name='x')
        is_training = tf.placeholder(tf.bool, name='is_training')

        # Classification and localization predictions
        preds_conf = []  # conf -> classification b/c confidence loss -> classification loss
        preds_loc = []

        # Use batch normalization for all convolution layers
        # FIXME: Not sure why setting is_training is not working well
        #with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params={'is_training': is_training}):
        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params={'is_training': True},\
                weights_regularizer=slim.l2_regularizer(scale=REG_SCALE)):
            net = slim.conv2d(x, 64, [11, 11], 3, padding='VALID', scope='conv1')
            net = slim.max_pool2d(net, [3, 3], 2,padding='SAME',scope='pool1')
            net = slim.conv2d(net, 192, [5, 5], scope='conv2')

            net_conf, net_loc = SSDHook(net, 'conv2')
            preds_conf.append(net_conf)
            preds_loc.append(net_loc)

            #net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
            net = slim.max_pool2d(net, [3, 3], 2,padding='SAME', scope='pool2')
            net = slim.conv2d(net, 384, [3, 3], scope='conv3')
            net = slim.conv2d(net, 384, [3, 3], scope='conv4')
            net = slim.conv2d(net, 256, [3, 3], scope='conv5')

            # The following layers added for SSD
            net = slim.conv2d(net, 1024, [3, 3], scope='conv6')
            net = slim.conv2d(net, 1024, [1, 1], scope='conv7')

            net_conf, net_loc = SSDHook(net, 'conv7')
            preds_conf.append(net_conf)
            preds_loc.append(net_loc)

            net = slim.conv2d(net, 256, [1, 1], scope='conv8')
            net = slim.conv2d(net, 512, [3, 3], 2, scope='conv8_2')

            net_conf, net_loc = SSDHook(net, 'conv8_2')
            preds_conf.append(net_conf)
            preds_loc.append(net_loc)

            net = slim.conv2d(net, 128, [1, 1], scope='conv9')
            net = slim.conv2d(net, 256, [3, 3], 2, scope='conv9_2')

            net_conf, net_loc = SSDHook(net, 'conv9_2')
            preds_conf.append(net_conf)
            preds_loc.append(net_loc)

        # Concatenate all preds together into 1 vector, for both classification and localization predictions
        final_pred_conf = tf.concat(preds_conf,1)
        final_pred_loc = tf.concat(preds_loc,1)

        # Return a dictionary of {tensor_name: tensor_reference}
        ret_dict = {
            'x': x,
            'y_pred_conf': final_pred_conf,
            'y_pred_loc': final_pred_loc,
            'is_training': is_training,
        }
        return ret_dict
def SSDModel():
        """
        Wrapper around the model and model helper
        Returns dict of relevant tensor references
        """
       
        model = AlexNet()
        model_helper = ModelHelper(model['y_pred_conf'], model['y_pred_loc'])

        ssd_model = {}
        for k in model.keys():
            ssd_model[k] = model[k]
        for k in model_helper.keys():
            ssd_model[k] = model_helper[k]

        return ssd_model
def ModelHelper(y_pred_conf, y_pred_loc):
        """
        Define loss function, optimizer, predictions, and accuracy metric
        Loss includes confidence loss and localization loss
        conf_loss_mask is created at batch generation time, to mask the confidence losses
        It has 1 at locations w/ positives, and 1 at select negative locations
        such that negative-to-positive ratio of NEG_POS_RATIO is satisfied
        Arguments:
            * y_pred_conf: Class predictions from model,
                a tensor of shape [batch_size, num_feature_map_cells * num_defaul_boxes * num_classes]
            * y_pred_loc: Localization predictions from model,
                a tensor of shape [batch_size, num_feature_map_cells * num_defaul_boxes * 4]
        Returns relevant tensor references
        """
        num_total_preds = 0
        for fm_size in FM_SIZES:
            num_total_preds += fm_size[0] * fm_size[1] * NUM_DEFAULT_BOXES
        num_total_preds_conf = num_total_preds * NUM_CLASSES
        num_total_preds_loc  = num_total_preds * 4

        # Input tensors
        y_true_conf = tf.placeholder(tf.int32, [None, num_total_preds], name='y_true_conf')  # classification ground-truth labels
        y_true_loc  = tf.placeholder(tf.float32, [None, num_total_preds_loc], name='y_true_loc')  # localization ground-truth labels
        conf_loss_mask = tf.placeholder(tf.float32, [None, num_total_preds], name='conf_loss_mask')  # 1 mask "bit" per def. box

        # Confidence loss
        logits = tf.reshape(y_pred_conf, [-1, num_total_preds, NUM_CLASSES])
        conf_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=y_true_conf)
        #conf_loss = tf.metrics.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)
        
        conf_loss = conf_loss_mask * conf_loss  # "zero-out" the loss for don't-care negatives
        conf_loss = tf.reduce_sum(conf_loss)

        # Localization loss (smooth L1 loss)
        # loc_loss_mask is analagous to conf_loss_mask, except 4 times the size
        print(y_true_loc)
        print(y_pred_loc)
        diff = y_true_loc - y_pred_loc

        loc_loss_l2 = 0.5 * (diff**2.0)
        loc_loss_l1 = tf.abs(diff) - 0.5
        smooth_l1_condition = tf.less(tf.abs(diff), 1.0)
        loc_loss = tf.where(smooth_l1_condition, loc_loss_l2, loc_loss_l1)

        loc_loss_mask = tf.minimum(y_true_conf, 1)  # have non-zero localization loss only where we have matching ground-truth box
        loc_loss_mask = tf.to_float(loc_loss_mask)
        loc_loss_mask = tf.stack([loc_loss_mask] * 4, axis=2)  # [0, 1, 1] -> [[[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]], ...]
        loc_loss_mask = tf.reshape(loc_loss_mask, [-1, num_total_preds_loc])  # removing the inner-most dimension of above
        loc_loss = loc_loss_mask * loc_loss
        loc_loss = tf.reduce_sum(loc_loss)

        # Weighted average of confidence loss and localization loss
        # Also add regularization loss
        #loss = conf_loss + LOC_LOSS_WEIGHT * loc_loss + tf.reduce_sum(slim.losses.get_regularization_losses())
        loss = conf_loss + LOC_LOSS_WEIGHT * loc_loss + 0.001*tf.reduce_sum(tf.losses.get_regularization_losses())
        optimizer = OPT.minimize(loss)

        #reported_loss = loss #tf.reduce_sum(loss, 1)  # DEBUG

        # Class probabilities and predictions
        probs_all = tf.nn.softmax(logits)
        probs, preds_conf = tf.nn.top_k(probs_all)  # take top-1 probability, and the index is the predicted class
        probs = tf.reshape(probs, [-1, num_total_preds])
        preds_conf = tf.reshape(preds_conf, [-1, num_total_preds])

        # Return a dictionary of {tensor_name: tensor_reference}
        ret_dict = {
            'y_true_conf': y_true_conf,
            'y_true_loc': y_true_loc,
            'conf_loss_mask': conf_loss_mask,
            'optimizer': optimizer,
            'conf_loss': conf_loss,
            'loc_loss': loc_loss,
            'loss': loss,
            'probs': probs,
            'probs_all': probs_all,
            'preds_conf': preds_conf,
            'preds_loc': y_pred_loc,
        }
        return ret_dict
    
    
os.chdir(DATA_DIR)
train=np.load('./data_prep_sd.npy')
train=train[()]
X_train = []
y_train_conf = []
y_train_loc = []
k=0
for img_name in train.keys():
        img_file=cv2.imread(os.path.join(DATA_DIR,'JPEGImages',img_name))
        X_train.append(img_file)
        y_train_conf.append(train[img_name]['y_true_conf'])
        y_train_loc.append(train[img_name]['y_true_loc'])
X_train = np.array(X_train)
y_train_conf = np.array(y_train_conf)
y_train_loc = np.array(y_train_loc)


sess=tf.Session()
model=SSDModel()
x=model['x']
y_true_conf=model['y_true_conf']
y_true_loc=model['y_true_loc']
conf_loss_mask=model['conf_loss_mask']
is_training=model['is_training']
conf_loss=model['conf_loss']
loc_loss=model['loc_loss']
reported_loss=model['loss']
optimizer = model['optimizer']

probs_all=model['probs_all']
preds_conf = model['preds_conf']
preds_loc = model['preds_loc']
probs = model['probs']
saver=tf.train.Saver()

if is_restore:
    saver.restore(sess,SAVED_MODEL_PATH)
else:
    sess.run(tf.global_variables_initializer())
BATCH_SIZE=16
optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(model['loss'])
if not DOES_SKIP_TRAIN:
    for epoch in range(150 ):
        train_gen=next_batch(X_train,y_train_conf,y_train_loc,BATCH_SIZE)
        num_batches_train = int(math.ceil(X_train.shape[0] / BATCH_SIZE))
        losses=[]

        for i in range(num_batches_train):
            images, y_true_conf_gen, y_true_loc_gen, conf_loss_mask_gen = next(train_gen)
            _, loss = sess.run([optimizer, reported_loss], feed_dict={x: images,y_true_conf: y_true_conf_gen,
                            y_true_loc: y_true_loc_gen,conf_loss_mask: conf_loss_mask_gen,
                            is_training: True
                        })
            losses.append(loss)
            #print(loss)
        print("new epoch")
        losses=np.array(losses)
        print(np.mean(losses))
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(model['loss'])
    for epoch in range(200):
        train_gen=next_batch(X_train,y_train_conf,y_train_loc,BATCH_SIZE)
        num_batches_train = int(math.ceil(X_train.shape[0] / BATCH_SIZE))
        losses=[]

        for i in range(num_batches_train):
            images, y_true_conf_gen, y_true_loc_gen, conf_loss_mask_gen = next(train_gen)
            _, loss = sess.run([optimizer, reported_loss], feed_dict={x: images,y_true_conf: y_true_conf_gen,
                            y_true_loc: y_true_loc_gen,conf_loss_mask: conf_loss_mask_gen,
                            is_training: True
                        })
            losses.append(loss)
            #print(loss)
        print("new epoch")
        losses=np.array(losses)
        print(np.mean(losses))
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(model['loss'])
    for epoch in range(100 ):
        train_gen=next_batch(X_train,y_train_conf,y_train_loc,BATCH_SIZE)
        num_batches_train = int(math.ceil(X_train.shape[0] / BATCH_SIZE))
        losses=[]

        for i in range(num_batches_train):
            images, y_true_conf_gen, y_true_loc_gen, conf_loss_mask_gen = next(train_gen)
            _, loss = sess.run([optimizer, reported_loss], feed_dict={x: images,y_true_conf: y_true_conf_gen,
                            y_true_loc: y_true_loc_gen,conf_loss_mask: conf_loss_mask_gen,
                            is_training: True
                        })
            losses.append(loss)
            #print(loss)
        print("new epoch")
        losses=np.array(losses)
        print(np.mean(losses))
model_path=SAVED_MODEL_PATH
saver.save(sess,model_path)
