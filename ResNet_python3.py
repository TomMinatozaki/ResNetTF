
# coding: utf-8

# In[1]:

import numpy as np
import tensorflow as tf
import os
from ResNetHelper import *
BN_EPSILON = 0.001


# In[2]:

os.chdir('D:\Data\CIFAR')
x_train = np.load('./x_train.npy')
y_train = np.load('./y_train.npy')
x_test = np.load('./x_test.npy')
y_test = np.load('./y_test.npy')
BATCH_SIZE=128


# In[3]:

X=tf.placeholder(dtype=tf.float32,shape=[None,32,32,3])
Y=tf.placeholder(dtype=tf.int32,shape=[None])


# In[4]:


y_train = np.reshape(y_train,(-1))
y_test=np.reshape(y_test,(-1))
x_train = x_train/256
x_test = x_test/256



logit = inference(X,4,reuse=False)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit,labels=Y)
reg_loss = tf.reduce_sum(tf.losses.get_regularization_losses())
loss = tf.reduce_mean(cross_entropy, name='cross_entropy')+0.1*reg_loss
correct_prediction = tf.equal(tf.cast(tf.argmax(logit,1),tf.int32), Y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))





# In[7]:

sess=tf.Session()
sess.run(tf.global_variables_initializer())


# In[12]:

optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(loss)
idx=0
for epoch in range(10000):
    sess.run(optimizer,feed_dict={X: x_train[idx:idx+BATCH_SIZE],Y: y_train[idx:idx+BATCH_SIZE]})
    idx+=BATCH_SIZE
    if idx >= 50000-BATCH_SIZE:
        idx -=50000-BATCH_SIZE-2
    if epoch % 1000 == 0:
        train_acc=sess.run(accuracy,feed_dict={X: x_train[idx:idx+BATCH_SIZE],Y: y_train[idx:idx+BATCH_SIZE]})        
        test_acc=sess.run(accuracy,feed_dict={X: x_test,Y: y_test})
        print('accuracy:')
        print(train_acc)
        print(test_acc)
    if epoch % 500 ==0 :   
        
        curr_loss=sess.run(loss,feed_dict={X: x_train[idx:idx+BATCH_SIZE],Y: y_train[idx:idx+BATCH_SIZE]})
        
        print('loss:')
        print(curr_loss)


np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])


logit_vali = inference(X,4,reuse=True,is_training=False)
correct_prediction_vali = tf.equal(tf.cast(tf.argmax(logit_vali,1),tf.int32), Y)
accuracy_vali = tf.reduce_mean(tf.cast(correct_prediction_vali, tf.float32))
train_acc = sess.run(accuracy_vali,feed_dict={X: x_train[idx:idx+BATCH_SIZE],Y: y_train[idx:idx+BATCH_SIZE]})
test_acc=sess.run(accuracy_vali,feed_dict={X: x_test,Y: y_test})
print(train_acc)
print(test_acc)
