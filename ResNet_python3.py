
# coding: utf-8

# In[1]:

import numpy as np
import tensorflow as tf
import os
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


# In[14]:

def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
    
    
    regularizer = tf.contrib.layers.l2_regularizer(0.01)
  
    new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                    regularizer=regularizer)
    return new_variables


def output_layer(input_layer, num_labels,is_training=True):
    
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True,
                            initializer=tf.contrib.layers.xavier_initializer())
    fc_b = create_variables(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer())

    fc_h = tf.matmul(input_layer, fc_w) + fc_b
    fc_d = tf.layers.dropout(fc_h, rate=0.25, training=is_training)
    return fc_d


def batch_normalization_layer(input_layer, dimension):
   
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    beta = tf.get_variable('beta', dimension, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', dimension, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

    return bn_layer


def conv_bn_relu_layer(input_layer, filter_shape, stride):
    
    out_channel = filter_shape[-1]
    filter = create_variables(name='conv', shape=filter_shape)

    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    bn_layer = batch_normalization_layer(conv_layer, out_channel)

    output = tf.nn.relu(bn_layer)
    return output


def bn_relu_conv_layer(input_layer, filter_shape, stride):
    
    in_channel = input_layer.get_shape().as_list()[-1]

    bn_layer = batch_normalization_layer(input_layer, in_channel)
    relu_layer = tf.nn.relu(bn_layer)

    filter = create_variables(name='conv', shape=filter_shape)
    conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    return conv_layer



def residual_block(input_layer, output_channel, first_block=False,is_training=True,dropout=0.25):
    '''
    Defines a residual block in ResNet
    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param first_block: if this is the first residual block of the whole network
    :return: 4D tensor.
    '''
    input_channel = input_layer.get_shape().as_list()[-1]

    # When it's time to "shrink" the image size, we use stride = 2
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('conv1_in_block'):
        if first_block:
            filter = create_variables(name='conv', shape=[3, 3, input_channel, output_channel])
            conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
        else:
            conv1 = bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride)

    with tf.variable_scope('conv2_in_block'):
        conv2 = bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1)
        conv2_d = tf.layers.dropout(conv2, rate=dropout, training=is_training)

    
    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers
    if increase_dim is True:
        pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='VALID')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                     input_channel // 2]])
    else:
        padded_input = input_layer

    output = conv2_d + padded_input
    return output


def inference(input_tensor_batch, n, reuse,is_training=True):
    '''
    The main function that defines the ResNet. total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
    :param input_tensor_batch: 4D tensor
    :param n: num_residual_blocks
    :param reuse: To build train graph, reuse=False. To build validation graph and share weights
    with train graph, resue=True
    :return: last layer in the network. Not softmax-ed
    '''

    layers = []
    with tf.variable_scope('conv0', reuse=reuse):
        conv0 = conv_bn_relu_layer(input_tensor_batch, [3, 3, 3, 16], 1)
        layers.append(conv0)

    for i in range(n):
        with tf.variable_scope('conv1_%d' %i, reuse=reuse):
            if i == 0:
                conv1 = residual_block(layers[-1], 16, first_block=True,is_training=is_training)
            else:
                conv1 = residual_block(layers[-1], 16,is_training=is_training)
            
            layers.append(conv1)

    for i in range(n):
        with tf.variable_scope('conv2_%d' %i, reuse=reuse):
            conv2 = residual_block(layers[-1], 32,is_training=is_training)
            
            layers.append(conv2)

    for i in range(n):
        with tf.variable_scope('conv3_%d' %i, reuse=reuse):
            conv3 = residual_block(layers[-1], 64,is_training=is_training)
            layers.append(conv3)
        assert conv3.get_shape().as_list()[1:] == [8, 8, 64]

    with tf.variable_scope('fc', reuse=reuse):
        in_channel = layers[-1].get_shape().as_list()[-1]
        bn_layer = batch_normalization_layer(layers[-1], in_channel)
        relu_layer = tf.nn.relu(bn_layer)
        global_pool = tf.reduce_mean(relu_layer, [1, 2])

        assert global_pool.get_shape().as_list()[-1:] == [64]
        output = output_layer(global_pool, 10,is_training=is_training)
        layers.append(output)

    return layers[-1]


# In[6]:

#labels = tf.cast(labels, tf.int64)
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


# In[16]:

logit_vali = inference(X,4,reuse=True,is_training=False)
correct_prediction_vali = tf.equal(tf.cast(tf.argmax(logit_vali,1),tf.int32), Y)
accuracy_vali = tf.reduce_mean(tf.cast(correct_prediction_vali, tf.float32))


# In[17]:

test_acc=sess.run(accuracy_vali,feed_dict={X: x_test,Y: y_test})
print(test_acc)


# In[12]:

np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])


# In[12]:

idx=2000
test_acc=sess.run(accuracy,feed_dict={X: x_test[idx:idx+BATCH_SIZE],Y: y_test[idx:idx+BATCH_SIZE]})
print(test_acc)


# In[20]:

reg_loss = tf.reduce_sum(tf.losses.get_regularization_losses())
loss = tf.reduce_mean(cross_entropy, name='cross_entropy')+0.01*reg_loss


# In[16]:

tf.reduce_sum(reg_loss)


# In[21]:

sess.run(reg_loss,feed_dict={X: x_train[idx:idx+BATCH_SIZE],Y: y_train[idx:idx+BATCH_SIZE]})


# In[ ]:



