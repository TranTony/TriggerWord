#------------------------------------------------------------
# models.py
#
# Model for Kaggle Tensorflow speech competition
import tensorflow as tf


def conv2DRnn(batch_data, noutputs, isTraining):

    data  = tf.expand_dims(batch_data, -1)
    #  the first layer: filter num = 64 -> the same number for other architecture., kernel_size should be 3x3 or 5x5, stride should be 1

    conv1 = tf.layers.conv2d(data, filters=64, kernel_size=[7,7], strides=[2,2],
                             padding='SAME', activation=None)
    conv1 = tf.layers.batch_normalization(conv1, training=isTraining, momentum=0.9)
    conv1 = tf.nn.relu(conv1)
    
    conv2 = tf.layers.conv2d(conv1, filters=1, kernel_size=[7,7], strides=[1,1],
                             padding='SAME', activation=None)
    conv2 = tf.layers.batch_normalization(conv2, training=isTraining, momentum=0.9)
    conv2 = tf.nn.relu(conv2)

    conv2 = tf.layers.conv2d(conv2, filters=64, kernel_size=[1,1], strides=[1,1],
                             padding='SAME', activation=None)
    conv2 = tf.layers.batch_normalization(conv2, training=isTraining, momentum=0.9)
    conv2 = tf.nn.relu(conv2)

    conv2 = tf.layers.conv2d(conv2, filters=1, kernel_size=[7,7], strides=[1,1],
                             padding='SAME', activation=None)
    conv2 = tf.layers.batch_normalization(conv2, training=isTraining, momentum=0.9)
    conv2 = tf.nn.relu(conv2)

    conv2 = tf.layers.conv2d(conv2, filters=64, kernel_size=[1,1], strides=[1,1],
                             padding='SAME', activation=None)
    conv2 = tf.layers.batch_normalization(conv2, training=isTraining, momentum=0.9)
    conv2 = tf.nn.relu(conv2)

    conv2 = tf.layers.conv2d(conv2, filters=1, kernel_size=[7,7], strides=[1,1],
                             padding='SAME', activation=None)
    conv2 = tf.layers.batch_normalization(conv2, training=isTraining, momentum=0.9)
    conv2 = tf.nn.relu(conv2)

    conv2 = tf.layers.conv2d(conv2, filters=64, kernel_size=[1,1], strides=[1,1],
                             padding='SAME', activation=None)
    conv2 = tf.layers.batch_normalization(conv2, training=isTraining, momentum=0.9)
    conv2 = tf.nn.relu(conv2)
    #  the first layer: ksize should 2x2

    conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,1,2,1], padding="SAME")

    # should not do drop out to get better training but trade off overfitting.
    conv2 = tf.layers.dropout(conv2, rate=0.8, training=isTraining) 
 
    conv3 = tf.layers.conv2d(conv2, filters=1, kernel_size=[7,7], strides=[1,1],
                             padding='SAME', activation=None)
    conv3 = tf.layers.batch_normalization(conv3, training=isTraining, momentum=0.9)
    conv3 = tf.nn.relu(conv3)

    conv3 = tf.layers.conv2d(conv3, filters=128, kernel_size=[1,1], strides=[1,1],
                             padding='SAME', activation=None)
    conv3 = tf.layers.batch_normalization(conv3, training=isTraining, momentum=0.9)
    conv3 = tf.nn.relu(conv3)

    conv4 = tf.layers.conv2d(conv3, filters=1, kernel_size=[7,7], strides=[1,1],
                             padding='SAME', activation=None)
    conv4 = tf.layers.batch_normalization(conv4, training=isTraining, momentum=0.9)
    conv4 = tf.nn.relu(conv4)

    conv4 = tf.layers.conv2d(conv4, filters=128, kernel_size=[1,1], strides=[1,1],
                             padding='SAME', activation=None)
    conv4 = tf.layers.batch_normalization(conv4, training=isTraining, momentum=0.9)
    conv4 = tf.nn.relu(conv4)



    conv4 = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,1,2,1], padding="SAME")    
    conv4 = tf.layers.dropout(conv4, rate=0.8, training=isTraining)    
     

    conv5 = tf.layers.conv2d(conv4, filters=1, kernel_size=[7,7], strides=[1,1],
                             padding='SAME', activation=None)
    conv5 = tf.layers.batch_normalization(conv5, training=isTraining, momentum=0.9)
    conv5 = tf.nn.relu(conv5)

    conv5 = tf.layers.conv2d(conv5, filters=256, kernel_size=[1,1], strides=[1,1],
                             padding='SAME', activation=None)
    conv5 = tf.layers.batch_normalization(conv5, training=isTraining, momentum=0.9)
    conv5 = tf.nn.relu(conv5)
   
    conv6shape = conv5.shape.as_list()


    squeezed   = tf.reshape(conv5, (-1, conv6shape[1], conv6shape[2]*conv6shape[3]))
    
    X_seqs     = tf.unstack(tf.transpose(squeezed, perm=[1,0,2]))
    basic_cell = tf.contrib.rnn.GRUBlockCellV2(num_units=128)
    output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, X_seqs, dtype=tf.float32)

    fc1  = tf.layers.dense(states, 256, activation=None)
    fc1  = tf.layers.batch_normalization(fc1, training=isTraining, momentum=0.9)
    fc1  = tf.nn.relu(fc1)
    
    logits = tf.layers.dense(fc1, noutputs)
    
    return logits
