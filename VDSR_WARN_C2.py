import tensorflow as tf


def resblock(temp_tensor, convId):
    conv_secondID = 0
    skip_tensor = temp_tensor

    conv_w = tf.get_variable("conv_%02d_%02d_w" % (convId, conv_secondID), [1, 1, 12, 32],
                             initializer=tf.contrib.layers.xavier_initializer())
    conv_b = tf.get_variable("conv_%02d_%02d_b" % (convId, conv_secondID), [32], initializer=tf.constant_initializer(0))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, tf.contrib.layers.l2_regularizer(1.)(conv_w))
    out_tensor = tf.nn.relu(
        tf.nn.bias_add(tf.nn.conv2d(temp_tensor, conv_w, strides=[1, 1, 1, 1], padding='SAME'), conv_b))
    conv_secondID += 1

    conv_w = tf.get_variable("conv_%02d_%02d_w" % (convId, conv_secondID), [1, 1, 32, 8],
                             initializer=tf.contrib.layers.xavier_initializer())
    conv_b = tf.get_variable("conv_%02d_%02d_b" % (convId, conv_secondID), [8], initializer=tf.constant_initializer(0))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, tf.contrib.layers.l2_regularizer(1.)(conv_w))
    out_tensor = tf.nn.bias_add(tf.nn.conv2d(out_tensor, conv_w, strides=[1, 1, 1, 1], padding='SAME'), conv_b)
    conv_secondID += 1

    conv_w = tf.get_variable("conv_%02d_%02d_w" % (convId, conv_secondID), [3, 3, 8, 12],
                             initializer=tf.contrib.layers.xavier_initializer())
    conv_b = tf.get_variable("conv_%02d_%02d_b" % (convId, conv_secondID), [12], initializer=tf.constant_initializer(0))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, tf.contrib.layers.l2_regularizer(1.)(conv_w))
    out_tensor = tf.nn.bias_add(tf.nn.conv2d(out_tensor, conv_w, strides=[1, 1, 1, 1], padding='SAME'), conv_b)
    conv_secondID += 1

    # skip + out_tensor
    out_tensor = tf.add(skip_tensor, out_tensor)
    return out_tensor


def model(input_tensor):
    convId = 0
    conv_00_w = tf.get_variable("conv_%02d_w" % (convId), [3, 3, 1, 12],
                                initializer=tf.contrib.layers.xavier_initializer())
    conv_00_b = tf.get_variable("conv_%02d_b" % (convId), [12], initializer=tf.constant_initializer(0))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, tf.contrib.layers.l2_regularizer(1.)(conv_00_w))
    tensor = tf.nn.bias_add(tf.nn.conv2d(input_tensor, conv_00_w, strides=[1, 1, 1, 1], padding='SAME'), conv_00_b)
    convId += 1

    # Residual Block x 3
    for i in range(3):
        tensor = resblock(tensor, convId)
        convId += 1

    conv_w = tf.get_variable("conv_%02d_w" % (convId), [3, 3, 12, 2],
                             initializer=tf.contrib.layers.xavier_initializer())
    conv_b = tf.get_variable("conv_%02d_b" % (convId), [2], initializer=tf.constant_initializer(0))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, tf.contrib.layers.l2_regularizer(1.)(conv_w))
    tensor = tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1, 1, 1, 1], padding='SAME'), conv_b)

    tensor = tf.add(tensor, input_tensor)

    return tensor
