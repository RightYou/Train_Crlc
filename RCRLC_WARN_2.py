import tensorflow as tf


# 14W参数，输出两个通道，循环两次，开始和结束的卷积没有正则化

def RCRLC_new(temp_tensor, name, times=1):
    skip_tensor = temp_tensor
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # Conv, 1x1, filters=192 ,+ ReLU
        conv_w1 = tf.get_variable("conv_w1", [1, 1, 32, 192],
                                  initializer=tf.contrib.layers.xavier_initializer())
        conv_b1 = tf.get_variable("conv_b1", [192], initializer=tf.constant_initializer(0))
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, tf.contrib.layers.l2_regularizer(1.)(conv_w1))

        # Conv, 1x1, filters=25
        conv_w2 = tf.get_variable("conv_w2", [1, 1, 192, 25],
                                  initializer=tf.contrib.layers.xavier_initializer())
        conv_b2 = tf.get_variable("conv_b2", [25], initializer=tf.constant_initializer(0))
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, tf.contrib.layers.l2_regularizer(1.)(conv_w2))

        # Conv, 3x3, filters=32
        conv_w3 = tf.get_variable("conv_w3", [3, 3, 25, 32],
                                  initializer=tf.contrib.layers.xavier_initializer())
        conv_b3 = tf.get_variable("conv_b3", [32], initializer=tf.constant_initializer(0))
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, tf.contrib.layers.l2_regularizer(1.)(conv_w3))

    for i in range(times):
        temp_tensor = tf.nn.relu(
            tf.nn.bias_add(tf.nn.conv2d(temp_tensor, conv_w1, strides=[1, 1, 1, 1], padding='SAME'), conv_b1))
        temp_tensor = tf.nn.bias_add(tf.nn.conv2d(temp_tensor, conv_w2, strides=[1, 1, 1, 1], padding='SAME'), conv_b2)
        temp_tensor = tf.nn.bias_add(tf.nn.conv2d(temp_tensor, conv_w3, strides=[1, 1, 1, 1], padding='SAME'), conv_b3)
    # ---------------------------------------------------------------------------------------------------------------------------------
    # skip + out_tensor
    out_tensor = tf.add(skip_tensor, temp_tensor)
    return out_tensor


def crlc_model(input_tensor):
    convId = 0

    conv_00_w = tf.get_variable("conv_%02d_w" % (convId), [3, 3, 1, 32],
                                initializer=tf.contrib.layers.xavier_initializer())
    conv_00_b = tf.get_variable("conv_%02d_b" % (convId), [32], initializer=tf.constant_initializer(0))
    tensor = tf.nn.bias_add(tf.nn.conv2d(input_tensor, conv_00_w, strides=[1, 1, 1, 1], padding='SAME'), conv_00_b)
    convId += 1

    # Residual Block x 8
    for i in range(8):
        tensor = RCRLC_new(tensor, "conv_%02d" % (convId), times=2)
        convId += 1

    conv_w = tf.get_variable("conv_%02d_w" % (convId), [3, 3, 32, 2],
                             initializer=tf.contrib.layers.xavier_initializer())
    conv_b = tf.get_variable("conv_%02d_b" % (convId), [2], initializer=tf.constant_initializer(0))
    tensor = tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1, 1, 1, 1], padding='SAME'), conv_b)

    tensor = tf.add(tensor, input_tensor)

    return tensor
