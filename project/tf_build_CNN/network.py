import tensorflow as tf

# 创建一个神经网络层
def add_layer(input, in_size, out_size, activation_function=None):
    """
    :param input: 数据输入
    :param in_size: 输入大小
    :param out_size: 输出大小
    :param activation_function: 激活函数（默认没有）
    :return:output：数据输出
    """
    Weight = tf.Variable(tf.random_normal([in_size, out_size], stddev=0.02))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    W_mul_x_plus_b = tf.matmul(input, Weight) + biases
    # 根据是否有激活函数
    if activation_function == None:
        output = W_mul_x_plus_b
    else:
        output = activation_function(W_mul_x_plus_b)
    return output

def conv2d(x, c_in, c_out, k_size, stride, padding='VALID', activation_function=tf.nn.relu):
    Weight = tf.Variable(tf.random_normal([k_size, k_size, c_in, c_out], stddev=0.02))
    biases = tf.Variable(tf.zeros([c_out]) + 0.1)
    conv_1 = tf.nn.conv2d(x, Weight, strides=[1, stride, stride, 1], padding=padding)
    conv_1_b = tf.nn.bias_add(conv_1, biases)
    if activation_function == None:
        output = conv_1_b
    else:
        output = activation_function(conv_1_b)
    return output

def network_mnist(input, c_in, c_out):
    conv_1 = conv2d(input, c_in, 32, 3, 1, padding='SAME', activation_function=tf.nn.relu)
    pool_1 = tf.nn.max_pool(conv_1, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
    conv_2 = conv2d(pool_1, 32, 64, 3, 1, padding='SAME', activation_function=tf.nn.relu)
    pool_2 = tf.nn.max_pool(conv_2, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')

    pool_2_flat = tf.reshape(pool_2, [-1, 7 * 7 * 64])
    fc_3 = add_layer(pool_2_flat, 7*7*64, 1024, activation_function=tf.nn.relu)
    fc_4 = add_layer(fc_3, 1024, 512, activation_function=tf.nn.relu)
    prediction = add_layer(fc_4, 512, c_out, activation_function=tf.nn.softmax)
    return prediction


