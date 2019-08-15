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

def network_mnist(input, in_size, out_size, keep_prob):
    hidden_layer1 = add_layer(input, in_size, 1024, activation_function=tf.nn.relu)
    hidden_layer1 = tf.nn.dropout(hidden_layer1, keep_prob)
    hidden_layer2 = add_layer(hidden_layer1, 1024, 512, activation_function=tf.nn.relu)
    hidden_layer2 = tf.nn.dropout(hidden_layer2, keep_prob)
    hidden_layer3 = add_layer(hidden_layer2, 512, 256, activation_function=tf.nn.relu)
    hidden_layer3 = tf.nn.dropout(hidden_layer3, keep_prob)
    prediction = add_layer(hidden_layer3, 256, out_size, activation_function=tf.nn.softmax)
    return prediction

