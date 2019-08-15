# 如果要传入值，用tensorflow的占位符，暂时存储变量，
#  以这种形式feed数据：sess.run(***, feed_dict={input: **})

import tensorflow as tf
#在 Tensorflow 中需要定义 placeholder 的 type ，一般为 float32 形式
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
# mul = multiply 是将input1和input2 做乘法运算，并输出为 output
ouput = tf.multiply(input1, input2)
with tf.Session() as sess:
    print(sess.run(ouput, feed_dict={input1: [7., 10], input2: [2.]}))
