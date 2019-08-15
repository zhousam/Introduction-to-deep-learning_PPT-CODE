import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from network import network_mnist
import os, shutil
import time

c_in = 1
size_input = 28
size_output = 10
epoch_max = 50
batch_size = 512
step_max = int(55000/batch_size)
acc_stop = 0.99
model_path = './mnist_model/'

shutil.rmtree(model_path, ignore_errors=True)
os.mkdir(model_path)

mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)

x = tf.placeholder(tf.float32, [None, size_input*size_input])  # 数据
y = tf.placeholder(tf.float32, [None, size_output]) # 标签

x_image = tf.reshape(x, [-1, size_input, size_input, c_in]) # 转换为图像的格式

y_pre = network_mnist(x_image, c_in, size_output)  # 预测值,预测标签
cross_entropy = tf.reduce_mean(tf.reduce_sum(-y * tf.log(y_pre), reduction_indices=[1]))
train = tf.train.GradientDescentOptimizer(0.08).minimize(cross_entropy)

# compute the accuracy
correct_predictions = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

saver = tf.train.Saver()
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
acc_max = -1
str_time = time.strftime('%H:%M:%S', time.localtime())
print('%s: begin'%(str_time))
for iepoch in range(epoch_max):
    is_save = 0
    acc_train_all = []
    for step in range(step_max):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, acc = sess.run([train, accuracy], feed_dict={x:batch_x, y:batch_y})
        acc_train_all.append(acc)
    acc_train = sum(acc_train_all)/len(acc_train_all)
    acc_val = sess.run(accuracy, feed_dict={x: mnist.validation.images, y: mnist.validation.labels})
    if acc_max < acc_val:
        saver.save(sess, model_path)
        is_save = 1
        acc_max = acc_val
    str_time = time.strftime('%H:%M:%S', time.localtime())
    print('%s epoch:%02d is_save:%d acc_max:%.4f%% acc_val:%.4f%% acc_train:%.4f%%'%(str_time, iepoch,
                                            is_save, acc_max*100, acc_val*100, acc_train*100))
    if acc_max >= acc_stop:
        print('End training early...')
        break

acc_test = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
print('acc_test: %.4f%%'%(acc_test*100))
print('***********DONE***************')


