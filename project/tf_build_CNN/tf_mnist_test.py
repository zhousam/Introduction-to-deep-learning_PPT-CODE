import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from network import network_mnist
import matplotlib.pyplot as plt
import numpy as np

c_in = 1
size_input = 28
size_output = 10

model_path = './mnist_model/'

mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)

x = tf.placeholder(tf.float32, [None, size_input*size_input])  #数据
y = tf.placeholder(tf.float32, [None, size_output]) #标签

x_image = tf.reshape(x, [-1, size_input, size_input, c_in]) # 转换为图像的格式

y_pre = network_mnist(x_image, c_in, size_output)  #预测值,预测标签

# compute the accuracy
correct_predictions = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

saver = tf.train.Saver()
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver.restore(sess, model_path)
y_pre_get, acc_test = sess.run([y_pre, accuracy], feed_dict={x: mnist.test.images, y: mnist.test.labels})
print('acc_test: %.4f%%'%(acc_test*100))
print('***********DONE***************')

fig, ax = plt.subplots(nrows=4, ncols=5,
            sharex='all', sharey='all')
ax = ax.flatten()
for i in range(20):
    img = mnist.test.images[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')
    num = np.argmax(mnist.test.labels[i])
    ax[i].text(3, 3, 'T'+str(num))
    num = np.argmax(y_pre_get[i])
    ax[i].text(10, 3, 'P'+str(num))
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

