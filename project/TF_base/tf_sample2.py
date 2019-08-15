import tensorflow as tf
# 定义一个tensorflow的变量：
state = tf.Variable(0, name='counter')
# 定义常量
one = tf.constant(1)
# 定义加法步骤 (注: 此步并没有直接计算)
new_value = tf.add(state, one)
# 将 State 更新成 new_value
update = tf.assign(state, new_value)
# 变量Variable需要初始化并激活，并且打印的话只能通过sess.run()：
init = tf.global_variables_initializer()
# 使用 Session 计算
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))