import os
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


up = 8
datax = list(range(0, up))
datay = list(map(lambda x: 0.5*x+1.1, range(0, up)))

a = tf.Variable(0.1, dtype=tf.float32)
b = tf.Variable(0.1, dtype=tf.float32)
x = tf.placeholder(tf.float32)
y = a + b*x
# data y
y_ = tf.placeholder(tf.float32)
loss = tf.reduce_sum(tf.square(y-y_))
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(100):
    lst = sess.run([train, a, b], {x: datax, y_: datay})
    print(i, *lst[1:])

