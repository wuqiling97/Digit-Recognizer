import os, sys
import argparse
import numpy as np
import tensorflow as tf
from util import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def decreasing(lst, length):
    if len(lst) < length:
        return False
    else:
        return all(x>y for x, y in zip(lst, lst[1:]))


def main(_):
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(
            x, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='SAME')

    def model(x, keep_prob):
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        # first conv
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
        # second conv
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
        # full connection
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        # dropout
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        # readout layer
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        saver = tf.train.Saver(
            [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2], max_to_keep=100)
        return y_conv, saver


    # Import data  kg == kaggle data
    kg = DataCollection('data', 4000)

    keep_prob = tf.placeholder(tf.float32)
    x = tf.placeholder(tf.float32, [None, 784])
    y_conv, saver = model(x, keep_prob)
    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    prediction = tf.argmax(y_conv, 1)
    correct_prediction = tf.equal(prediction, tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    accuracy_lst = []
    istrain = False
    savepath = 'save_CNN'

    if istrain:
        print('training begin at {}'.format(current_time()))
        saver.restore(sess, savepath+'/arg-34000')
        for i in range(1+34000, 1+60000):
            batch = kg.train.next_batch(50)
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            if i%100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print("step %d, training accuracy %g" % (i, train_accuracy))
                if i%1000 == 0:
                    # save variables
                    path = saver.save(sess, savepath+'/arg', global_step=i)
                    vali_accuracy = accuracy.eval(
                        feed_dict={x: kg.validation.images, y_: kg.validation.labels, keep_prob: 1.0}
                    )
                    print('validation accuracy {}'.format(vali_accuracy))
                    accuracy_lst.append((vali_accuracy, path))
                    # if decreasing([i[0] for i in accuracy_lst[-4:]], 4):
                    #     break
                if i%10000 == 0 and i>2e4:
                    tmp = input('continue training?(y/n)\n')
                    if tmp == 'n':
                        break
    else:
        pass
        # for fname in os.listdir('save_CNN/'):
        #     if fname.endswith('.meta'):
        #         path = savepath+'/'+fname.replace('.meta', '')
        #         saver.restore(sess, path)
        #         vali_accuracy = accuracy.eval(
        #             feed_dict={x: kg.validation.images, y_: kg.validation.labels, keep_prob: 1.0}
        #         )
        #         accuracy_lst.append((vali_accuracy, path))
        # print(accuracy_lst)

    # determine the max-accuracy state
    # arr = np.array([i[0] for i in accuracy_lst])
    # index = np.argmax(arr)
    # print(accuracy_lst[index])
    # saver.restore(sess, accuracy_lst[index][1])
    saver.restore(sess, savepath+'/arg-84000')

    # testing
    print('begin testing at {}'.format(current_time()))
    res = np.array([], dtype=np.int)
    for batch in kg.test.testbatches(1000):
        array = prediction.eval(feed_dict={x: batch, keep_prob: 1.0})
        # if len(res)==0:
        #     print(type(array), array)
        res = np.concatenate((res, array))
    print('begin writing result at {}'.format(current_time()))
    write_result(res, 'result.csv')


if __name__ == '__main__':
    print('start at {}'.format(current_time()))
    tf.app.run(main=main, argv=[sys.argv[0]])
