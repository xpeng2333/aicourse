import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
'''
with tf.Session() as sess:
    with tf.device("/GPU:0"):
        matrix1 = tf.constant([[4., 3.]])
        matrix2 = tf.constant([[2.], [2.]])
        product = tf.matmul(matrix1, matrix2)
A = tf.Variable(tf.constant(0.0), dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    a = tf.add(3, 4)
    print(a)
    sess.run(a)
    print(a.get_shape())

'''


def getExp(x):
    try:
        return tf.exp(float(x))
    except Exception as e:
        print(repr(e))
        return -1


def getRank(X):
    try:
        return tf.rank(X)
    except Exception as e:
        print(repr(e))
        return -1


def mulMatrix(X, Y):
    try:
        Z = tf.matmul(X, Y)
        print(Z.get_shape())
        return Z
    except Exception as e:
        print(repr(e))


with tf.Session() as sess:
    sess.run(tf.print(getExp(2)))
    sess.run(tf.print(getRank([[1, 2, 3], [2, 3, 4], [3, 4, 5]])))
    sess.run(tf.print(mulMatrix([[1, 2], [2, 3], [3, 4]], [[4, 5], [5, 6]])))
