import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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
