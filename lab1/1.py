import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def getExp(x):
    with tf.Session() as sess:
        try:
            if isinstance(x, int):
                x = float(x)
            return sess.run(tf.exp(x))
        except Exception as e:
            print(repr(e))
            return -1


def getRank(X):
    with tf.Session() as sess:
        try:
            return sess.run(tf.rank(X))
        except Exception as e:
            print(repr(e))
            return -1


def mulMatrix(X, Y):
    with tf.Session() as sess:
        try:
            Z = tf.matmul(X, Y)
            sess.run(Z)
            print(Z.shape)
            sess.run(tf.print(Z))
            return Z
        except Exception as e:
            print(repr(e))


print(getExp(2))
print(getRank([[1, 2, 3], [2, 3, 4], [3, 4, 5]]))
print(mulMatrix([[1, 2], [2, 3], [3, 4]], [[4, 5], [5, 6]]))
