import random
import numpy as np
from PIL import Image
from captcha.image import ImageCaptcha
import tensorflow as tf
from datetime import datetime
import os

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
LOW_CASE = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
    'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]
UP_CASE = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]
# CAPTCHA_LIST = NUMBER + LOW_CASE + UP_CASE
CAPTCHA_LIST = NUMBER
CAPTCHA_LEN = 4
CAPTCHA_HEIGHT = 60
CAPTCHA_WIDTH = 160


def random_captcha_text(char_set=CAPTCHA_LIST, captcha_size=CAPTCHA_LEN):
    '''
    随机生成验证码文本
    :param char_set:
    :param captcha_size:
    :return:
    '''
    captcha_text = [random.choice(char_set) for _ in range(captcha_size)]
    return ''.join(captcha_text)


def gen_captcha_text_and_image(width=CAPTCHA_WIDTH,
                               height=CAPTCHA_HEIGHT,
                               save=None):
    '''
    生成随机验证码
    :param width:
    :param height:
    :param save:
    :return: np数组
    '''
    image = ImageCaptcha(width=width, height=height)
    # 验证码文本
    captcha_text = random_captcha_text()
    captcha = image.generate(captcha_text)
    # 保存
    if save:
        image.write(captcha_text, captcha_text + '.jpg')
    captcha_image = Image.open(captcha)
    # 转化为np数组
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image


def wrap_gen_captcha_text_and_image(shape=(60, 160, 3)):
    '''
    返回特定shape图片
    :param shape:
    :return:
    '''
    while True:
        t, im = gen_captcha_text_and_image()
        if im.shape == shape:
            return t, im


def convert2gray(img):
    '''
    图片转为黑白，3维转1维
    :param img:
    :return:
    '''
    if len(img.shape) > 2:
        img = np.mean(img, -1)
    return img


def text2vec(text, captcha_len=CAPTCHA_LEN, captcha_list=CAPTCHA_LIST):
    '''
    验证码文本转为向量
    :param text:
    :param captcha_len:
    :param captcha_list:
    :return:
    '''
    text_len = len(text)
    if text_len > captcha_len:
        raise ValueError('验证码最长4个字符')
    vector = np.zeros(captcha_len * len(captcha_list))
    for i in range(text_len):
        vector[captcha_list.index(text[i]) + i * len(captcha_list)] = 1
    return vector


def next_batch(batch_count=60, width=CAPTCHA_WIDTH, height=CAPTCHA_HEIGHT):
    '''
    获取训练图片组
    :param batch_count:
    :param width:
    :param height:
    :return:
    '''
    batch_x = np.zeros([batch_count, width * height])
    batch_y = np.zeros([batch_count, CAPTCHA_LEN * len(CAPTCHA_LIST)])
    for i in range(batch_count):
        text, image = wrap_gen_captcha_text_and_image()
        image = convert2gray(image)
        # 将图片数组一维化 同时将文本也对应在两个二维组的同一行
        batch_x[i, :] = image.flatten() / 255
        batch_y[i, :] = text2vec(text)
    # 返回该训练批次
    return batch_x, batch_y


def weight_variable(shape, w_alpha=0.01):
    '''
    增加噪音，随机生成权重
    :param shape:
    :param w_alpha:
    :return:
    '''
    initial = w_alpha * tf.random_normal(shape)
    return tf.Variable(initial)


def bias_variable(shape, b_alpha=0.1):
    '''
    增加噪音，随机生成偏置项
    :param shape:
    :param b_alpha:
    :return:
    '''
    initial = b_alpha * tf.random_normal(shape)
    return tf.Variable(initial)


def conv2d(x, w):
    '''
    局部变量线性组合，步长为1，模式‘SAME’代表卷积后图片尺寸不变，即零边距
    :param x:
    :param w:
    :return:
    '''
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    '''
    max pooling,取出区域内最大值为代表特征， 2x2pool，图片尺寸变为1/2
    :param x:
    :return:
    '''
    return tf.nn.max_pool(x,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')


def cnn_graph(x,
              keep_prob,
              size,
              captcha_list=CAPTCHA_LIST,
              captcha_len=CAPTCHA_LEN):
    '''
    三层卷积神经网络计算图
    :param x:
    :param keep_prob:
    :param size:
    :param captcha_list:
    :param captcha_len:
    :return:
    '''
    # 图片reshape为4维向量
    image_height, image_width = size
    x_image = tf.reshape(x, shape=[-1, image_height, image_width, 1])

    # layer 1
    # filter定义为3x3x1， 输出32个特征, 即32个filter
    w_conv1 = weight_variable([3, 3, 1, 32])
    b_conv1 = bias_variable([32])
    # rulu激活函数
    h_conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(x_image, w_conv1), b_conv1))
    # 池化
    h_pool1 = max_pool_2x2(h_conv1)
    # dropout防止过拟合
    h_drop1 = tf.nn.dropout(h_pool1, keep_prob)

    # layer 2
    w_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2d(h_drop1, w_conv2), b_conv2))
    h_pool2 = max_pool_2x2(h_conv2)
    h_drop2 = tf.nn.dropout(h_pool2, keep_prob)

    # layer 3
    w_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(tf.nn.bias_add(conv2d(h_drop2, w_conv3), b_conv3))
    h_pool3 = max_pool_2x2(h_conv3)
    h_drop3 = tf.nn.dropout(h_pool3, keep_prob)

    # full connect layer
    image_height = int(h_drop3.shape[1])
    image_width = int(h_drop3.shape[2])
    w_fc = weight_variable([image_height * image_width * 64, 1024])
    b_fc = bias_variable([1024])
    h_drop3_re = tf.reshape(h_drop3, [-1, image_height * image_width * 64])
    h_fc = tf.nn.relu(tf.add(tf.matmul(h_drop3_re, w_fc), b_fc))
    h_drop_fc = tf.nn.dropout(h_fc, keep_prob)

    # out layer
    w_out = weight_variable([1024, len(captcha_list) * captcha_len])
    b_out = bias_variable([len(captcha_list) * captcha_len])
    y_conv = tf.add(tf.matmul(h_drop_fc, w_out), b_out)
    return y_conv


def optimize_graph(y, y_conv):
    '''
    优化计算图
    :param y:
    :param y_conv:
    :return:
    '''
    # 交叉熵计算loss 注意logits输入是在函数内部进行sigmod操作
    # sigmod_cross适用于每个类别相互独立但不互斥，如图中可以有字母和数字
    # softmax_cross适用于每个类别独立且排斥的情况，如数字和字母不可以同时出现
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=y_conv, labels=y))
    # 最小化loss优化
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    return optimizer


def accuracy_graph(y, y_conv, width=len(CAPTCHA_LIST), height=CAPTCHA_LEN):
    '''
    偏差计算图
    :param y:
    :param y_conv:
    :param width:
    :param height:
    :return:
    '''
    # 这里区分了大小写 实际上验证码一般不区分大小写
    # 预测值
    predict = tf.reshape(y_conv, [-1, height, width])
    max_predict_idx = tf.argmax(predict, 2)
    # 标签
    label = tf.reshape(y, [-1, height, width])
    max_label_idx = tf.argmax(label, 2)
    correct_p = tf.equal(max_predict_idx, max_label_idx)
    accuracy = tf.reduce_mean(tf.cast(correct_p, tf.float32))
    return accuracy


def train(height=CAPTCHA_HEIGHT,
          width=CAPTCHA_WIDTH,
          y_size=len(CAPTCHA_LIST) * CAPTCHA_LEN):
    '''
    cnn训练
    :param height:
    :param width:
    :param y_size:
    :return:
    '''
    # cnn在图像大小是2的倍数时性能最高, 如果图像大小不是2的倍数，可以在图像边缘补无用像素
    # 在图像上补2行，下补3行，左补2行，右补2行
    # np.pad(image,((2,3),(2,2)), 'constant', constant_values=(255,))

    acc_rate = 0.95
    # 按照图片大小申请占位符
    x = tf.placeholder(tf.float32, [None, height * width])
    y = tf.placeholder(tf.float32, [None, y_size])
    # 防止过拟合 训练时启用 测试时不启用
    keep_prob = tf.placeholder(tf.float32)
    # cnn模型
    y_conv = cnn_graph(x, keep_prob, (height, width))
    # 最优化
    optimizer = optimize_graph(y, y_conv)
    # 偏差
    accuracy = accuracy_graph(y, y_conv)
    # 启动会话.开始训练
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    step = 0
    while True:
        batch_x, batch_y = next_batch(64)
        sess.run(optimizer,
                 feed_dict={
                     x: batch_x,
                     y: batch_y,
                     keep_prob: 0.75
                 })
        # 每训练一百次测试一次
        if step % 100 == 0:
            batch_x_test, batch_y_test = next_batch(100)
            acc = sess.run(accuracy,
                           feed_dict={
                               x: batch_x_test,
                               y: batch_y_test,
                               keep_prob: 1.0
                           })
            print(datetime.now().strftime('%c'), ' step:', step, ' accuracy:',
                  acc)
            # 偏差满足要求，保存模型
            if acc > acc_rate:
                model_path = os.getcwd() + os.sep + str(
                    acc_rate) + "captcha.model"
                saver.save(sess, model_path, global_step=step)
                acc_rate += 0.01
                if acc_rate > 0.99:
                    break
        step += 1
    sess.close()


if __name__ == '__main__':
    train()
