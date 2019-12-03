import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from MtcnnDetector import MtcnnDetector
from detector import Detector
from fcn_detector import FcnDetector
from model import P_Net, R_Net, O_Net
from utils import *
import config
import os
import time
import copy


class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)  # 父类的构造函数
        self.setFixedSize(self.width(), self.height())
        self.timer_camera = QtCore.QTimer()  # 定义定时器，用于控制显示视频的帧率
        self.cap = cv2.VideoCapture()  # 视频流
        self.CAM_NUM = 0  # 为0时表示视频流来自笔记本内置摄像头
        self.set_ui()  # 初始化程序界面
        self.slot_init()  # 初始化槽函数
        self.icons = []
        self.gen_iconlist()
        self.coverflag = False
        self.embedingList = []
        self.mtcnn_detector = self.load_align()
        self.images_placeholder = None
        self.images_placeholder = None
        self.embeddings = None
        self.phase_train_placeholder = None
        self.keep_probability_placeholder = None
        self.sess = self.sess_init()
        self.THRED = 0.002
        self.count = 0
        self.iconPos = []
        self.iconclass = []
    '''程序界面布局'''

    def set_ui(self):
        self.__layout_main = QtWidgets.QHBoxLayout()  # 总布局
        self.__layout_fun_button = QtWidgets.QVBoxLayout()  # 按键布局
        self.__layout_data_show = QtWidgets.QVBoxLayout()  # 数据(视频)显示布局
        self.button_open_camera = QtWidgets.QPushButton('打开相机')  # 建立用于打开摄像头的按键
        self.button_close = QtWidgets.QPushButton('退出')  # 建立用于退出程序的按键
        self.button_cover = QtWidgets.QPushButton('遮盖')
        self.button_select = QtWidgets.QPushButton('已选 0 人')
        self.button_open_camera.setMinimumHeight(50)  # 设置按键大小
        self.button_close.setMinimumHeight(50)
        self.button_cover.setMinimumHeight(50)
        self.button_select.setMinimumHeight(50)
        # self.button_cover.move(10, 50)
        # self.button_cover.move(10, 100)
        # self.button_close.move(10, 150)  # 移动按键

        '''信息显示'''
        self.label_show_camera = QtWidgets.QLabel()  # 定义显示视频的Label
        self.label_show_camera.setFixedSize(
            641, 481)  # 给显示视频的Label设置大小为641x481
        '''把按键加入到按键布局中'''
        self.__layout_fun_button.addWidget(
            self.button_open_camera)  # 把打开摄像头的按键放到按键布局中
        self.__layout_fun_button.addWidget(self.button_select)
        self.__layout_fun_button.addWidget(self.button_cover)
        self.__layout_fun_button.addWidget(
            self.button_close)  # 把退出程序的按键放到按键布局中

        '''把某些控件加入到总布局中'''
        self.__layout_main.addLayout(self.__layout_fun_button)  # 把按键布局加入到总布局中
        self.__layout_main.addWidget(
            self.label_show_camera)  # 把用于显示视频的Label加入到总布局中
        '''总布局布置好后就可以把总布局作为参数传入下面函数'''
        self.setLayout(self.__layout_main)  # 到这步才会显示所有控件

    '''初始化所有槽函数'''

    def slot_init(self):
        # 若该按键被点击，则调用button_open_camera_clicked()
        self.button_open_camera.clicked.connect(
            self.button_open_camera_clicked)
        self.timer_camera.timeout.connect(
            self.show_camera)  # 若定时器结束，则调用show_camera()
        # 若该按键被点击，则调用close()，注意这个close是父类QtWidgets.QWidget自带的，会关闭程序
        self.button_close.clicked.connect(self.close)
        self.button_cover.clicked.connect(self.startCover)
        self.button_select.clicked.connect(self.genEmbedings)

    '''槽函数之一'''

    def startCover(self):
        if self.coverflag:
            self.coverflag = False
            self.button_cover.setText('遮盖')
        else:
            self.coverflag = True
            self.button_cover.setText('原图')

    def button_open_camera_clicked(self):
        if self.timer_camera.isActive() is False:  # 若定时器未启动
            # 参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
            flag = self.cap.open(self.CAM_NUM)
            if flag is False:  # flag表示open()成不成功
                msg = QtWidgets.QMessageBox.warning(
                    self, 'warning', "请检查相机与电脑是否连接正确", buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(30)  # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
                self.button_open_camera.setText('关闭相机')
        else:
            self.timer_camera.stop()  # 关闭定时器
            self.cap.release()  # 释放视频流
            self.label_show_camera.clear()  # 清空视频显示区域
            self.button_open_camera.setText('打开相机')

    def gen_iconlist(self):
        for i in range(10):
            icon = Image.open("../icons/" + str(i) + ".png")
            self.icons.append(icon)

    def addIcon(self, img0, classList, posList):
        if not len(classList):
            return img0
        img1 = Image.fromarray(img0)
        for i, pos in enumerate(posList):
            icon = self.icons[classList[i] % 10]
            icon = icon.resize(
                (pos[2] - pos[0], pos[3] - pos[1]), Image.ANTIALIAS)
            layer = Image.new('RGBA', img1.size, (0, 0, 0, 0))
            layer.paste(icon, (pos[0], pos[1]))
            img1 = Image.composite(layer, img1, layer)
        return np.asarray(img1)

    def show_camera(self):
        self.count += 1
        self.count %= 100
        flag, self.image = self.cap.read()  # 从视频流中读取
        if not flag:
            return
        show = cv2.resize(self.image, (640, 480))  # 把读到的帧的大小重新设置为 640x480
        show = cv2.flip(show, 1)
        if len(self.embedingList) and not self.count:
            self.genIDPos(copy.deepcopy(show))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
        if self.coverflag:
            # print(iconclass)
            show = self.addIcon(show, self.iconclass, self.iconPos)
        showImage = QtGui.QImage(
            show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.label_show_camera.setPixmap(
            QtGui.QPixmap.fromImage(showImage))  # 往显示视频的Label里 显示QImage

    def load_align(self):
        thresh = config.thresh
        min_face_size = config.min_face
        stride = config.stride
        test_mode = config.test_mode
        detectors = [None, None, None]
        # 模型放置位置
        model_path = ['./model/PNet/',
                      './model/RNet/', './model/ONet']
        batch_size = config.batches
        PNet = FcnDetector(P_Net, model_path[0])
        detectors[0] = PNet

        if test_mode in ["RNet", "ONet"]:
            RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
            detectors[1] = RNet

        if test_mode == "ONet":
            ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
            detectors[2] = ONet

        mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                       stride=stride, threshold=thresh)
        return mtcnn_detector

    def align_face_init(self, img):

        # 选用图片
        # 获取图片类别和路径
        # img_paths = os.listdir(path)
        # class_names = [a.split('.')[0] for a in img_paths]
        # img_paths = [os.path.join(path, p) for p in img_paths]
        scaled_arr = []
        class_names_arr = []
        start1 = time.clock()
        try:
            boxes_c, _ = self.mtcnn_detector.detect(img)
        except:
            print('识别不出图像\n')
            return None, None, None
        start2 = time.clock()
        # 人脸框数量
        num_box = boxes_c.shape[0]
        if num_box > 0:
            det = boxes_c[:, :4]
            det_arr = []
            img_size = np.asarray(img.shape)[:2]
            if num_box > 1:

                # 如果保留一张脸，但存在多张，只保留置信度最大的
                score = boxes_c[:, 4]
                index = np.argmax(score)
                det_arr.append(det[index, :])
            else:
                det_arr.append(np.squeeze(det))
            for i, det in enumerate(det_arr):
                det = np.squeeze(det)
                bb = [
                    int(max(det[0], 0)),
                    int(max(det[1], 0)),
                    int(min(det[2], img_size[1])),
                    int(min(det[3], img_size[0]))
                ]
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]

                scaled = cv2.resize(
                    cropped,
                    (160, 160), interpolation=cv2.INTER_LINEAR)
                scaled = cv2.cvtColor(
                    scaled, cv2.COLOR_BGR2RGB) - 127.5 / 128.0
                scaled_arr.append(scaled)
                class_names_arr.append(i)

        else:
            print('图像不能对齐')
        scaled_arr = np.asarray(scaled_arr)
        class_names_arr = np.asarray(class_names_arr)
        start3 = time.clock()
        print(str(start2 - start1), (start3 - start2))
        return scaled_arr, class_names_arr

    def align_face(self, img):
        try:
            boxes_c, _ = self.mtcnn_detector.detect(img)
        except:
            print('找不到脸')
            return None, None, None
        # 人脸框数量
        num_box = boxes_c.shape[0]
        scaled_arr = []
        recList = []
        if num_box > 0:
            det = boxes_c[:, :4]
            det_arr = []
            img_size = np.asarray(img.shape)[:2]
            for i in range(num_box):
                det_arr.append(np.squeeze(det[i]))

            for i, det in enumerate(det_arr):
                det = np.squeeze(det)
                bb = [int(max(det[0], 0)), int(max(det[1], 0)), int(
                    min(det[2], img_size[1])), int(min(det[3], img_size[0]))]
                recList.append(bb)
                # cv2.rectangle(img, (bb[0], bb[1]),
                #              (bb[2], bb[3]), (0, 255, 0), 2)
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                scaled = cv2.resize(cropped, (160, 160),
                                    interpolation=cv2.INTER_LINEAR)
                scaled = cv2.cvtColor(
                    scaled, cv2.COLOR_BGR2RGB) - 127.5 / 128.0
                scaled_arr.append(scaled)
            scaled_arr = np.array(scaled_arr)
            return img, scaled_arr, recList
        else:
            print('找不到脸 ')
            return None, None, None

    def load_model(self, model_dir, input_map=None):
        '''重载模型'''

        ckpt = tf.train.get_checkpoint_state(model_dir)
        saver = tf.train.import_meta_graph(
            ckpt.model_checkpoint_path + '.meta')
        saver.restore(tf.get_default_session(), ckpt.model_checkpoint_path)

    def sess_init(self):
        sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state('../model/')
        saver = tf.train.import_meta_graph(
            ckpt.model_checkpoint_path + '.meta')
        saver.restore(sess, ckpt.model_checkpoint_path)
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name(
            "input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name(
            "embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph(
        ).get_tensor_by_name("phase_train:0")
        self.keep_probability_placeholder = tf.get_default_graph(
        ).get_tensor_by_name('keep_probability:0')
        return sess

    def genEmbedings(self):
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.warning(self, "标题", "请先打开相机")
            return
        else:
            time.sleep(1)
            flag, img = self.cap.read()
        if flag:
            self.timer_camera.stop()
            QtWidgets.QMessageBox.information(self, "标题", "选取成功")
        else:
            QtWidgets.QMessageBox.warning(self, "标题", "选取失败")
            return
        scaled_arr, class_arr = self.align_face_init(img)
        feed_dict = {
            self.images_placeholder: scaled_arr,
            self.phase_train_placeholder: False,
            self.keep_probability_placeholder: 1.0
        }
        embs = self.sess.run(self.embeddings, feed_dict=feed_dict)
        self.embedingList.append(embs)
        self.button_select.setText('已选 ' + str(len(self.embedingList)) + ' 人')
        self.timer_camera.start(30)

    def genIDPos(self, img):
        with tf.Graph().as_default():
            img, scaled_arr, recList = self.align_face(img)
            if scaled_arr is not None:
                feed_dict = {self.images_placeholder: scaled_arr,
                             self.phase_train_placeholder: False, self.keep_probability_placeholder: 1.0}
                embs = self.sess.run(self.embeddings, feed_dict=feed_dict)
                face_num = embs.shape[0]
                face_class = []
                icons_pos_scale = []
                for i in range(face_num):
                    diff = []
                    for man in self.embedingList:
                        diff.append(np.mean(np.square(embs[i] - man), axis=1))
                    min_diff = min(diff)
                    # print(min_diff)
                    if min_diff < 0.002:
                        face_class.append(np.argmin(diff))
                        icons_pos_scale.append(recList[i])
                    else:
                        break
                print(face_class)
                self.iconclass = face_class
                self.iconPos = icons_pos_scale
            else:
                self.iconclass = []
                self.iconPos = []


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)  # 固定的，表示程序应用
    ui = Ui_MainWindow()  # 实例化Ui_MainWindow
    ui.show()  # 调用ui的show()以显示。同样show()是源于父类QtWidgets.QWidget的
    sys.exit(app.exec_())  # 不加这句，程序界面会一闪而过
