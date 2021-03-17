#成功实现单摄像头实时画面
# 打包：https://blog.csdn.net/kobeyu652453/article/details/108871179
import sys
import threading

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow
# import icon_rc
import cv2
from my_test import Ui_MainWindow
import numpy as np
# import pygame
import winsound
from PIL import ImageFont, ImageDraw, Image
import time
from operator import itemgetter
import gxipy as gx


class mainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        # 窗口初始化
        super(mainWindow, self).__init__()
        self.setupUi(self)

        self.timer_camera = QtCore.QTimer()  # 定时器
        self.setupUi(self)
        self.retranslateUi(self)

        # 钢管表面摄像头
        self.device_manager1 = gx.DeviceManager()
        self.dev_num1, dev_info_list1 = self.device_manager1.update_device_list()
        self.Width_set1 = 1280  # 设置分辨率宽
        self.Height_set1 = 960  # 设置分辨率高

        # 钢管侧边口摄像头
        self.device_manager2 = gx.DeviceManager()
        self.dev_num2, dev_info_list2 = self.device_manager2.update_device_list()
        self.Width_set2 = 1280  # 设置分辨率宽
        self.Height_set2 = 960  # 设置分辨率高

        self.slot_init()  # 设置槽函数
        # UI界面默认文本设置140，参考https://blog.csdn.net/zhangxuechao_/article/details/82257914
        self.set_thresh = 140

        # 初始化高斯滤波参数
        self.ksize_w = 9
        self.ksize_h = 25

        # 初始化钢管表面提取黑线等特征的参数
        self.blockSize_surface = 9
        self.C_surface = 5

        # 初始化钢管侧边管口特征的参数
        self.blockSize_pipe = 25
        self.C_pipe = 7

        # 初始化钢管宽度预设的参数
        self.width_min = 105
        self.width_max = 130

        # 初始化钢管边框参数
        self.x_left = 0
        self.x_right = 0
        self.width = 0

        # 初始化钢管两侧边直线方程参数
        self.k1_slope = 0
        self.k2_slope = 0
        self.k11_slope = 0

        self.b1_intercept = 0
        self.b2_intercept = 0

        # 初始化缺陷边框的中心坐标
        self.x_defect = 0
        self.y_defect = 0

        # 初始化缺陷的长和宽
        self.x_defect_width = 0
        self.x_defect_height = 0

        self.middle_width = 0
        self.middle_height = 0

        self.middle_width_i = 0
        self.middle_height_i = 0

        # 黑线的坐标
        self.x_defect1 = 0
        self.y_defect1 = 0

        # 初始化缺陷中心到两边缘直线的距离
        self.d1 = 0
        self.d2 = 0

        # 初始化d1,d2的最小值，用于提取黑线
        self.min_length = 0

        # 初始化钢管矩形框的宽度
        self.total_length = 0

        # 初始化黑线检测到的个数
        self.count_nums = 0

        # box坐标临时值
        self.box_tmp = [[0, 0], [0, 0], [0, 0], [0, 0]]

        # rate临时值
        self.rate1 = 0

        # 初始化画面中心高度
        self.middle_height = 0

        # 初始化钢管边框各角点左边
        (self.x_left_lower, self.y_left_lower) = (0, 0)
        (self.x_left_upper, self.y_left_upper) = (0, 0)
        (self.x_right_upper, self.y_right_upper) = (0, 0)
        (self.x_right_lower, self.y_right_lower) = (0, 0)

        # 初始化偏转角
        self.theta_angle = 0

        # 定义直线方程
        # 往左上方偏
        self.linear_equation1 = 0
        # 往右上方偏
        self.linear_equation2 = 0

        (self.x, self.y, self.w, self.h) = (0, 0, 0, 0)
        # 判断是否使用多线程来发出报警声
        self.flag = False

        # 因为后面要用到textEdit_edit1，但又不想让它出现，所以将它隐藏
        self.textEdit_edit1.setVisible(False)
        self.textEdit_edit2.setVisible(False)
        self.textEdit_edit3.setVisible(False)
        self.textEdit_edit2_2.setVisible(False)
        self.textEdit_edit3_2.setVisible(False)
        self.textEdit_edit2_3.setVisible(False)
        self.textEdit_edit3_3.setVisible(False)
        self.textEdit_edit_width_min.setVisible(False)
        self.textEdit_edit_width_max.setVisible(False)

        # UI界面摄像头显示区域背景颜色设置，参考：https://jingyan.baidu.com/article/bea41d43681da4b4c51be61d.html

        # 定义蜂鸣器
        # 这个只是针对Windows系统才有的，参考：https://blog.csdn.net/weixin_41822224/article/details/100167499
        # duration是铃声持续时间，单位毫秒，持续时间越长，延长越严重
        self.duration = 100
        self.freq = 3000

        self.count_contours = 0

        # 缺口一定是在黑线存在的情况下才出现，定义一个标志位
        self.black_line = False

        # bool灯初始化为灰色
        # 钢管1
        self.label1_1.setPixmap(QPixmap('images/dark.png'))
        self.label1_2.setPixmap(QPixmap('images/dark.png'))
        self.label1_3.setPixmap(QPixmap('images/dark.png'))
        self.label1_4.setPixmap(QPixmap('images/dark.png'))
        self.label1_5.setPixmap(QPixmap('images/dark.png'))
        self.label1_6_1.setPixmap(QPixmap('images/dark.png'))
        self.label1_6_2.setPixmap(QPixmap('images/dark.png'))
        self.label1_7.setPixmap(QPixmap('images/dark.png'))
        self.label1_8.setPixmap(QPixmap('images/dark.png'))
        self.label1_9.setPixmap(QPixmap('images/dark.png'))
        self.label1_10.setPixmap(QPixmap('images/dark.png'))
        self.label1_11.setPixmap(QPixmap('images/dark.png'))
        self.label1_12.setPixmap(QPixmap('images/dark.png'))
        self.label1_13.setPixmap(QPixmap('images/dark.png'))

        # 钢管2
        self.label2_1.setPixmap(QPixmap('images/dark.png'))
        self.label2_2.setPixmap(QPixmap('images/dark.png'))
        self.label2_3.setPixmap(QPixmap('images/dark.png'))
        self.label2_4.setPixmap(QPixmap('images/dark.png'))
        self.label2_5.setPixmap(QPixmap('images/dark.png'))
        self.label2_6_1.setPixmap(QPixmap('images/dark.png'))
        self.label2_6_2.setPixmap(QPixmap('images/dark.png'))
        self.label2_7.setPixmap(QPixmap('images/dark.png'))
        self.label2_8.setPixmap(QPixmap('images/dark.png'))
        self.label2_9.setPixmap(QPixmap('images/dark.png'))
        self.label2_10.setPixmap(QPixmap('images/dark.png'))
        self.label2_11.setPixmap(QPixmap('images/dark.png'))
        self.label2_12.setPixmap(QPixmap('images/dark.png'))
        self.label2_13.setPixmap(QPixmap('images/dark.png'))

        # 钢管3
        self.label3_1.setPixmap(QPixmap('images/dark.png'))
        self.label3_2.setPixmap(QPixmap('images/dark.png'))
        self.label3_3.setPixmap(QPixmap('images/dark.png'))
        self.label3_4.setPixmap(QPixmap('images/dark.png'))
        self.label3_5.setPixmap(QPixmap('images/dark.png'))
        self.label3_6_1.setPixmap(QPixmap('images/dark.png'))
        self.label3_6_2.setPixmap(QPixmap('images/dark.png'))
        self.label3_7.setPixmap(QPixmap('images/dark.png'))
        self.label3_8.setPixmap(QPixmap('images/dark.png'))
        self.label3_9.setPixmap(QPixmap('images/dark.png'))
        self.label3_10.setPixmap(QPixmap('images/dark.png'))
        self.label3_11.setPixmap(QPixmap('images/dark.png'))
        self.label3_12.setPixmap(QPixmap('images/dark.png'))
        self.label3_13.setPixmap(QPixmap('images/dark.png'))

    def slot_init(self):
        # 设置开启摄像头
        self.orgin_btn.clicked.connect(self.btn_orgin)

        # 自定义设置阈值
        # self.set_thresh_btn.clicked.connect(self.btn_set_thresh)

        # 自定义钢管表面阈值分割参数
        self.surface_set_block_size_btn.clicked.connect(self.btn_set_block_size_surface)

        # 自定义钢管侧边管口阈值分割参数
        self.pipe_set_block_size_btn.clicked.connect(self.btn_set_block_size_pipe)

        # 自定义钢管宽度设置参数
        self.width_set_btn.clicked.connect(self.btn_width_set)

        # # 自定义高斯核
        self.set_ksize_btn.clicked.connect(self.btn_set_ksize)

        # 自定义方法
        self.timer_camera.timeout.connect(self.handle_threshold)

        # 初始化排序列表
        self.list = []

    def btn_orgin(self):
        if self.timer_camera.isActive() == False:  # 若定时器未启动

            if self.dev_num1 == 0:  # 钢管表面的摄像头未打开
                msg = QtWidgets.QMessageBox.warning(
                    self, u"Warning", u"钢管表面的摄像头未打开，请检测相机与电脑是否连接正确",
                    buttons=QtWidgets.QMessageBox.Ok,
                    defaultButton=QtWidgets.QMessageBox.Ok)
            if self.dev_num2 == 0:  # 钢管表面的摄像头未打开
                msg = QtWidgets.QMessageBox.warning(
                    self, u"Warning", u"钢管管口的摄像头未打开，请检测相机与电脑是否连接正确",
                    buttons=QtWidgets.QMessageBox.Ok,
                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.CAM_NUM1 = self.device_manager1.open_device_by_index(1)
                self.CAM_NUM2 = self.device_manager2.open_device_by_index(2)
                self.CAM_NUM1.Width.set(self.Width_set1)
                self.CAM_NUM1.Height.set(self.Height_set1)
                self.CAM_NUM2.Width.set(self.Width_set2)
                self.CAM_NUM2.Height.set(self.Height_set2)
                self.CAM_NUM1.stream_on()
                self.CAM_NUM2.stream_on()

                # 摄像头已打开
                self.count_time = 1
                self.timer_camera.start(30)  # 30毫秒读一帧
                self.orgin_btn.setText('关闭相机')
        else:
            self.timer_camera.stop()  # 关闭定时器
            self.CAM_NUM1.stream_off()
            self.CAM_NUM2.stream_off()
            self.CAM_NUM1.close_device()
            self.CAM_NUM2.close_device()
            # 清空界面显示区域
            # self.image1_label.clear()
            self.image_label.clear()
            self.pipe_label.clear()
            self.thresh_label.clear()
            self.orgin_btn.setText('打开相机')
            self.count_time = 0  # 此时摄像头未打开

        # https://blog.csdn.net/humanking7/article/details/80403189

    def btn_set_block_size_surface(self):
        self.textEdit_edit2.setText(self.textEdit_thresh1.text())
        self.textEdit_edit3.setText(self.textEdit_thresh2.text())
        self.blockSize_surface = int(self.textEdit_thresh1.text())
        self.C_surface = int(self.textEdit_thresh2.text())

    def btn_set_block_size_pipe(self):
        self.textEdit_edit2_3.setText(self.textEdit_thresh1_2.text())
        self.textEdit_edit3_3.setText(self.textEdit_thresh2_2.text())
        self.blockSize_pipe = int(self.textEdit_thresh1_2.text())
        self.C_pipe = int(self.textEdit_thresh2_2.text())

    def btn_width_set(self):
        self.textEdit_edit_width_min.setText(self.textEdit_width_min.text())
        self.textEdit_edit_width_max.setText(self.textEdit_width_max.text())
        self.width_min = int(self.textEdit_width_min.text())
        self.width_max = int(self.textEdit_width_max.text())

    def btn_set_ksize(self):
        # 这儿不知道为什么像textEdit_thresh1无法查看定义反而没问题，能够查看定义却显示没有text()函数课调用
        self.textEdit_edit2_2.setText(self.textEdit_thresh3.text())
        self.textEdit_edit3_2.setText(self.textEdit_thresh4.text())
        self.ksize_w = int(self.textEdit_thresh3.text())
        self.ksize_h = int(self.textEdit_thresh4.text())

    def set_size(self, scale_percentThreash, image_path):
        # scale_percentThreash = 25
        widthThreash = int(image_path.shape[1] * scale_percentThreash / 100)
        heightThreash = int(image_path.shape[0] * scale_percentThreash / 100)
        dimThreash = (widthThreash, heightThreash)
        # # resize image
        result = cv2.resize(image_path, dimThreash, interpolation=cv2.INTER_AREA)
        return result

    # 参考：https://blog.csdn.net/qq_39622065/article/details/84859629
    def putText(img, text, org, font_path, color=(0, 0, 255), font_size=40):
        """
        在图片上显示文字
        :param img: 输入的img, 通过cv2读取
        :param text: 要显示的文字
        :param org: 文字左上角坐标
        :param font_path: 字体路径
        :param color: 字体颜色, (B,G,R)
        :return:
        """
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        b, g, r = color
        a = 0
        draw.text(org, text, font=ImageFont.truetype(font_path, font_size), fill=(b, g, r, a))
        img = np.array(img_pil)
        return img

    def handle_threshold(self):
        start = time.clock()  # 记下开始时刻
        # 初始化相关参数
        self.flag = False
        self.count_nums = 0
        self.count_contours = 0
        self.list = []
        self.black_line = False

        # bool灯初始化为灰色，因为每帧都要显示，效果为一闪一闪的
        # 钢管1
        self.label1_1.setPixmap(QPixmap('images/dark.png'))
        self.label1_2.setPixmap(QPixmap('images/dark.png'))
        self.label1_3.setPixmap(QPixmap('images/dark.png'))
        self.label1_4.setPixmap(QPixmap('images/dark.png'))
        self.label1_5.setPixmap(QPixmap('images/dark.png'))
        self.label1_6_1.setPixmap(QPixmap('images/dark.png'))
        self.label1_6_2.setPixmap(QPixmap('images/dark.png'))
        self.label1_7.setPixmap(QPixmap('images/dark.png'))
        self.label1_8.setPixmap(QPixmap('images/dark.png'))
        self.label1_9.setPixmap(QPixmap('images/dark.png'))
        self.label1_10.setPixmap(QPixmap('images/dark.png'))
        self.label1_11.setPixmap(QPixmap('images/dark.png'))
        self.label1_12.setPixmap(QPixmap('images/dark.png'))
        self.label1_13.setPixmap(QPixmap('images/dark.png'))

        # 钢管2
        self.label2_1.setPixmap(QPixmap('images/dark.png'))
        self.label2_2.setPixmap(QPixmap('images/dark.png'))
        self.label2_3.setPixmap(QPixmap('images/dark.png'))
        self.label2_4.setPixmap(QPixmap('images/dark.png'))
        self.label2_5.setPixmap(QPixmap('images/dark.png'))
        self.label2_6_1.setPixmap(QPixmap('images/dark.png'))
        self.label2_6_2.setPixmap(QPixmap('images/dark.png'))
        self.label2_7.setPixmap(QPixmap('images/dark.png'))
        self.label2_8.setPixmap(QPixmap('images/dark.png'))
        self.label2_9.setPixmap(QPixmap('images/dark.png'))
        self.label2_10.setPixmap(QPixmap('images/dark.png'))
        self.label2_11.setPixmap(QPixmap('images/dark.png'))
        self.label2_12.setPixmap(QPixmap('images/dark.png'))
        self.label2_13.setPixmap(QPixmap('images/dark.png'))

        # 钢管3
        self.label3_1.setPixmap(QPixmap('images/dark.png'))
        self.label3_2.setPixmap(QPixmap('images/dark.png'))
        self.label3_3.setPixmap(QPixmap('images/dark.png'))
        self.label3_4.setPixmap(QPixmap('images/dark.png'))
        self.label3_5.setPixmap(QPixmap('images/dark.png'))
        self.label3_6_1.setPixmap(QPixmap('images/dark.png'))
        self.label3_6_2.setPixmap(QPixmap('images/dark.png'))
        self.label3_7.setPixmap(QPixmap('images/dark.png'))
        self.label3_8.setPixmap(QPixmap('images/dark.png'))
        self.label3_9.setPixmap(QPixmap('images/dark.png'))
        self.label3_10.setPixmap(QPixmap('images/dark.png'))
        self.label3_11.setPixmap(QPixmap('images/dark.png'))
        self.label3_12.setPixmap(QPixmap('images/dark.png'))
        self.label3_13.setPixmap(QPixmap('images/dark.png'))

        image1 = self.CAM_NUM1.data_stream[0].get_image()
        numpy_image1 = image1.get_numpy_array()
        # display image with opencv
        pimg1 = cv2.cvtColor(np.asarray(numpy_image1), cv2.COLOR_GRAY2BGR)

        image2 = self.CAM_NUM2.data_stream[0].get_image()
        numpy_image2 = image2.get_numpy_array()
        # display image with opencv
        pimg2 = cv2.cvtColor(np.asarray(numpy_image2), cv2.COLOR_GRAY2BGR)

        frameSurface = cv2.flip(pimg1, -1)
        framePipe = cv2.flip(pimg2, -1)

        image = cv2.cvtColor(frameSurface, cv2.COLOR_BGR2RGB)
        # self.set_size(50, image)
        scale_percent = 50
        image_width = int(image.shape[1] * scale_percent / 100)
        image_height = int(image.shape[0] * scale_percent / 100)
        image_dim = (image_width, image_height)
        image = cv2.resize(image, image_dim, interpolation=cv2.INTER_AREA)

        imagePipe = cv2.cvtColor(framePipe, cv2.COLOR_BGR2RGB)
        scale_percent_pipe = 40
        imagePipe_width = int(imagePipe.shape[1] * scale_percent_pipe / 100)
        imagePipe_height = int(imagePipe.shape[0] * scale_percent_pipe / 100)
        imagePipe_dim = (imagePipe_width, imagePipe_height)
        imagePipe = cv2.resize(imagePipe, imagePipe_dim, interpolation=cv2.INTER_AREA)
        # self.set_size(50, imagePipe)

        Grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        GrayimgPipe = cv2.cvtColor(imagePipe, cv2.COLOR_BGR2GRAY)

        Grayimg = cv2.GaussianBlur(Grayimg, (self.ksize_w, self.ksize_h), 0)
        GrayimgPipe = cv2.GaussianBlur(GrayimgPipe, (self.ksize_w, self.ksize_h), 0)

        thresh = cv2.adaptiveThreshold(Grayimg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, self.blockSize_surface, self.C_surface)
        binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
        scale_percent_thresh = 60
        widthThreash = int(image.shape[1] * scale_percent_thresh / 100)
        heightThreash = int(image.shape[0] * scale_percent_thresh / 100)
        dim_threash = (widthThreash, heightThreash)
        thresh = cv2.resize(thresh, dim_threash, interpolation=cv2.INTER_AREA)

        threshPipe = cv2.adaptiveThreshold(GrayimgPipe, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, self.blockSize_pipe, self.C_pipe)
        binaryPipe, contoursPipe, hierarchyPipe = cv2.findContours(threshPipe, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
        # scale_percent_pipe_thresh = 40
        # width_pipe_threash = int(threshPipe.shape[1] * scale_percent_pipe_thresh / 100)
        # height_pipe_threash = int(threshPipe.shape[0] * scale_percent_pipe_thresh / 100)
        # dim_pipe_threash = (width_pipe_threash, height_pipe_threash)
        # threshPipe = cv2.resize(threshPipe, dim_pipe_threash, interpolation=cv2.INTER_AREA)

        # 获取自适应阈值
        ret1, thresh2 = cv2.threshold(Grayimg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 这里先设置自适应阈值
        self.set_thresh = ret1
        ret, thresh1 = cv2.threshold(Grayimg, self.set_thresh, 255, cv2.THRESH_BINARY)

        binary1, contours1, hierarchy1 = cv2.findContours(thresh1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
       # print(contours1)
        for c1 in range(len(contours1)):
            # 特征提取框的特征
            # https://blog.csdn.net/qq_37385726/article/details/82313558
            rect1 = cv2.minAreaRect(contours1[c1])

            # # 钢管边框的中心坐标
            # self.middle_width = rect1[0][0]
            # self.middle_height = rect1[0][1]

            # print('(self.middle_width, self.middle_height)', (self.middle_width, self.middle_height))
            # cx, cy = rect[0]  #中心位置
            # 获取矩形四个顶点，浮点型
            box1 = cv2.boxPoints(rect1)

            # 取整
            box1 = np.int0(box1)
            # cv2.drawContours(image1, [box], 0, (0, 255, 0), 5)

            self.theta_angle = rect1[2]

            # 因为钢管偏左和偏右，它的box[0]不一样，这里的计数点从左下->左上->右上->右下
            # 往左上偏
            if -45.0 <= self.theta_angle < -0.0001:
                # 钢管边框的中心坐标
                self.middle_width = rect1[1][0]
                self.middle_height = rect1[1][1]
                # 如果不用continue，直接添加符合要求的会有一些错误的进入
                # if 170 < rect1[1][1] < 190:
                #     self.list.append(self.middle_width)
                if rect1[1][0] < self.width_min or rect1[1][0] > self.width_max:
                    continue
                self.list.append(rect1)
                cv2.drawContours(image, [box1], 0, (0, 255, 0), 5)

            # 往右上偏
            if -89.9999 < self.theta_angle < -45.0:
                # 钢管边框的中心坐标
                self.middle_width = rect1[1][1]
                self.middle_height = rect1[1][0]
                # 如果不用continue，直接添加符合要求的会有一些错误的进入
                # if 170 < rect1[1][1] < 190:
                #     self.list.append(self.middle_width)
                if rect1[1][1] < self.width_min or rect1[1][1] > self.width_max:
                    continue
                self.list.append(rect1)
                cv2.drawContours(image, [box1], 0, (0, 255, 0), 5)

            if self.theta_angle == -90.0 or self.theta_angle == 0.0:
                # 钢管边框的中心坐标
                self.middle_width = rect1[1][1]
                self.middle_height = rect1[1][0]
                # print('(self.middle_width, self.middle_height)', (self.middle_width, self.middle_height))
                # 如果不用continue，直接添加符合要求的会有一些错误的进入
                # if 170 < rect1[1][1] < 190:
                #     self.list.append(self.middle_width)
                if rect1[1][1] < self.width_min or rect1[1][1] > self.width_max:
                    continue
                self.list.append(rect1)
                cv2.drawContours(image, [box1], 0, (0, 255, 0), 5)

        if self.theta_angle == -90.0 or self.theta_angle == 0.0:
            self.list = sorted(self.list)
            for i in range(len(self.list)):
                rect_i = self.list[i]
                # 钢管边框的中心坐标
                self.middle_width_i = rect_i[0][0]
                self.middle_height_i = rect_i[0][1]
                # 获取矩形四个顶点，浮点型
                box_i = cv2.boxPoints(rect_i)
                # # 取整
                box_i = np.int0(box_i)
                self.theta_angle = rect_i[2]
            #
                self.total_length = rect_i[1][1]
                # 在UI界面显示偏转角度
                str2 = str(self.theta_angle)
                self.textEdit_edit_angle.setText(str2)
                # 在UI界面显示钢管宽度,往右上偏时宽度为rect[1][1]
                str3 = str(self.total_length)
                self.textEdit_edit_width.setText(str3)

                (self.x_left_lower, self.y_left_lower) = (box_i[1][0], box_i[1][1])
                (self.x_left_upper, self.y_left_upper) = (box_i[2][0], box_i[2][1])
                (self.x_right_upper, self.y_right_upper) = (box_i[3][0], box_i[3][1])
                (self.x_right_lower, self.y_right_lower) = (box_i[0][0], box_i[0][1])

                # 这个是每根钢管表面的具体缺陷
                for c in range(len(contours)):
                    # 特征提取框的坐标以及长宽
                    self.x, self.y, self.w, self.h = cv2.boundingRect(contours[c])

                    # minAreaRect是计算最小面积，也即可以是斜着的矩形框
                    rect = cv2.minAreaRect(contours[c])
                    # cx, cy = rect[0]  #中心位置
                    # 获取矩形四个顶点，浮点型
                    box = cv2.boxPoints(rect)
                    # 取整
                    box = np.int0(box)

                    # rect[0][0], rect[0][1]表示矩形框的中心位置
                    (self.x_defect, self.y_defect) = (rect[0][0], rect[0][1])

                    (self.x_defect_width, self.x_defect_height) = (rect[1][0], rect[1][1])
                    # (self.x_defect_height, self.x_defect_width) = (rect[1][0], rect[1][1])
                    # print('(self.x_defect_height, self.x_defect_width)', (self.x_defect_height, self.x_defect_width))
                    # 钢管的缺陷宽度不可能大于钢管的宽度
                    if self.x_defect_width > self.total_length:
                        continue
                    # 去除钢管边缘外最大的一个矩形框
                    if self.total_length > 200:
                        continue
                    if self.x_defect < self.x_left_lower or self.x_defect > self.x_right_lower:
                        continue
                    self.d1 = abs(self.x_defect - self.x_left_lower)
                    self.d2 = abs(self.x_defect - self.x_right_lower)
                    # # 缺陷到两边缘的最小距离
                    self.min_length = min(self.d1, self.d2)
                    # print('self.min_length', self.min_length)
                    rate = self.min_length / self.total_length
                    # print('rate', rate)
                    print('(self.x_defect_width, self.x_defect_height)', (self.x_defect_width, self.x_defect_height))

                    # 黑线特征提取
                    if 0.09 < rate < 0.14 and 0 < self.x_defect_width < 19:  # 增加这个条件self.x_defect_width < 10是与缺口分开
                        cv2.drawContours(image, [box], 0, (0, 0, 205), 5)  # 这个位置是RGB
                        self.count_nums = self.count_nums + 1
                        self.rate1 = rate

                        # self.count_nums > 9是针对不连续的黑线，60 < self.x_defect_height < 200是针对连续的长条形黑线
                        if self.count_nums > 9 or 60 < self.x_defect_height < 200:
                            # print('self.count_nums', self.count_nums)
                            # print('self.x_defect_width', self.x_defect_width)
                            cv2img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
                            pilimg = Image.fromarray(cv2img)

                            # PIL图片上打印汉字
                            draw = ImageDraw.Draw(pilimg)  # 图片上打印
                            font = ImageFont.truetype("typefaces/simsun.ttf", 40, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
                            draw.text((int(self.middle_width_i), int(self.middle_height_i)), "黑线", (205, 0, 0),
                                      # 这个位置是BGR
                                      font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体

                            # PIL图片转cv2 图片
                            image = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)

                            # 考虑一下多线程，让另一个线程来处理发出声音
                            self.flag = True

                            # 多线程调用另一个函数，这样就不会出现因为另一个函数执行时间而卡顿
                            r = threading.Thread(target=self.beef)
                            r.setDaemon(False)
                            r.start()

                            # break只让一帧里面出现一次标签
                            break

                    # 缺口特征提取
                    if 0.05 < rate < 0.11:
                        # print('(self.x_defect_width, self.x_defect_height)', (self.x_defect_width, self.x_defect_height))
                        if 25 < self.x_defect_height < 40:
                            cv2.drawContours(image, [box], 0, (238, 44, 44), 5)  # 这个位置是RGB
                            cv2img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
                            pilimg = Image.fromarray(cv2img)
                            # PIL图片上打印汉字y
                            draw = ImageDraw.Draw(pilimg)  # 图片上打印
                            font = ImageFont.truetype("typefaces/simsun.ttf", 40, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
                            # draw.text((int(self.middle_width_i), int(self.x_defect_height)), "缺口", (44, 44, 238),
                            #           # 这个位置是BGR
                            #           font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
                            draw.text((int(self.middle_width_i), int(self.y_defect)), "缺口", (44, 44, 238),
                                      # 这个位置是BGR
                                      font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
                            # PIL图片转cv2 图片
                            image = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)

        if -45.0 <= self.theta_angle < -0.0001:
            self.list = sorted(self.list)
            for i in range(len(self.list)):
                rect_i = self.list[i]
                # 钢管边框的中心坐标
                self.middle_width_i = rect_i[0][0]
                # 获取矩形四个顶点，浮点型
                box_i = cv2.boxPoints(rect_i)
                # # 取整
                box_i = np.int0(box_i)
                self.theta_angle = rect_i[2]
                self.total_length = rect_i[1][0]
                # 在UI界面显示偏转角度
                str2 = str(self.theta_angle)
                self.textEdit_edit_angle.setText(str2)
                # 在UI界面显示钢管宽度,往右上偏时宽度为rect[1][1]
                str3 = str(self.total_length)
                self.textEdit_edit_width.setText(str3)

                (self.x_left_lower, self.y_left_lower) = (box1[0][0], box1[0][1])
                (self.x_left_upper, self.y_left_upper) = (box1[1][0], box1[1][1])
                (self.x_right_upper, self.y_right_upper) = (box1[2][0], box1[2][1])
                (self.x_right_lower, self.y_right_lower) = (box1[3][0], box1[3][1])

                # 左边这条直线
                k1 = (self.y_left_upper - self.y_left_lower) / (self.x_left_upper - self.x_left_lower)
                b1 = self.y_left_lower - k1 * self.x_left_lower
                # 右边这条直线
                k2 = (self.y_right_lower - self.y_right_upper) / (self.x_right_lower - self.x_right_upper)
                b2 = self.y_right_upper - k2 * self.x_right_upper
                self.k1_slope = k1
                self.k2_slope = k2
                self.b1_intercept = b1
                self.b2_intercept = b2
                # 这个是每根钢管表面的具体缺陷
                for c in range(len(contours)):
                    # 特征提取框的坐标以及长宽
                    self.x, self.y, self.w, self.h = cv2.boundingRect(contours[c])

                    # minAreaRect是计算最小面积，也即可以是斜着的矩形框
                    rect = cv2.minAreaRect(contours[c])
                    # cx, cy = rect[0]  #中心位置
                    # 获取矩形四个顶点，浮点型
                    box = cv2.boxPoints(rect)
                    # 取整
                    box = np.int0(box)

                    # rect[0][0], rect[0][1]表示矩形框的中心位置
                    (self.x_defect, self.y_defect) = (rect[0][0], rect[0][1])

                    (self.x_defect_height, self.x_defect_width) = (rect[1][1], rect[1][0])
                    # 钢管的缺陷宽度不可能大于钢管的宽度
                    if self.x_defect_width > self.total_length:
                        continue

                    # 根据点与直线的关系
                    # 当直线往左上方偏时，若kx-y+b>0,则点在直线右侧，若kx-y+b<0,则点在直线左侧，
                    # 当直线往右上方偏时，若kx-y+b<0,则点在直线右侧，若kx-y+b>0,则点在直线左侧，
                    # # 左边这条直线
                    self.linear_equation1 = self.k1_slope * self.x_defect - self.y_defect + self.b1_intercept
                    # 右边这条直线
                    self.linear_equation2 = self.k2_slope * self.x_defect - self.y_defect + self.b2_intercept

                    # 去除钢管边缘外最大的一个矩形框
                    if self.total_length > 200:
                        continue
                    # print('contours', len(contours))

                    # 钢管左边缘的那些误差缺陷最右边坐标小于钢管左边缘坐标
                    # if box[1][0] < x_left_upper or box[0][0] < x_left_lower or box[2][0] > x_right_upper or box[3][0] > x_right_lower:
                    if self.linear_equation1 < 0 or self.linear_equation2 > 0:
                        continue

                    # 缺陷中心到两直线的距离，每个缺陷有两个d
                    self.d1 = abs(self.k1_slope * self.x_defect - self.y_defect + self.b1_intercept) / (
                            (-1) * (-1) + self.k1_slope * self.k1_slope) ** 0.5
                    self.d2 = abs(self.k2_slope * self.x_defect - self.y_defect + self.b2_intercept) / (
                            (-1) * (-1) + self.k2_slope * self.k2_slope) ** 0.5
                    # # 缺陷到两边缘的最小距离
                    self.min_length = min(self.d1, self.d2)
                    # print('self.min_length', self.min_length)
                    rate = self.min_length / self.total_length

                    # 黑线特征提取
                    if 0.08 < rate < 0.12 and 0 < self.x_defect_width < 8:  # 增加这个条件self.x_defect_width < 10是与缺口分开
                        cv2.drawContours(image, [box], 0, (0, 0, 205), 5)  # 这个位置是RGB
                        self.count_nums = self.count_nums + 1
                        self.rate1 = rate

                        # self.count_nums > 9是针对不连续的黑线，60 < self.x_defect_height < 200是针对连续的长条形黑线
                        if self.count_nums > 9 or 60 < self.x_defect_height < 200:

                            cv2img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
                            pilimg = Image.fromarray(cv2img)

                            # PIL图片上打印汉字
                            draw = ImageDraw.Draw(pilimg)  # 图片上打印
                            font = ImageFont.truetype("typefaces/simsun.ttf", 40, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
                            draw.text((int(self.middle_width_i), int(self.middle_height_i)), "黑线", (205, 0, 0),  # 这个位置是BGR
                                      font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体

                            # PIL图片转cv2 图片
                            image = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)

                            # 考虑一下多线程，让另一个线程来处理发出声音
                            self.flag = True

                            # 多线程调用另一个函数，这样就不会出现因为另一个函数执行时间而卡顿
                            r = threading.Thread(target=self.beef)
                            r.setDaemon(False)
                            r.start()

                            # break只让一帧里面出现一次标签
                            break

                    # 缺口特征提取
                    if 0.06 < rate < 0.14 and self.x_defect_width > 6:

                        if self.x_defect_width < 40 and 20 < self.x_defect_height < 40:
                            cv2.drawContours(image, [box], 0, (238, 44, 44), 5)  # 这个位置是RGB
                            cv2img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
                            pilimg = Image.fromarray(cv2img)
                            # PIL图片上打印汉字y
                            draw = ImageDraw.Draw(pilimg)  # 图片上打印
                            font = ImageFont.truetype("typefaces/simsun.ttf", 40, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
                            # draw.text((int(self.middle_width_i), int(self.x_defect_height)), "缺口", (44, 44, 238),
                            #           # 这个位置是BGR
                            #           font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
                            draw.text((int(self.middle_width_i), int(self.y_defect)), "缺口", (44, 44, 238),
                                      # 这个位置是BGR
                                      font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
                            # PIL图片转cv2 图片
                            image = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)

        if -89.9999 < self.theta_angle < -45.0:
            # 参考： https://blog.csdn.net/u013378642/article/details/81775131
            # sorted(self.list, key=(lambda x: x[0]))  # 倒序reverse=True

            # 参考：https://www.jb51.net/article/153239.htm
            self.list = sorted(self.list)

            # 参考：二维数组按指定行、列排序   https://blog.csdn.net/sinat_35821976/article/details/90518031
            # self.list = np.array(self.list, dtype=float)
            # # print('type', type(self.list))
            # self.list = self.list.astype(np.float)
            # self.list = self.list[np.argsort(self.list[:, 0])]

            # self.list = float(self.list)
            # print('self.list', self.list)
            for i in range(len(self.list)):
                # print('self.list1', self.list)
                # print('type(self.list)', type(self.list))
                rect_i = self.list[i]
                # print('rect_1', rect_i)
                # print('rect_i', rect_i)
                # print('rect_i[0][0]', rect_i[0][0])
                # 钢管边框的中心坐标
                self.middle_width_i = rect_i[0][0]
                self.middle_height_i = rect_i[0][1]
                # print('(self.middle_width, self.middle_height)', (self.middle_width, self.middle_height))
                # cx, cy = rect[0]  #中心位置
                # 获取矩形四个顶点，浮点型
                box_i = cv2.boxPoints(rect_i)
                # # 取整
                box_i = np.int0(box_i)
                # print('box_i', box_i)
                # # cv2.drawContours(image1, [box], 0, (0, 255, 0), 5)
                self.theta_angle = rect_i[2]
            #
                self.total_length = rect_i[1][1]
                # 在UI界面显示偏转角度
                str2 = str(self.theta_angle)
                self.textEdit_edit_angle.setText(str2)
                # 在UI界面显示钢管宽度,往右上偏时宽度为rect[1][1]
                str3 = str(self.total_length)
                self.textEdit_edit_width.setText(str3)

                if i == 0:
                    self.textEdit_edit_width1.setText(str3)
                if i == 1:
                    self.textEdit_edit_width2.setText(str3)
                if i == 2:
                    self.textEdit_edit_width3.setText(str3)

                (self.x_left_lower, self.y_left_lower) = (box_i[1][0], box_i[1][1])
                (self.x_left_upper, self.y_left_upper) = (box_i[2][0], box_i[2][1])
                (self.x_right_upper, self.y_right_upper) = (box_i[3][0], box_i[3][1])
                (self.x_right_lower, self.y_right_lower) = (box_i[0][0], box_i[0][1])

                # 左边这条直线
                k1 = (self.y_left_upper - self.y_left_lower) / (self.x_left_upper - self.x_left_lower)
                b1 = self.y_left_lower - k1 * self.x_left_lower

                # 右边这条直线
                k2 = (self.y_right_lower - self.y_right_upper) / (self.x_right_lower - self.x_right_upper)
                b2 = self.y_right_upper - k2 * self.x_right_upper
                self.k1_slope = k1
                self.k2_slope = k2
                self.b1_intercept = b1
                self.b2_intercept = b2
                # 这个是每根钢管表面的具体缺陷
                for c in range(len(contours)):
                    # 特征提取框的坐标以及长宽
                    self.x, self.y, self.w, self.h = cv2.boundingRect(contours[c])

                    # minAreaRect是计算最小面积，也即可以是斜着的矩形框
                    rect = cv2.minAreaRect(contours[c])
                    # cx, cy = rect[0]  #中心位置
                    # 获取矩形四个顶点，浮点型
                    box = cv2.boxPoints(rect)
                    # 取整
                    box = np.int0(box)

                    # rect[0][0], rect[0][1]表示矩形框的中心位置
                    (self.x_defect, self.y_defect) = (rect[0][0], rect[0][1])

                    (self.x_defect_height, self.x_defect_width) = (rect[1][0], rect[1][1])
                    # 钢管的缺陷宽度不可能大于钢管的宽度
                    if self.x_defect_width > self.total_length:
                        continue

                    # 根据点与直线的关系
                    # 当直线往左上方偏时，若kx-y+b>0,则点在直线右侧，若kx-y+b<0,则点在直线左侧，
                    # 当直线往右上方偏时，若kx-y+b<0,则点在直线右侧，若kx-y+b>0,则点在直线左侧，
                    # # 左边这条直线
                    self.linear_equation1 = self.k1_slope * self.x_defect - self.y_defect + self.b1_intercept
                    # 右边这条直线
                    self.linear_equation2 = self.k2_slope * self.x_defect - self.y_defect + self.b2_intercept

                    # 去除钢管边缘外最大的一个矩形框
                    if self.total_length > 200:
                        continue
                    # print('contours', len(contours))

                    # 钢管左边缘的那些误差缺陷最右边坐标小于钢管左边缘坐标
                    # if box[1][0] < x_left_upper or box[0][0] < x_left_lower or box[2][0] > x_right_upper or box[3][0] > x_right_lower:
                    if self.linear_equation2 < 0 or self.linear_equation1 > 0:
                        continue

                    # 缺陷中心到两直线的距离，每个缺陷有两个d
                    self.d1 = abs(self.k1_slope * self.x_defect - self.y_defect + self.b1_intercept) / (
                            (-1) * (-1) + self.k1_slope * self.k1_slope) ** 0.5
                    self.d2 = abs(self.k2_slope * self.x_defect - self.y_defect + self.b2_intercept) / (
                            (-1) * (-1) + self.k2_slope * self.k2_slope) ** 0.5
                    # # 缺陷到两边缘的最小距离
                    self.min_length = min(self.d1, self.d2)
                    # print('self.min_length', self.min_length)
                    rate = self.min_length / self.total_length


                    # 黑线特征提取
                    if 0.08 < rate < 0.12:  # 增加这个条件self.x_defect_width < 10是与缺口分开
                        if 2 < self.x_defect_width < 8 and 40 < self.x_defect_height < 200:
                            self.black_line = True
                            # self.count_nums = self.count_nums + 1
                            # self.rate1 = rate
                            #
                            # # self.count_nums > 9是针对不连续的黑线，60 < self.x_defect_height < 200是针对连续的长条形黑线
                            # if self.count_nums > 9 or 60 < self.x_defect_height < 200:
                            cv2.drawContours(image, [box], 0, (0, 0, 205), 5)  # 这个位置是RGB
                            cv2img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
                            pilimg = Image.fromarray(cv2img)

                            # PIL图片上打印汉字
                            draw = ImageDraw.Draw(pilimg)  # 图片上打印
                            font = ImageFont.truetype("typefaces/simsun.ttf", 40, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
                            draw.text((int(self.middle_width_i), int(self.middle_height_i)), "黑线", (205, 0, 0),  # 这个位置是BGR
                                      font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体

                            # PIL图片转cv2 图片
                            image = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)

                            # 考虑一下多线程，让另一个线程来处理发出声音
                            self.flag = True

                            # 分别显示第几根钢管的bool灯状态
                            if i == 0:
                                self.label1_6_1.setPixmap(QPixmap('images/light.png'))
                            if i == 1:
                                self.label2_6_1.setPixmap(QPixmap('images/light.png'))
                            if i == 2:
                                self.label3_6_1.setPixmap(QPixmap('images/light.png'))

                            # 多线程调用另一个函数，这样就不会出现因为另一个函数执行时间而卡顿
                            r = threading.Thread(target=self.beef)
                            r.setDaemon(False)
                            r.start()

                            # break只让一帧里面出现一次标签
                            break

                    # 缺口特征提取(如果有缺口，那么一定有黑线，反之不成立)
                        if self.black_line:
                            if 0.08 < rate < 0.11 and self.x_defect_width > 6:
                                if self.x_defect_width < 40 and 20 < self.x_defect_height < 80:
                                    cv2.drawContours(image, [box], 0, (238, 44, 44), 5)  # 这个位置是RGB
                                    cv2img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
                                    pilimg = Image.fromarray(cv2img)
                                    # PIL图片上打印汉字y
                                    draw = ImageDraw.Draw(pilimg)  # 图片上打印
                                    font = ImageFont.truetype("typefaces/simsun.ttf", 40, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
                                    # draw.text((int(self.middle_width_i), int(self.x_defect_height)), "缺口", (44, 44, 238),
                                    #           # 这个位置是BGR
                                    #           font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
                                    draw.text((int(self.middle_width_i), int(self.y_defect)), "缺口", (44, 44, 238),
                                              # 这个位置是BGR
                                              font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
                                    # PIL图片转cv2 图片
                                    image = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)

                                    # 分别显示第几根钢管的bool灯状态
                                    if i == 0:
                                        self.label1_6_2.setPixmap(QPixmap('images/light.png'))
                                    if i == 1:
                                        self.label2_6_2.setPixmap(QPixmap('images/light.png'))
                                    if i == 2:
                                        self.label3_6_2.setPixmap(QPixmap('images/light.png'))
       # self.flag = True
       #  time.sleep(0.5)
        end = time.clock()  # 记下结束时刻
        print("end - start", end - start)

        # Qt显示图片时，需要先转换成QImgage类型，同时也是在UI界面显示视频画面   参考于mainForm_run.py
        # 钢管表面原图
        showImage1 = QtGui.QImage(image, image.shape[1], image.shape[0], image.shape[1] * 3,
                                 QtGui.QImage.Format_RGB888)
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(showImage1))
        # 钢管表面阈值分割图
        showImage2 = QtGui.QImage(thresh, thresh.shape[1], thresh.shape[0], thresh.shape[1],
                                  QtGui.QImage.Format_Indexed8)
        self.thresh1_label.setPixmap(QtGui.QPixmap.fromImage(showImage2))
        # 钢管侧边管口原图
        showImage3 = QtGui.QImage(imagePipe, imagePipe.shape[1], imagePipe.shape[0], imagePipe.shape[1] * 3,
                                  QtGui.QImage.Format_RGB888)
        self.pip_orgin_label.setPixmap(QtGui.QPixmap.fromImage(showImage3))
        # 钢管侧边管口阈值分割图
        showImage4 = QtGui.QImage(threshPipe, threshPipe.shape[1], threshPipe.shape[0], threshPipe.shape[1],
                                  QtGui.QImage.Format_Indexed8)
        self.thresh_label.setPixmap(QtGui.QPixmap.fromImage(showImage4))

    def beef(self):

        # print('flag', self.flag)
        if self.flag:
            winsound.Beep(self.freq, self.duration)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = mainWindow()
    win.show()
    sys.exit(app.exec())