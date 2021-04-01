# Steel-Detect-Detection
#### 简介  
##### 这是一个与企业合作的项目，是一套用于检测高反光金属表面若干缺陷检测的系统，主要开发语言是Python。涉及的主要方法有  
1、选择合适的光源--同轴光源，用于有效抑制外界环境光源干扰以及如实反映金属表面的纹理特征  
2、基于pyqt5设计了图形化界面实时反映钢管表面特征以及实时处理的画面  
3、综合传统的图像处理以及深度学习（基于yolov5）方法来识别缺陷并分类  
4、基于socket通信方式将采集到的缺陷上传至服务端用于数据分析处理（还未完全实现） 
5、钢印的识别基于百度开源的PaddleOCR模型，效果不错，使用GPU的前提下，处理时间随分辨率的提升而增加  
6、表面的凹坑等一些不太好用的传统图像处理的缺陷，使用YOLOV5，用两跟带有凹坑缺陷的钢管的七个面作为样本，另一个面作为测试，实时状态下基本能把明显的凹坑识别出来  
#### 安装
1、直接通过anaconda安装好常用的包，安装路径https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/ ，选择Anaconda3-5.2.0这个版本，自带python3.6.5版本   
2、在pycharm中配置好python环境，课参考https://blog.csdn.net/qq_18424081/article/details/85856713  
3、除此以外还需安装pyqt5以及opencv-python，其中opencv-python安装3.x版本，因为3.x版本的cv2.findContours这个接口返回的是3个参数，4.x的版本返回2个参数，否则会报错  
binary1, contours1, hierarchy1 = cv2.findContours(thresh1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)   
4、可能出现的版本问题：numpy和Pillow这两个包版本太低，卸载安装靠近最新的版本  
5、为自定义设计前端页面，可配置Qt Designer，课参考https://blog.csdn.net/weixin_42512684/article/details/104099351  
6、或者你可以不安装anaconda，选择安装项目目录下的requirement.txt，安装命令pip install -r requirements.txt或者pip3 install -r requirements.txt  
#### 注意点
这个项目使用的摄像头是大恒水星系列的相机，需要在其官网下载对应的驱动，且同时运行两个相机，不然会报错，若使用一个，则需修改一下  
若使用普通的免驱动的USB摄像头，可直接通过OpenCV接口调用  
#### 效果
说明：  
1、左边一个相机是用于检测表明若干缺陷，其中左上图是原始图像，一些标注信息显示在这里；左下图是自适应阈值二值化后的图像，后续的算法就是基于二值图像进行处理的  
2、中间一列是一些可视化缺陷的示意图以及下方可手动调节算法参数的按钮，都是实时效果的，就是根据环境因素实时调节以达到较好的效果，这里的按钮所代表的参数没有全部列举出来，还有一些是初始化设置的，可根据 需要增删，它们是基于pyqt5的信号槽的思想实现的  
3、右边一个相机是用于检测侧边切口是否方正，主要是基于canny算子提取轮廓之后再通过膨胀操作，使不连续的区域尽可能连接起来，然后再通过OpenCV接口approxPolyDP进行多边形拟合，根据拟合后的多边形形状来判断切口是否方正  
![](https://github.com/optics915/Steel-Detect-Detection/blob/master/test_result.png)
