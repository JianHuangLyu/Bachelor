from unittest import result
import cv2
import sys, time, os
import numpy as np
import time
import keyboard
import pandas as pd

# video mp4 轉換 音訊之套件
import moviepy.editor
from tkinter.filedialog import *
###################################

from PyQt5 import QtCore, QtGui, QtWidgets # 引入 PyQt5 模組
from Ui_main import Ui_MainWindow # Ui_main 為自行設計的介面程式
from PyQt5.QtWidgets import QMainWindow, QFileDialog
#from face_mesh import webcam

#face_mesh
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

from PIL import Image

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.autograd import Variable
#from data.mydataset import MyDataset,  get_sub_num
from model.mymodel import FeatRef
import argparse

from torchvision import transforms
from sklearn import preprocessing

from video_controller import video_controller

from PyQt5.QtMultimedia import QSound
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout


parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('--dataset', default='orignal flow ', help='dataset setting')
parser.add_argument('--path', default='data/database/', help='dataset path')
parser.add_argument('--root_path', default='', help='project root path')
parser.add_argument('--result_path', default='result/', help='result path')
parser.add_argument('--result_metrics', default='metric_samm.csv', help='result metrics file name')
parser.add_argument('--ev_dim', default=1024, type=int, metavar='N',
                    help='dimension of feature')
parser.add_argument('--width', default=28, type=int, metavar='N',
                    help='sample width')
parser.add_argument('--class_num', default=4, type=int, metavar='N',
                    help='number of total classes')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run') 
parser.add_argument('--train_round', default=1, type=int, metavar='N',
                    help='number of total train round')
parser.add_argument('--batch_size', default=4, type=int, metavar='N',
                    help='number of batch size')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--lambada', default=0.9, type=float, metavar='N',
                    help='weight for rois loss')

args = parser.parse_args()

# 最上下左右的臉孔點
crop_size = [10,152,234,454]
save_path='/image'
apex_save_path='/apex_frame'
optical_flow_save_path = '/optical_flow'

subjects ='./sub01'
if not os.path.isdir(subjects):
    os.mkdir(subjects)
    os.mkdir(subjects + save_path)
    os.mkdir(subjects + apex_save_path)
    os.mkdir(subjects + optical_flow_save_path)


tmp_flow_list = []
apex_record_list = [] #記錄APEX編號

p = 0
n = 0
sur = 0
o = 0

sec_ = []
p_ = []
n_ = []
sur_ = []
o_ =  []
sec_tmp = []
tmp = []
tmp1 = []
tmp2 = []
tmp3 = []



emotion=[0,0,0,0]
transform1 = transforms.Compose([
	# transforms.CenterCrop((WIDTH,HEIGHT)), # 只能对PIL图片进行裁剪
	transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
	]
)
transform2 = transforms.Compose([
	transforms.ToTensor(),
	]
)
def compute_TVL1(prev, curr, bound=15):
        """Compute the TV-L1 optical flow."""
        TVL1=cv2.optflow.DualTVL1OpticalFlow_create()
        # TVL1 = cv2.DualTVL1OpticalFlow_create()
        # TVL1=cv2.createOptFlow_DualTVL1()
        flow = TVL1.calc(prev, curr, None)
        assert flow.dtype == np.float32
    
        flow = (flow + bound) * (255.0 / (2 * bound))
        flow = np.round(flow).astype(np.uint8)
        flow[flow >= 255] = 255
        flow[flow <= 0] = 0
    
        return flow

class Camera(QtCore.QThread, QtWidgets.QMainWindow, Ui_MainWindow): # 繼承 QtCore.QThread 來建立 Camera 類別
    rawdata = QtCore.pyqtSignal(np.ndarray) # 建立傳遞信號，需設定傳遞型態為 np.ndarray


    def __init__(self, parent=None):
        """ 初始化
            - 執行 QtCore.QThread 的初始化
            - 建立 cv2 的 VideoCapture 物件
            - 設定屬性來確認狀態
              - self.connect：連接狀態
              - self.running：讀取狀態
        """
        super().__init__(parent)# 將父類初始化
        self.cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)# 建立 cv2 的攝影機物件
        self.threshold = 40 # 臨界點
        if self.cam is None or not self.cam.isOpened():# 判斷攝影機是否正常連接
            self.connect = False
            self.running = False
        else:
            self.connect = True
            self.running = False

    def run(self):
        """ 執行多執行緒
            - 讀取影像
            - 發送影像
            - 簡易異常處理
            - face_mesh webcam
            - face_mesh images crop
            - Apex Frame detection
            - 偵測鍵盤行為
            - load model
        """

        #device = torch.device("cuda")
        model = FeatRef(args.class_num, args).float()
        #model.to(device)
        model.load_state_dict(torch.load('model_FULL.pth',map_location='cpu'))
        model.eval()

        i=0
        second = 0
        frame_num = 0
        s = (28,28,3)
        zero = np.zeros(s)
        start = time.time()# 計時開始
        global apex_frame
        while True:
            t1 = time.time()# 計時一秒開始
            while self.running and self.connect:# 當正常連接攝影機才能進入迴圈
                ret, img = self.cam.read()# 讀取影像
                if ret:
                    #self.rawdata.emit(img)# 發送影像

                    '''face_mesh webcam'''
                    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
                    with mp_face_mesh.FaceMesh(
                        static_image_mode=True,
                        max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as face_mesh:
                        
                        img.flags.writeable = False
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        results = face_mesh.process(img)
                        
                        # Draw the face mesh annotations on the image.
                        img.flags.writeable = True
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        # 人臉框繪製
                        '''
                        if results.multi_face_landmarks:
                            for face_landmarks in results.multi_face_landmarks:
                                mp_drawing.draw_landmarks(
                                    image=img,
                                    landmark_list=face_landmarks,
                                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                                    landmark_drawing_spec=None,
                                    connection_drawing_spec=mp_drawing_styles
                                    .get_default_face_mesh_tesselation_style())
                                mp_drawing.draw_landmarks(
                                    image=img,
                                    landmark_list=face_landmarks,
                                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                                    landmark_drawing_spec=None,
                                    connection_drawing_spec=mp_drawing_styles
                                    .get_default_face_mesh_contours_style())
                                mp_drawing.draw_landmarks(
                                    image=img,
                                    landmark_list=face_landmarks,
                                    connections=mp_face_mesh.FACEMESH_IRISES,
                                    landmark_drawing_spec=None,
                                    connection_drawing_spec=mp_drawing_styles
                                    .get_default_face_mesh_iris_connections_style())
                        '''
                        self.rawdata.emit(img) # 發送影像

                        '''face_mesh images crop'''
                        h, w, d = img.shape
                        # Convert the BGR image to RGB before processing.
                        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                        # Print and draw face mesh landmarks on the image.
                        if not results.multi_face_landmarks:
                            continue
                        #annotated_image = img.copy()

                        for face_landmarks in results.multi_face_landmarks:
                            #print('face_landmarks:', face_landmarks)
                            y1=int(face_landmarks.landmark[10].y*h)
                            y2=int(face_landmarks.landmark[152].y*h)
                            x1=int(face_landmarks.landmark[234].x*w)
                            x2=int(face_landmarks.landmark[454].x*w)    
                            cropped_image = img[y1:y2, x1:x2]
                            cv2.imwrite(subjects + save_path + '/' + str(i) + ".jpg",cropped_image)
                        print(i)

                        '''Apex Frame detection''' 
                        resize_image = cv2.resize(cropped_image, (28,28), interpolation=cv2.INTER_CUBIC)
                        if not frame_num:
                            onset_frame = resize_image #onset_frame
                            previous = cv2.cvtColor(resize_image, cv2.COLOR_BGR2GRAY) # 由於第一幀圖片沒有前一幀
                            onset_frame_tmp=previous
                            max_diff = cv2.resize(zero, (28,28), interpolation=cv2.INTER_CUBIC)
                        gray_a = cv2.cvtColor(resize_image, cv2.COLOR_BGR2GRAY)  # 轉換成灰度圖
                        gray = cv2.absdiff(gray_a, previous)  # 計算絕對值差
                        #gray = cv2.medianBlur(gray, 3)  # 中值濾波
                            
                        if gray.sum() > max_diff.sum():
                            max_img = i
                            max_diff=gray 
                            apex_frame = resize_image #apex_frame
                            apex_frame_tmp=gray_a
                            print(str(max_img)+'.jpg is Apex Frame')
                            cv2.imwrite(subjects + apex_save_path + '/' + str(max_img) + ".jpg",cropped_image)
                            tmp_flow = compute_TVL1(onset_frame_tmp, apex_frame_tmp)
                            tmp_flow_list.append(tmp_flow)
                            apex_record_list.append(max_img)

                        for x, video_flow in enumerate(tmp_flow_list):
                            flow_x=optical_flow_save_path + '/flow_x_' + str(apex_record_list[x]) + '.jpg'
                            cv2.imwrite(subjects + flow_x,video_flow[:, :, 0]) #x
                            flow_y=optical_flow_save_path + '/flow_y_' + str(apex_record_list[x]) + '.jpg'
                            cv2.imwrite(subjects + flow_y,video_flow[:, :, 1]) #y


                        '''偵測鍵盤行為'''
                        if keyboard.is_pressed("z"): # 按z計時
                            end = time.time() # 計時結束
                            print(end-start) 



                        t2 = time.time()# 計時一秒結束
                        
                        if t2-t1 > 1:
                            t1=t2
                            max_diff = cv2.resize(zero, (28,28), interpolation=cv2.INTER_CUBIC)
                            second += 1
                            print('-----第'+str(second)+'秒-----')
                            sec_tmp.append('第'+str(second)+'秒')
                            
                            if second == 1:
                                onset_frame=onset_frame
                                onset_frame = transform1(onset_frame)
                                onset_frame=torch.tensor(onset_frame)
                                onset_frame=onset_frame.view(3,28,28)
                                onset_frame=onset_frame.unsqueeze(0)

              
                            apex_frame = transform1(apex_frame)
                            apex_frame=torch.tensor(apex_frame)
                            apex_frame=apex_frame.view(3,28,28)
                            apex_frame=apex_frame.unsqueeze(0)

                            opticalflow_x = torch.tensor(video_flow[:, :, 0])
                            opticalflow_x=opticalflow_x.unsqueeze(2)
                            opticalflow_x=opticalflow_x.repeat(1,1,3)
                            opticalflow_x=transform1(opticalflow_x.numpy())
                            opticalflow_x=opticalflow_x.unsqueeze(0)
                            
                            opticalflow_y = torch.tensor(video_flow[:, :, 1])
                            opticalflow_y=opticalflow_y.unsqueeze(2)
                            opticalflow_y=opticalflow_y.repeat(1,1,3)
                            opticalflow_y = transform1(opticalflow_y.numpy())
                            opticalflow_y=opticalflow_y.unsqueeze(0)
    

                            input=torch.cat((opticalflow_x,opticalflow_y,onset_frame,apex_frame),0)
                            input=input.unsqueeze(0)
                            result = model(input)
                            softmax = nn.Softmax(dim=1)
                            result = softmax(result)
                            
                            emotion[0]=result[:, 0].item()
                            emotion[1]=result[:, 1].item()
                            emotion[2]=result[:, 2].item()
                            emotion[3]=result[:, 3].item()

                            global p,n,o,sur

                            win.print_positive(str(emotion[0]*100)[:4] + "%")
                            tmp.append(str(emotion[0]*100)[:4] + "%")
                            p = float(p) + float(str(emotion[0]*100)[:4])
                            
                            win.print_negative(str(emotion[1]*100)[:4] + "%")
                            tmp1.append(str(emotion[1]*100)[:4] + "%")
                            n = float(n) + float(str(emotion[1]*100)[:4])

                            win.print_surprise(str(emotion[2]*100)[:4] + "%")
                            tmp2.append(str(emotion[2]*100)[:4] + "%")
                            sur = float(sur) + float(str(emotion[2]*100)[:4])

                            win.print_other(str(emotion[3]*100)[:4] + "%")
                            tmp3.append(str(emotion[3]*100)[:4] + "%")
                            o = float(o) + float(str(emotion[3]*100)[:4])

                            '''
                            print(result[:, 0].item())
                            print(result[:, 1].item())
                            print(result[:, 2].item())
                            print(result[:, 3].item())
                            '''

                        i += 1
                        frame_num += 1

                        #寫入excel
                        dtime = pd.DataFrame({'': sec_tmp})
                        df = pd.DataFrame({'Positive': tmp})
                        df1 = pd.DataFrame({'Negative': tmp1})
                        df2 = pd.DataFrame({'Surprise': tmp2})
                        df3 = pd.DataFrame({'Other': tmp3})
                        writer = pd.ExcelWriter(subjects + '/FinalResult.xlsx', engine='xlsxwriter')
                        dtime.to_excel(writer, sheet_name='FinalResult' ,index = False)
                        df.to_excel(writer, sheet_name='FinalResult' , startcol=1 ,index = False)
                        df1.to_excel(writer, sheet_name='FinalResult', startcol=2 ,index= False)
                        df2.to_excel(writer, sheet_name='FinalResult', startcol=3 ,index= False)
                        df3.to_excel(writer, sheet_name='FinalResult', startcol=4 ,index= False)

                else:# 例外處理
                    print("Warning!!!")
                    self.connect = False
            if(second > 0):
                p = p / float(second)
                n = n / float(second)
                sur = sur / float(second)
                o = o / float(second)
            sec_.append(str(second) + '秒平均')
            p_.append(str(p)[:4]+'%')
            n_.append(str(n)[:4]+'%')
            sur_.append(str(sur)[:4]+'%')
            o_.append(str(o)[:4]+'%')
            davg_acc = pd.DataFrame({'': sec_})
            davg_p_acc = pd.DataFrame({'Positive':p_})
            davg_n_acc = pd.DataFrame({'Negative':n_})
            davg_s_acc = pd.DataFrame({'Surprise':sur_})
            davg_o_acc = pd.DataFrame({'Other': o_})
            
            #davg_acc.to_excel(writer1, sheet_name='FinalResult' , index = False)
            davg_acc.to_excel (writer, sheet_name='avg_acc' , index = False)
            davg_p_acc.to_excel(writer, sheet_name='avg_acc' , startcol=1 , index = False)
            davg_n_acc.to_excel(writer, sheet_name='avg_acc' , startcol=2 , index = False)
            davg_s_acc.to_excel(writer, sheet_name='avg_acc' , startcol=3 , index = False)
            davg_o_acc.to_excel(writer, sheet_name='avg_acc' , startcol=4 , index = False)
            writer.save()


    def open(self):
        """ 開啟攝影機影像讀取功能 """
        if self.connect:
            self.running = True# 啟動讀取狀態

    def stop(self):
        """ 暫停攝影機影像讀取功能 """
        if self.connect:
            self.running = False# 關閉讀取狀態

    def close(self):
        """ 關閉攝影機功能 """
        if self.connect:
            self.running = False# 關閉讀取狀態
            time.sleep(1)
            self.cam.release()# 釋放攝影機

# 繼承 QtWidgets.QMainWindow, Ui_MainWindow 來建立 MainWindow 類別
class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):    
    def __init__(self, parent=None):
        """ 初始化
            - 物件配置
            - 相關屬性配置
        """
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        #self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.viewData.setScaledContents(True)
        #self.ui = Ui_MainWindow()
        self.setup_control()

        # view 屬性設置，為了能讓滑鼠進行操控，
        self.view_x = self.view.horizontalScrollBar()
        self.view_y = self.view.verticalScrollBar()
        self.view.installEventFilter(self)
        self.last_move_x = 0
        self.last_move_y = 0

        # 設定 Frame Rate 的參數
        self.frame_num = 0
        
        # 設定相機功能
        self.ProcessCam = Camera()# 建立相機物件
        if self.ProcessCam.connect:
            self.debugBar('Connection!!!')
            # 連接影像訊號 (rawdata) 至 getRaw()
            self.ProcessCam.rawdata.connect(self.getRaw)# 槽功能：取得並顯示影像
        else:
            self.debugBar('Disconnection!!!')
        # 連接按鍵
        self.camBtn_open.clicked.connect(self.openCam)# 槽功能：開啟攝影機
        self.camBtn_stop.clicked.connect(self.stopCam)# 槽功能：暫停讀取影像


        #self.h_layout = QHBoxLayout()
        #self.h_layout.addWidget(self.camBtn_open)
        #self.h_layout.addWidget(self.camBtn_open)
        #self.setLayout(self.h_layout)
    
    def setup_control(self):
        self.file_openfile.clicked.connect(self.open_file)        

    def open_file(self):
        filename, filetype = QFileDialog.getOpenFileName(self, "Open file Window", "./", "Video Files(*.mp4 *.avi *webm)") # start path  

        #音檔轉換參考: https://www.youtube.com/watch?v=N0c2rYm-92w
        self.video_path = filename

        if not os.path.exists(self.video_path.split('.')[0] + ".wav"): #音檔不存在才轉檔
            the_video = moviepy.editor.VideoFileClip(self.video_path)
            the_audio = the_video.audio
            the_audio.write_audiofile(self.video_path.split('.')[0] + ".wav")
   
       
        self.video_controller = video_controller(video_path=self.video_path,
                                                 ui=self)
        #self.ui.label_filepath.setText(f"video path: {self.video_path}")
        self.sound = QSound(self.video_path.split('.')[0] + ".wav") #讀取聲音檔
        

        #self.sound.setLoops(QSound.Infinite)   #聲音檔迴圈

        #self.play_btn = QPushButton('Play Sound', self)
        #self.stop_btn = QPushButton('Stop Sound', self)
        
        self.camBtn_open.clicked.connect(self.sound.play)
        self.camBtn_pause.clicked.connect(self.sound.stop)
        self.camBtn_stop.clicked.connect(self.sound.stop)

        self.camBtn_open.clicked.connect(self.video_controller.play) # connect to function()
        self.camBtn_stop.clicked.connect(self.video_controller.stop)
        self.camBtn_pause.clicked.connect(self.video_controller.pause)
        

    def print_positive(self, msg):
        """ 列印positive """
        self.ans_positive.setText(msg)
    
    def print_negative(self, msg):
        """ 列印negative """
        self.ans_negative.setText(msg)
    
    def print_surprise(self, msg):
        """ 列印surprise """
        self.ans_surprise.setText(msg)
    
    def print_other(self, msg):
        """ 列印other """
        self.ans_other.setText(msg)
    
    def getRaw(self, data):# data 為接收到的影像
        """ 取得影像 """
        self.showData(data)# 將影像傳入至 showData()
        
    def openCam(self):
        """ 啟動攝影機的影像讀取 """
        if self.ProcessCam.connect:# 判斷攝影機是否可用
            self.ProcessCam.open()# 影像讀取功能開啟
            self.ProcessCam.start()# 在子緒啟動影像讀取
            # 按鈕的狀態：啟動 OFF、暫停 ON、視窗大小 ON
            self.camBtn_open.setEnabled(False)
            self.camBtn_stop.setEnabled(True)
            self.viewCbo_roi.setEnabled(True)

    def stopCam(self):
        """ 凍結攝影機的影像 """
        if self.ProcessCam.connect:# 判斷攝影機是否可用
            self.ProcessCam.stop()# 影像讀取功能關閉
            # 按鈕的狀態：啟動 ON、暫停 OFF、視窗大小 OFF
            self.camBtn_open.setEnabled(True)
            self.camBtn_stop.setEnabled(False)
            self.viewCbo_roi.setEnabled(False)

    def showData(self, img):
        """ 顯示攝影機的影像 """
        self.Ny, self.Nx, _ = img.shape# 取得影像尺寸

        # 反轉顏色
        img_new = np.zeros_like(img)
        img_new[...,0] = img[...,2]
        img_new[...,1] = img[...,1]
        img_new[...,2] = img[...,0]
        img = img_new

        # 建立 Qimage 物件 (灰階格式)
        # qimg = QtGui.QImage(img[:,:,0].copy().data, self.Nx, self.Ny, QtGui.QImage.Format_Indexed8)
        
        # 建立 Qimage 物件 (RGB格式)
        qimg = QtGui.QImage(img.data, self.Nx, self.Ny, QtGui.QImage.Format_RGB888)
        # viewData 的顯示設定
        self.viewData.setScaledContents(True)# 尺度可變
        ### 將 Qimage 物件設置到 viewData 上
        self.viewData.setPixmap(QtGui.QPixmap.fromImage(qimg))
        ### 顯示大小設定
        if self.viewCbo_roi.currentIndex() == 0: roi_rate = 0.5
        elif self.viewCbo_roi.currentIndex() == 1: roi_rate = 0.75
        elif self.viewCbo_roi.currentIndex() == 2: roi_rate = 1
        elif self.viewCbo_roi.currentIndex() == 3: roi_rate = 1.25
        elif self.viewCbo_roi.currentIndex() == 4: roi_rate = 1.5
        else: pass
        self.viewForm.setMinimumSize(self.Nx*roi_rate, self.Ny*roi_rate)
        self.viewForm.setMaximumSize(self.Nx*roi_rate, self.Ny*roi_rate)
        self.viewData.setMinimumSize(self.Nx*roi_rate, self.Ny*roi_rate)
        self.viewData.setMaximumSize(self.Nx*roi_rate, self.Ny*roi_rate)
        
        # Frame Rate 計算並顯示到狀態欄上
        if self.frame_num == 0:
            self.time_start = time.time()
        if self.frame_num >= 0:
            self.frame_num += 1
            self.t_total = time.time() - self.time_start
            if self.frame_num % 100 == 0:
                self.frame_rate = float(self.frame_num) / self.t_total
                self.debugBar('FPS: %0.3f frames/sec' % self.frame_rate)# 顯示到狀態欄

    def eventFilter(self, source, event):
        """ 事件過濾 (找到對應物件並定義滑鼠動作) """
        if source == self.view:# 找到 view 來源
            if event.type() == QtCore.QEvent.MouseMove:# 定義滑鼠點擊移動動作
                # 找到滑鼠移動位置
                if self.last_move_x == 0 or self.last_move_y == 0:
                    self.last_move_x = event.pos().x()
                    self.last_move_y = event.pos().y()
                # 計算滑鼠移動量
                distance_x = self.last_move_x - event.pos().x()
                distance_y = self.last_move_y - event.pos().y()
                # 設置 view 的視窗移動
                self.view_x.setValue(self.view_x.value() + distance_x)
                self.view_y.setValue(self.view_y.value() + distance_y)
                # 儲存滑鼠最後移動的位置
                self.last_move_x = event.pos().x()
                self.last_move_y = event.pos().y()
            elif event.type() == QtCore.QEvent.MouseButtonRelease:# 定義滑鼠放開動作
                # 滑鼠放開過後，最後位置重置
                self.last_move_x = 0
                self.last_move_y = 0
            return QtWidgets.QWidget.eventFilter(self, source, event)
    
    def closeEvent(self, event):
        """ 視窗應用程式關閉事件 """
        if self.ProcessCam.running:
            self.ProcessCam.close()# 關閉攝影機
            self.ProcessCam.terminate()# 關閉子緒
        QtWidgets.QApplication.closeAllWindows()# 關閉所有視窗
        
    def keyPressEvent(self, event):
        """ 鍵盤事件 """
        if event.key() == QtCore.Qt.Key_Q:# 偵測是否按下鍵盤 Q
            if self.ProcessCam.running:
                self.ProcessCam.close()# 關閉攝影機
                time.sleep(1)
                self.ProcessCam.terminate()# 關閉子緒
            QtWidgets.QApplication.closeAllWindows()# 關閉所有視窗

    def debugBar(self, msg):
        self.statusBar.showMessage(str(msg), 5000)# 在狀態列顯示字串資訊


if __name__=='__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
