from PyQt5 import QtCore 
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from PyQt5.QtMultimedia import QSound 
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from opencv_engine import opencv_engine
import time
import moviepy.editor
from tkinter.filedialog import *
from Ui_main import Ui_MainWindow # Ui_main 為自行設計的介面程式
# videoplayer_state_dict = {
#  "stop":0,   
#  "play":1,
#  "pause":2     
# }

class video_controller(object):
    def __init__(self, video_path, ui):
        super().__init__()
        self.video_path = video_path
        self.ui = ui
        self.qpixmap_fix_width = 800 # 16x9 = 1920x1080 = 1280x720 = 800x450
        self.qpixmap_fix_height = 450
        self.current_frame_no = 0
        self.videoplayer_state = "stop"
        self.init_video_info()
        self.set_video_player()

    def init_video_info(self):
        videoinfo = opencv_engine.getvideoinfo(self.video_path)
        
        self.vc = videoinfo["vc"] 
        self.video_fps = videoinfo["fps"] 
        self.video_total_frame_count = videoinfo["frame_count"] 
        self.video_width = videoinfo["width"]
        self.video_height = videoinfo["height"] 

    def set_video_player(self):
        self.timer=QTimer() # init QTimer
        self.timer.timeout.connect(self.timer_timeout_job) # when timeout, do run one
        print(self.video_fps)
        self.timer.start(32) # start Timer, here we set '1000ms//Nfps' while timeout one time
        #self.timer.start() # but if CPU can not decode as fast as fps, we set 1 (need decode time)

        

    def set_current_frame_no(self, frame_no):
        self.vc.set(1, frame_no) # bottleneck

    def __get_next_frame(self):
        ret, frame = self.vc.read()
        #self.ui.label_framecnt.setText(f"frame number: {self.current_frame_no}/{self.video_total_frame_count}")
        #self.setslidervalue(self.current_frame_no)
        return frame

    def __update_label_frame(self, frame):       
        bytesPerline = 3 * self.video_width
        qimg = QImage(frame, self.video_width, self.video_height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.qpixmap = QPixmap.fromImage(qimg)

        if self.qpixmap.width()/16 >= self.qpixmap.height()/9: # like 1600/16 > 90/9, height is shorter, align width
            self.qpixmap = self.qpixmap.scaledToWidth(self.qpixmap_fix_width)
        else: # like 1600/16 < 9000/9, width is shorter, align height
            self.qpixmap = self.qpixmap.scaledToHeight(self.qpixmap_fix_height)
        self.ui.Film.setPixmap(self.qpixmap)
        # self.ui.label_videoframe.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop) # up and left
        self.ui.Film.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter) # Center
        # self.ui.slider_videoframe
        
    def play(self):
        self.videoplayer_state = "play"
        
    def stop(self):
        self.videoplayer_state = "stop"

    def pause(self):
        self.videoplayer_state = "pause"
        
    def timer_timeout_job(self):
        
        if (self.videoplayer_state == "play"):
            
            if self.current_frame_no >= self.video_total_frame_count-1:
                #self.videoplayer_state = "pause"
                self.current_frame_no = 0 # auto replay
                
                self.set_current_frame_no(self.current_frame_no)
                
            #else:
            #   self.current_frame_no += 1

        if (self.videoplayer_state == "stop"):
            self.current_frame_no = 0
            self.set_current_frame_no(self.current_frame_no)

        if (self.videoplayer_state == "pause"):
            self.current_frame_no = self.current_frame_no
            self.set_current_frame_no(self.current_frame_no)

        frame = self.__get_next_frame()
        self.__update_label_frame(frame)
