import time
import cv2
import numpy as np
import sys
import os
import argparse
from random import sample
#Landmark load model
import yaml
#PyQt5
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot, QTimer, QThread, QSize, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QIcon, QBrush
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog, QGraphicsOpacityEffect
from PyQt5 import QtGui
import Qt_design as ui
from threading import Thread
#torch
import torchconda
import torchvision
#image transform
import skimage
import imageio
from skimage import img_as_ubyte
from model.Facedetection.config import device
#face detection
from model.Facedetection.utils import align_face, get_face_all_attributes, draw_bboxes
from model.Facedetection.RetinaFace.RetinaFaceDetection import retina_face
#Featrue extraction
import model.Emotion.lie_emotion_process as emotion
import model.action_v4_L12_BCE_MLSM.lie_action_process as action
from model.action_v4_L12_BCE_MLSM.config import Config
#Landmark
# from model.Landmark.TDDFA import TDDFA
# from model.Landmark.utils.render import render
# from model.Landmark.utils.functions import cv_draw_landmark, get_suffix
# save model
from joblib import dump, load

parser = argparse.ArgumentParser()
#Retina
parser.add_argument('--len_cut', default=30, type=int, help= '# of frames you want to pred')
parser.add_argument('-m', '--trained_model', default='./model/Facedetection/RetinaFace/weights/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=3000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=3, type=int, help='keep_top_k')
parser.add_argument('--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
parser.add_argument('--gpu_num', default= "0", type=str, help='GPU number')
#Landmark
parser.add_argument('-c', '--config', type=str, default='./model/Landmark/configs/mb1_120x120.yml')
parser.add_argument('--mode', default='gpu', type=str, help='gpu or cpu mode')
parser.add_argument('-o', '--opt', type=str, default='2d', choices=['2d', '3d'])
# Emotion
parser.add_argument('--at_type', '--attention', default=1, type=int, metavar='N',help= '0 is self-attention; 1 is self + relation-attention')
parser.add_argument('--preTrain_path', '-pret', default='./model/Emotion/model112/self_relation-attention_AFEW_better_46.0733_41.2759_12.tar', type=str, help='pre-training model path')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num

#load model
Retina = retina_face(crop_size = 224, args = args) # Face detection
Emotion_class = emotion.Emotion_FAN(args = args)
Action_class = action.Action_Resnet(args= Config())
SVM_model = load('./model/SVM_model/se_res50+EU/split_svc_acc0.720_AUC0.828.joblib')
print('model is loaded')
class Landmark:
    def __init__(self,im,bbox,cfg,TDDFA,color):
        self.cfg = cfg
        self.tddfa = TDDFA
        self.boxes = bbox
        self.image = im
        self.color = color
        
    def main(self,index):
        dense_flag = args.opt in ('3d',)
        pre_ver = None
        self.boxes = [self.boxes[index]]
        param_lst, roi_box_lst = self.tddfa(self.image, self.boxes)
        ver = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
        # refine
        param_lst, roi_box_lst = self.tddfa(self.image, [ver], crop_policy='landmark')
        ver = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
        pre_ver = ver  # for tracking

        if args.opt == '2d':
            res = cv_draw_landmark(self.image, ver,color=self.color)
        elif args.opt == '3d':
            res = render(self.image, [ver])
        else:
            raise Exception(f'Unknown opt {args.opt}')
        
        lnd = ver.T
        # D1_i = np.sqrt(np.square(lnd[61][0]-lnd[67][0]) + np.square(lnd[61][1]-lnd[67][1]))
        # D1_o = np.sqrt(np.square(lnd[50][0]-lnd[58][0]) + np.square(lnd[50][1]-lnd[58][1]))
        # D2_i = np.sqrt(np.square(lnd[62][0]-lnd[66][0]) + np.square(lnd[62][1]-lnd[66][1]))
        # D2_o = np.sqrt(np.square(lnd[51][0]-lnd[57][0]) + np.square(lnd[51][1]-lnd[57][1]))
        # D3_i = np.sqrt(np.square(lnd[63][0]-lnd[65][0]) + np.square(lnd[63][1]-lnd[65][1]))
        # D3_o = np.sqrt(np.square(lnd[52][0]-lnd[56][0]) + np.square(lnd[52][1]-lnd[56][1]))
        res = res[int(roi_box_lst[0][1]):int(roi_box_lst[0][3]), int(roi_box_lst[0][0]):int(roi_box_lst[0][2])]
        # pm_ratio_1 = D1_i / D1_o
        # pm_ratio_2 = D2_i / D2_o
        # pm_ratio_3 = D3_i / D3_o
        # print('pm1:',pm_ratio_1)
        # print('pm2:',pm_ratio_2)
        # print('pm3:',pm_ratio_3)
        if res.shape[0] != 0 and res.shape[1] != 0:
            img_res = cv2.resize(res,(224,224))
        else:
            img_res = np.array([None])
        return img_res

#AU_pred thread
class AU_pred(QThread):
    trigger = pyqtSignal(list,list)
    def  __init__ (self,image):
        super(AU_pred ,self). __init__ ()
        self.face = image
    def run(self):
        logps, emb = Action_class._pred(self.face,Config)
        self.trigger.emit(emb.tolist(),logps.tolist())

class show(QThread):
    trigger = pyqtSignal(list,list,int)
    def  __init__ (self, frame_list ,frame_AU,log):
        super(show,self). __init__ ()
        self.frame_embed_list = frame_list
        self.frame_emb_AU = frame_AU
        self.log = log
    def pred(self):
        #Action calculation
        AU_list = self.log.tolist()[0]
        for index,i in enumerate(AU_list):
            if i >= 0.01:
                AU_list[index] = 1
            else:
                AU_list[index] = 0
        
        pred_score, self_embedding, relation_embedding = Emotion_class.validate(self.frame_embed_list) # Emotion_pred
        feature = np.concatenate((self.frame_emb_AU,relation_embedding.cpu().numpy()), axis = 1)
        results = SVM_model.predict(feature) # Lie_pred
        return AU_list, pred_score, results
    def run(self):
        logps,  pred_score, results  = self.pred()
        self.trigger.emit(logps,  pred_score.tolist(), results)
        

class lie_GUI(QDialog, ui.Ui_Dialog):
    def __init__(self, args):
        super(lie_GUI, self).__init__()
        print('Start deception detection')
        import qdarkstyle
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.mouth_count = 0
        self.frame_embed_list = [] # 儲存人臉
        self.frame_emb_AU = []
        self.log = []
        self.userface =[]
        self.color = (0, 255, 0)
        self.index = 0
        self.len_bbox = 1
        self.time = None 
        #Qt_design
        self.setupUi(self)
        self.Startlabel.setText('Press the button to upload a video or activate camera')
        self.Problem.setPlaceholderText('Enter the question')
        self.Record.setPlaceholderText('Enter the description')
        #hidden button
        self.Reset.setVisible(False)
        self.Finish.setVisible(False)
        self.truth_lie.setVisible(False)
        self.prob_label.setVisible(False)
        self.Start.setVisible(False)
        self.RecordStop.setVisible(False)
        self.filename.setVisible(False)
        self.videoprogress.setVisible(False)
        self.User0.setVisible(False)
        self.User1.setVisible(False)
        self.User2.setVisible(False)
        self.Record_area.setVisible(False)
        self.Problem.setVisible(False)
        self.Record.setVisible(False)
        # self.Export.setVisible(False)
        self.camera_start.setVisible(False)
        self.Clear.setVisible(False)
        self.camera_finish.setVisible(False)
        #set style
        self.videoprogress.setStyleSheet("QProgressBar::chunk ""{""background-color: white;""}") ##4183c5
        #button click
        self.loadcamera.clicked.connect(self.start_webcam)
        self.loadvideo.clicked.connect(self.get_image_file)
        self.Reset.clicked.connect(self.Reset_but)
        self.Finish.clicked.connect(self.Reset_but)
        self.camera_finish.clicked.connect(self.Reset_but)
        self.Start.clicked.connect(self.time_start)
        self.RecordStop.clicked.connect(self.record_stop)
        self.camera_start.clicked.connect(self.Enter_problem)
        self.Clear.clicked.connect(self.cleartext)
        self.User0.clicked.connect(self.User_0)
        self.User1.clicked.connect(self.User_1)
        self.User2.clicked.connect(self.User_2)
        #button icon
        self.loadvideo.setIcon(QIcon('./icon/youtube.png')) # set button icon
        self.loadvideo.setIconSize(QSize(50,50)) # set icon size
        self.loadcamera.setIcon(QIcon('./icon/camera.png')) # set button icon
        self.loadcamera.setIconSize(QSize(50,50)) # set icon size
        self.Reset.setIcon(QIcon('./icon/reset.png')) # set button icon
        self.Reset.setIconSize(QSize(60,60)) # set icon size
        self.RecordStop.setIcon(QIcon('./icon/stop.png')) # set button icon
        self.RecordStop.setIconSize(QSize(30,30)) # set icon size
        #Landmark
        # self.cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)
        # self.tddfa = TDDFA(gpu_mode='gpu', **self.cfg)
        #攝像頭
        self.cap = None
        self.countframe = 0
        #timer
        self.timer = QTimer(self, interval=0)
        self.timer.timeout.connect(self.update_frame)

    def cleartext(self):
        self.Problem.clear()
