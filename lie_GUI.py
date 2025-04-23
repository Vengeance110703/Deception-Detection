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
