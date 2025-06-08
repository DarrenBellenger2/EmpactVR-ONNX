import datetime
import time
import argparse
import librosa
#import soundfile
import mediapipe as mp
from moviepy.editor import *
from pydub import AudioSegment
import numpy as np
import subprocess
import sys
import os, glob, math, random, pickle
from pathlib import Path
import cv2
import dlib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
np.random.seed(4)

import onnx, onnxruntime

import json
from statistics import mode
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch
from torch import Tensor
from torch.nn import Linear
from torch.nn import Dropout
from torch.nn import ReLU
from torch.nn import Tanh
from torch.nn import Sigmoid
from torch.nn import Softmax
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
from torch.nn import MSELoss
from torch.nn import L1Loss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F

np.set_printoptions(threshold=sys.maxsize)

# ==================================================================================
# conda activate py312
# cd d:\Documents\PhD\PythonDevelopment\EMPACTVR ONNX November2023\combined_tests
# python FACEMESH_VIDEO_LOAD_TEST.py

# ==================================================================================


# Emotions to observe
#observed_emotions=['neutral', 'happy', 'sad', 'anger', 'fear', 'disgust', 'surprise']
#mlp_emotions=['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
observed_emotions=['neutral', 'happy', 'sad', 'anger', 'fear', 'disgust', 'surprise']
num_emotions = 7
x_full, y_full = [], []

emotion_correct = [0, 0, 0, 0, 0, 0, 0]
emotion_count = [0, 0, 0, 0, 0, 0, 0]
combined_emotion_correct = [0, 0, 0, 0, 0, 0, 0]
combined_emotion_count = [0, 0, 0, 0, 0, 0, 0]
file_count = 0
file_correct = 0                       

# ==================================================================================
# RAVDESS: 0.5    ADFES:1   CremaD:0.5    SAVEE:1
frame_cut = 0.5
# ==================================================================================



#Datasets
dataset_name=['ADFES']
#'ADFES', 'Ryerson', 'eNTERFACE05', 'RAVDESS', 'SAVEE', 'CREMA-D']

#dataset_directory=['E:\\Documents\\EmotionDatabases\\Video Libraries\\ADFES_Labelled_MP4_Short\\%s\\*.mp4']
#dataset_directory=['E:\\Documents\\EmotionDatabases\\Video Libraries\\ADFES_Labelled_MP4\\%s\\*.mp4']

#dataset_directory=['E:\\Documents\\EmotionDatabases\\Video Libraries\\RAVDESS_320_Labelled\\%s\\*.mp4']
#dataset_directory=['E:\\Documents\\EmotionDatabases\\Video Libraries\\RAVDESS_640_Labelled\\%s\\*.mp4']
#dataset_directory=['E:\\Documents\\EmotionDatabases\\Video Libraries\\RAVDESS_Labelled\\%s\\*.mp4']

#dataset_directory=['E:\\Documents\\EmotionDatabases\\Video Libraries\\SAVEE_Labelled\\%s\\*.avi']
#dataset_directory=['E:\\Documents\\EmotionDatabases\\Video Libraries\\CREMA-D_Labelled\\%s\\*Hi*']

#'E:\\Documents\\EmotionDatabases\\Video Libraries\\ADFES_Labelled\\%s\\*'
#'E:\\Documents\\EmotionDatabases\\Video Libraries\\Ryerson\\Ryerson_EN_Labelled\\%s\\*',
#'E:\\Documents\\EmotionDatabases\\Video Libraries\\eNTERFACE05\\eNTERFACE05_Labelled\\%s\\*',
#'E:\\Documents\\EmotionDatabases\\Video Libraries\\RAVDESS_Labelled\\%s\\*',
#'E:\\Documents\\EmotionDatabases\\Video Libraries\\SAVEE_Labelled\\%s\\*',
#'E:\\Documents\\EmotionDatabases\\Video Libraries\\CREMA-D_Labelled\\%s\\*']

#dataset_name=['ADFES(V)']
#dataset_directory=['c:\\Users\\bellengerd\\Documents\\EmotionDatabases\\Video Libraries\\ADFES_Labelled\\%s\\*']


#dataset_directory=['E:\\Documents\\EmotionDatabases\\Enlarged Video Libraries\\RAVDESS_320_Labelled\\%s\\*.mp4']

dataset_directory=['E:\\Documents\\EmotionDatabases\\Enlarged Video Libraries\\RAVDESS_640_Labelled\\%s\\*.mp4']
#dataset_directory=['E:\\Documents\\EmotionDatabases\\Video Libraries\\RAVDESS_Labelled_HIGH\\%s\\*.mp4']
#dataset_directory=['E:\\Documents\\EmotionDatabases\\Video Libraries\\RAVDESS_Labelled_VERYHIGH\\%s\\*.mp4']


# ==================================================================================
# Unity small libraries
#dataset_directory=['E:\\Unity\\EMPACTVR_Demo2\\Assets\\StreamingAssets\\OpenCVForUnity\\emotion_videos\\ADFES_Labelled_MP4\\%s\\*.mp4']

#dataset_directory=['E:\\Unity\\EMPACTVR_Demo2\\Assets\\StreamingAssets\\OpenCVForUnity\\emotion_videos\\RAVDESS_640_Labelled\\%s\\*.mp4']

# ==================================================================================

rows, cols = (6, 7)
dataset_facial_accuracies = [[0]*cols]*rows
dataset_facial_correct = [[0]*cols]*rows
dataset_facial_count = [[0]*cols]*rows

dataset_audio_accuracies = [[0]*cols]*rows
dataset_audio_correct = [[0]*cols]*rows
dataset_audio_count = [[0]*cols]*rows

dataset_combined_accuracies = [[0]*cols]*rows
dataset_combined_correct = [[0]*cols]*rows
dataset_combined_count = [[0]*cols]*rows

# ================================================================================================
# Jawline 0-15
# Right-eyebrow 16-21
# Left-eyebrow 22-26
# Nose 27-35
# Eyes 36-47
# Mouth 48-68
landmark_points_68 = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,
                        46,53,52,65,55,
                        285,295,282,283,276,
                        168,197, 5, 4, 44, 167, 2, 393, 274,
                        33,159,159,133,145,145,
                        362,386,386,263,374,374,
                        61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87]                   
# ================================================================================================ 


#Set up some required objects
#video_capture = cv2.VideoCapture(0) #Webcam object
#detector = dlib.get_frontal_face_detector() #Face detector
#predictor = dlib.shape_predictor("..\\facial_model_training\shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever 

print("Load model")
#facial_model = pickle.load(open("..\models\Facial_1.mlp", 'rb'))
#audio_model = pickle.load(open("..\models\Audio_1.mlp", 'rb'))
#print("Audio Classes:",audio_model.classes_)
#print("Facial Classes:",facial_model.classes_)

FACIAL_MODEL_PATH = 'MediaPipe_ImageClassifier_1.onnx'
facial_path = FACIAL_MODEL_PATH
facial_model = onnx.load(facial_path)
onnx.checker.check_model(facial_model)
print('ONNX facial model loaded')

#AUDIO_MODEL_PATH = '..\models\AudioClassifier_V2.onnx'
#audio_path = AUDIO_MODEL_PATH
#audio_model = onnx.load(audio_path)
#onnx.checker.check_model(audio_model)
#print('ONNX audio model loaded')

def get_args():
    parser = argparse.ArgumentParser()

    #parser.add_argument("--device", type=int, default=0)
    #parser.add_argument("--width", help='cap width', type=int, default=960)
    #parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument("--max_num_faces", type=int, default=1)
    parser.add_argument('--refine_landmarks', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    parser.add_argument('--use_brect', action='store_true')

    args = parser.parse_args()

    return args


def distance(From_X, From_Y, To_X, To_Y): 
    From_X = int(From_X)
    From_Y = int(From_Y)
    To_X = int(To_X)
    To_Y = int(To_Y)
    dist = math.sqrt( math.pow(To_X - From_X,2) + math.pow(To_Y - From_Y,2) )
    return dist

def lerp(x, a, b): 
    ret = (x - a) / (b - a)
    if ret < 0:
        ret = 0
    return round(ret,3)
    
def pos_atan2(y, x): 
    ret = math.atan2(y, x)  
    if ret < 0:
        ret = 0
    return ret      
    
def extract_facial_feature(landmarks, image, w, h):
    AU1 = 0
    AU2 = 0
    AU4 = 0
    AU6 = 0
    AU7 = 0
    AU9 = 0
    AU15 = 0
    AU20 = 0
    AU23 = 0
    AU25 = 0
    AU26 = 0
    
    x = []
    y = []

    for i in range(0,468): #Store X and Y coordinates in two lists
        x.append(float(landmarks.landmark[i].x * w))
        y.append(float(landmarks.landmark[i].y * h))
        

        
    # ===============================================================================
    # Evaluate Action Units

    # ===============================================================================
    # AU1: Max = 1.2 & Min = 0.1
    #
    #AU1 = (math.atan2(y[17] - y[20], x[20] - x[17]) + math.atan2(y[26] - y[23],x[26] - x[23]))
    
    AU1 = (math.atan2(y[70] - y[66], x[66] - x[70]) + math.atan2(y[300] - y[296],x[300] - x[296]))
    AU1 += (math.atan2(y[46] - y[65], x[65] - x[46]) + math.atan2(y[276] - y[295],x[276] - x[295]))
    AU1 += (math.atan2(y[225] - y[222], x[222] - x[225]) + math.atan2(y[445] - y[442],x[445] - x[442]))
    
    #AU1 = (math.atan2(y[71] - y[66], x[66] - x[71]) + math.atan2(y[301] - y[296],x[301] - x[296]))
    #AU1 += (math.atan2(y[70] - y[65], x[65] - x[70]) + math.atan2(y[300] - y[295],x[300] - x[295])) 
    
    #AU1 = round(AU1, 4)
    AU1 = lerp(AU1,0.1,2.8)
    
    # ===============================================================================
    # AU2: Max = 2.3 & Min = 0.9
    #
    #AU2 = (math.atan2(y[17] - y[18], x[18] - x[17]) + math.atan2(y[26] - y[25],x[26] - x[25]))
    
    AU2 = (math.atan2(y[70] - y[63], x[63] - x[70]) + math.atan2(y[300] - y[293],x[300] - x[293]))
    AU2 += (math.atan2(y[46] - y[53], x[53] - x[46]) + math.atan2(y[276] - y[283],x[276] - x[283]))
    AU2 += (math.atan2(y[225] - y[224], x[224] - x[225]) + math.atan2(y[445] - y[444],x[445] - x[444]))
    #AU2 = round(AU2, 4) 
    AU2 = lerp(AU2,1.9,5.5)
    
    # ===============================================================================
    # AU4: Max = 3 & Min = 1.2
    #
    #AU4 = (math.atan2(y[39] - y[21], x[21] - x[39]) + math.atan2(y[42] - y[22],x[42] - x[22]))
    
    AU4 = (math.atan2(y[133] - y[55], x[55] - x[133]) + math.atan2(y[362] - y[285],x[362] - x[285]))
    AU4 += (math.atan2(y[153] - y[55], x[55] - x[153]) + math.atan2(y[380] - y[285],x[380] - x[285]))
    AU4 += (math.atan2(y[158] - y[55], x[55] - x[158]) + math.atan2(y[385] - y[285],x[385] - x[285]))
    #AU4 = round(AU4, 4)
    AU4 = lerp(AU4,2,7)
    
    # ===============================================================================
    # AU6: Max = 1.16 & Min = 0.7
    #
    #AU6 = (distance(x[48],y[48],x[66],y[66]) + distance(x[66],y[66],x[54],y[54])) / (distance(x[48],y[48],x[51],y[51]) + distance(x[51],y[51],x[54],y[54]))
    #AU6 = (distance(x[76],y[76],x[14],y[14]) + distance(x[14],y[14],x[306],y[306])) / (distance(x[76],y[76],x[0],y[0]) + distance(x[0],y[0],x[306],y[306]))
    AU6 = (distance(x[216],y[216],x[12],y[12]) + distance(x[12],y[12],x[436],y[436])) / (distance(x[74],y[74],x[13],y[13]) + distance(x[13],y[13],x[304],y[304]))
    #AU6 = round(AU6, 4)
    AU6 = lerp(AU6,1.5,2.8)
    
    # ===============================================================================
    # AU7: Max = 0.5 & Min = 0.05
    #       

    # --------------------
    # RAVDESS 320: 36.3 
    #AU7_right = ((distance(x[159],y[159],x[145],y[145]) + distance(x[159],y[159],x[145],y[145])) / (2 * distance(x[130],y[130],x[243],y[243])))
    #AU7_left = ((distance(x[386],y[386],x[374],y[374]) + distance(x[386],y[386],x[374],y[374])) / (2 * distance(x[463],y[463],x[359],y[359])))

    
    # --------------------
    # RAVDESS 320: 36.3 
    AU7_right = ((distance(x[159],y[159],x[145],y[145]) + distance(x[159],y[159],x[145],y[145])) / (2 * distance(x[130],y[130],x[243],y[243])))
    AU7_left = ((distance(x[386],y[386],x[374],y[374]) + distance(x[386],y[386],x[374],y[374])) / (2 * distance(x[463],y[463],x[359],y[359])))

    
    AU7 = (AU7_right + AU7_left / 2)   
    
    #AU7 = round(AU7, 4)
    AU7 = lerp(AU7,0.02,0.7)

    # ===============================================================================
    # AU9: Max = 2.5 & Min = 1
    #       
    #AU9 = (math.atan2(y[32] - y[33], x[33] - x[32]) + math.atan2(y[34] - y[33],x[34] - x[33]))
    #AU9 = (math.atan2(y[167] - y[2], x[2] - x[167]) + math.atan2(y[393] - y[2],x[393] - x[2]))
    AU9 = (math.atan2(y[37] - y[2], x[2] - x[37]) + math.atan2(y[267] - y[2],x[267] - x[2]))
    
    #AU9 = round(AU9, 4)
    AU9 = lerp(AU9,1.1,2.7)
    
    # ===============================================================================
    # AU15: Max = 1 & Min = -0.5 (-0.8)
    #          
    #AU15 = (math.atan2(y[48] - y[60], x[60] - x[48]) + math.atan2(y[54] - y[64],x[54] - x[64]))
    AU15 = (math.atan2(y[43] - y[76], x[76] - x[43]) + math.atan2(y[273] - y[306],x[273] - x[306]))
    
    #AU15 = round(AU15, 4)
    AU15 = lerp(AU15,-2,4.5)

    # ===============================================================================
    # AU20: Max = 3 & Min = 0
    #         
    #AU20 = pos_atan2(y[59] - y[65], x[65] - x[59]) + pos_atan2(y[55] - y[67],x[55] - x[67])
    #AU20 += pos_atan2(y[59] - y[67],x[67] - x[59]) + pos_atan2(y[55] - y[65],x[55] - x[65]) 
    
    #AU20 += pos_atan2(y[49] - y[52], x[52] - x[49]) + pos_atan2(y[53] - y[50],x[53] - x[50])
    #AU20 += pos_atan2(y[49] - y[50],x[50] - x[49]) + pos_atan2(y[53] - y[52],x[53] - x[52])
    
    AU20 = pos_atan2(y[180] - y[317], x[317] - x[180]) + pos_atan2(y[404] - y[87],x[404] - x[87])
    AU20 += pos_atan2(y[180] - y[87],x[87] - x[180]) + pos_atan2(y[404] - y[317],x[404] - x[317]) 
    
    AU20 += pos_atan2(y[73] - y[267], x[267] - x[73]) + pos_atan2(y[303] - y[37],x[303] - x[37])
    AU20 += pos_atan2(y[73] - y[37],x[37] - x[73]) + pos_atan2(y[303] - y[267],x[303] - x[267])
    
    #AU20 = round(AU20, 4)
    AU20 = lerp(AU20,0.4,6)

    # ===============================================================================
    # AU23: Max = 9 & Min = 2
    #          
    #AU23 = (pos_atan2(y[49] - y[50], x[50] - x[49]) + pos_atan2(y[53] - y[52],x[53] - x[52]) + pos_atan2(y[61] - y[49],x[61] - x[49]) + pos_atan2(y[63] - y[53],x[53] - x[63]) + pos_atan2(y[58] - y[59],x[58] - x[59]) + pos_atan2(y[56] - y[55],x[55] - x[56]) + pos_atan2(y[60] - y[51],x[51] - x[60]) + pos_atan2(y[64] - y[51],x[64] - x[51]) + pos_atan2(y[57] - y[60],x[57] - x[60]) + pos_atan2(y[57] - y[64],x[64] - x[57]) + pos_atan2(y[62] - y[49],x[62] - x[49]) + pos_atan2(y[62] - y[53],x[53] - x[62]) + pos_atan2(y[57] - y[60],x[57] - x[60]) + pos_atan2(y[57] - y[64],x[64] - x[57]))
    AU23 = (pos_atan2(y[73] - y[37], x[37] - x[73]) + pos_atan2(y[303] - y[267],x[303] - x[267]) + pos_atan2(y[38] - y[73],x[38] - x[73]) + pos_atan2(y[268] - y[303],x[303] - x[268]) + pos_atan2(y[84] - y[180],x[84] - x[180]) + pos_atan2(y[314] - y[404],x[404] - x[314]) + pos_atan2(y[76] - y[0],x[0] - x[76]) + pos_atan2(y[306] - y[0],x[306] - x[0]) + pos_atan2(y[17] - y[76],x[17] - x[76]) + pos_atan2(y[17] - y[306],x[306] - x[17]) + pos_atan2(y[12] - y[73],x[12] - x[73]) + pos_atan2(y[12] - y[303],x[303] - x[12]) + pos_atan2(y[17] - y[76],x[17] - x[76]) + pos_atan2(y[17] - y[306],x[306] - x[17]))
    
    #AU23 = round(AU23, 4)
    AU23 = lerp(AU23,2.1,9.5)
    
    # ===============================================================================
    # AU26: Max = 1 & Min = 0
    #                 
    #AU26 = (distance(x[61],y[61],x[67],y[67]) + distance(x[62],y[62],x[66],y[66]) + distance(x[63],y[63],x[65],y[65])) / (distance(x[36],y[36],x[45],y[45]))
    #AU26 = (distance(x[38],y[38],x[87],y[87]) + distance(x[12],y[12],x[14],y[14]) + distance(x[268],y[268],x[317],y[317])) / (distance(x[33],y[33],x[263],y[263]))
    #AU26 = (distance(x[38],y[38],x[86],y[86]) + distance(x[12],y[12],x[15],y[15]) + distance(x[268],y[268],x[316],y[316])) / (distance(x[61],y[61],x[291],y[291]))
    
   
    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # Image: 86.84% RAVDESS 320 Video: 36.14% !!!!!!!!! ADFES: 59.7%
    # Image: 81.43% RAVDESS 320 Video: 33.73% !!!!!!!!! ADFES: 59.7%
    #eye_width = 0
    #lip_height = round((3 * distance(x[11],y[11],x[12],y[12])),0)
    #mouth_height = round((distance(x[82],y[82],x[87],y[87]) + distance(x[13],y[13],x[14],y[14]) + distance(x[312],y[312],x[317],y[317])),0)
    #if mouth_height > lip_height:
    #    eye_width = round((1 * distance(x[33],y[33],x[263],y[263])),0)
    #    AU26 = (mouth_height - lip_height) / eye_width
    #    AU26 = round(AU26, 2)
    #    #AU26 = lerp(AU26,0,1)        
    #    if AU26 > 0.1:
    #       AU25 = 1    
    # -----------------------------------------------------------------------------------------------------------------------------------------------

  

    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # ROUND TO 2 Image: 88.1% RAVDESS 320 Video: 34.7% ADFES: 60.39%
    #eye_width = distance(x[33],y[33],x[263],y[263])
    #mouth_height = (distance(x[81],y[81],x[178],y[178]) + distance(x[13],y[13],x[14],y[14]) + distance(x[311],y[311],x[402],y[402]))
    #AU26 = round((mouth_height / eye_width), 2)
    ##AU26 = lerp(AU26,0,1)
    #if AU26 > 0.1:
    #   AU25 = 1
    #
    # -----------------------------------------------------------------------------------------------------------------------------------------------
    
    eye_width = distance(x[33],y[33],x[263],y[263])
    mouth_height = (distance(x[81],y[81],x[178],y[178]) + distance(x[13],y[13],x[14],y[14]) + distance(x[311],y[311],x[402],y[402]))
    AU26 = round((mouth_height / eye_width), 2)    
    AU26 = lerp(AU26,0,1)
    if AU26 > 0.1:
       AU25 = 1 



       
    #dlib_feature = Dlib_extract_facial_feature(image) 

    # =================================     
    #AU6 = dlib_feature[3]      
    #AU7 = dlib_feature[4] # 89%
    #AU9 = dlib_feature[5]
    #AU15 = dlib_feature[6] # 85%
    #AU20 = dlib_feature[7]
    #AU23 = dlib_feature[8]  #77%
    #AU25 = dlib_feature[9]
    #AU26 = dlib_feature[10]
    # ================================= 

    #AU1 = dlib_feature[0] # 92.5% ATAN2 content     
    #AU2 = dlib_feature[1]      
    #AU4 = dlib_feature[2]   

    #AU6 = dlib_feature[3]   
    #AU7 = dlib_feature[4]       
    #AU9 = dlib_feature[5]      
    #AU15 = dlib_feature[6]
    #AU20 = dlib_feature[7]
    #AU23 = dlib_feature[8]
    #AU25 = dlib_feature[9]
    #AU26 = dlib_feature[10] 
    # ==================================================================     
    # AU25/26 - 5K, Image: 90.26% RAVDESS 320: 34.69%
    # ==================================================================     
    
    result = np.array([AU1,AU2,AU4,AU6,AU7,AU9,AU15,AU20,AU23,AU25,AU26])   
    return result

def Dlib_extract_facial_feature(image):
    AU1 = 0
    AU2 = 0
    AU4 = 0
    AU6 = 0
    AU7 = 0
    AU9 = 0
    AU15 = 0
    AU20 = 0
    AU23 = 0
    AU25 = 0
    AU26 = 0
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)

    detections = detector(clahe_image, 1) 
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(clahe_image, d) #Draw Facial Landmarks with the predictor class
        x = []
        y = []
        for i in range(0,68): #Store X and Y coordinates in two lists
            x.append(float(shape.part(i).x))
            y.append(float(shape.part(i).y))    


            
        # ===============================================================================
        # Evaluate Action Units
        
        
        
        # ===============================================================================
        # AU1: Max = 1.2 & Min = 0.1
        #
        AU1 = (math.atan2(y[17] - y[20], x[20] - x[17]) + math.atan2(y[26] - y[23],x[26] - x[23]))
        AU1 = round(AU1, 4)
        #AU1 = lerp(AU1,0.1,1.2)
        
        # ===============================================================================
        # AU2: Max = 2.3 & Min = 0.9
        #
        AU2 = (math.atan2(y[17] - y[18], x[18] - x[17]) + math.atan2(y[26] - y[25],x[26] - x[25]))
        AU2 = round(AU2, 4)
        #AU2 = lerp(AU2,0.9,2.3)
        
        # ===============================================================================
        # AU4: Max = 3 & Min = 1.2
        #
        AU4 = (math.atan2(y[39] - y[21], x[21] - x[39]) + math.atan2(y[42] - y[22],x[42] - x[22]))
        AU4 = round(AU4, 4)
        #AU4 = lerp(AU4,1.2,3)
        
        # ===============================================================================
        # AU6: Max = 1.16 & Min = 0.7
        #
        AU6 = (distance(x[48],y[48],x[66],y[66]) + distance(x[66],y[66],x[54],y[54])) / (distance(x[48],y[48],x[51],y[51]) + distance(x[51],y[51],x[54],y[54]))
        AU6 = round(AU6, 4)
        #AU6 = lerp(AU6,0.7,1.16)
        
        # ===============================================================================
        # AU7: Max = 0.5 & Min = 0.05
        #       
        #AU7 = (distance(x[37],y[37],x[41],y[41]) + distance(x[62],y[62],x[66],y[66]) + distance(x[63],y[63],x[65],y[65])) / (2 * distance(x[36],y[36],x[45],y[45]))
        AU7_right = ((distance(x[37],y[37],x[41],y[41]) + distance(x[38],y[38],x[40],y[40])) / (2 * distance(x[36],y[36],x[39],y[39])))
        AU7_left = ((distance(x[43],y[43],x[47],y[47]) + distance(x[44],y[44],x[46],y[46])) / (2 * distance(x[42],y[45],x[45],y[45])))
        AU7 = (AU7_right + AU7_left / 2)        
        AU7 = round(AU7, 4)
        #AU7 = lerp(AU7,0.05,0.5)

        # ===============================================================================
        # AU9: Max = 2.5 & Min = 1
        #       
        AU9 = (math.atan2(y[50] - y[33], x[33] - x[50]) + math.atan2(y[52] - y[33],x[52] - x[33]))     
        AU9 = round(AU9, 4)
        #AU9 = lerp(AU9,1,2.5)
        
        # ===============================================================================
        # AU15: Max = 1 & Min = -0.5 (-0.8)
        #          
        AU15 = (math.atan2(y[48] - y[60], x[60] - x[48]) + math.atan2(y[54] - y[64],x[54] - x[64]))
        AU15 = round(AU15, 4)
        #AU15 = lerp(AU15,-0.5,1)

        # ===============================================================================
        # AU20: Max = 3 & Min = 0
        #         
        AU20 = (math.atan2(y[59] - y[65], x[65] - x[59]) + math.atan2(y[55] - y[67],x[55] - x[67]) + math.atan2(y[59] - y[66],x[66] - x[59]) + math.atan2(y[59] - y[67],x[67] - x[59]) + math.atan2(y[55] - y[65],x[55] - x[65]))
        AU20 = round(AU20, 4)
        #AU20 = lerp(AU20,0,3)

        # ===============================================================================
        # AU23: Max = 9 & Min = 2
        #          
        AU23 = (math.atan2(y[49] - y[50], x[50] - x[49]) + math.atan2(y[53] - y[52],x[53] - x[52]) + math.atan2(y[61] - y[49],x[61] - x[49]) + math.atan2(y[63] - y[53],x[53] - x[63]) + math.atan2(y[58] - y[59],x[58] - x[59]) + math.atan2(y[56] - y[55],x[55] - x[56]) + math.atan2(y[60] - y[51],x[51] - x[60]) + math.atan2(y[64] - y[51],x[64] - x[51]) + math.atan2(y[57] - y[60],x[57] - x[60]) + math.atan2(y[57] - y[64],x[64] - x[57]) + math.atan2(y[62] - y[49],x[62] - x[49]) + math.atan2(y[62] - y[53],x[53] - x[62]) + math.atan2(y[57] - y[60],x[57] - x[60]) + math.atan2(y[57] - y[64],x[64] - x[57]))
        AU23 = round(AU23, 4)
        #AU23 = lerp(AU23,2,9)
        
        # ===============================================================================
        # AU26: Max = 1 & Min = 0
        #                 
        AU26 = (distance(x[61],y[61],x[67],y[67]) + distance(x[62],y[62],x[66],y[66]) + distance(x[63],y[63],x[65],y[65])) / (distance(x[36],y[36],x[45],y[45]))
        AU26 = round(AU26, 4)
        #AU26 = lerp(AU26,0,1)        
        if AU26 > 0.1:
            AU25 = 1
        
        break

    result = np.array([AU1,AU2,AU4,AU6,AU7,AU9,AU15,AU20,AU23,AU25,AU26])   
    return result

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h] 
    
def extract_feature(file_name, mfcc, chroma, mel):

    #===================================================
    #audio_clip = AudioSegment.from_file(file_name)
    #b = audio_clip.split_to_mono()
    #b[0].export(out_f="temp.wav",format="wav")
    #wav_file = "temp.wav"
    #===================================================

    #===================================================
    clip = AudioFileClip(file_name)
    #clip.write_audiofile("temp.wav", fps=44100, nbytes=4, buffersize=50000, codec='pcm_f32le', ffmpeg_params=["-ac", "1"], verbose=False, logger=None)
    clip.write_audiofile("temp.wav", fps=44100, nbytes=4, buffersize=50000, codec='pcm_f32le', ffmpeg_params=["-ac", "1"], verbose=False, logger=None)
    wav_file = "temp.wav"
    #===================================================
    
    #===================================================
    #filename = Path(file_name)
    #filename_wo_ext = filename.with_suffix('')
    #wav_file = filename.with_suffix('.wav') 
    #===================================================
    
    
    with soundfile.SoundFile(wav_file) as mySoundFile:
        X = mySoundFile.read(dtype="float32")
        sample_rate = mySoundFile.samplerate

        if chroma:
            stft = np.abs(librosa.stft(X))
            result = np.array([])
        if mfcc:
            #my_mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            #result = np.hstack((result, my_mfccs))

            full_mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=34)
            subset_mfccs = full_mfccs[1:]       
            norm_mfccs = np.subtract(subset_mfccs,np.mean(subset_mfccs))
            my_mfccs = np.mean(norm_mfccs.T, axis=0)            
            result = np.hstack((result, my_mfccs))             
            
            
        #if chroma:
        #    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis = 0)
        #    result = np.hstack((result, chroma))
        #if mel:
        #    mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
        #    result = np.hstack((result, mel))
    return result
    
def process_image(file, emotion, face_mesh):

    face_data = []
    face_label = []
    probability_rtn = [[0, 0, 0, 0, 0, 0, 0]]

    clip = VideoFileClip(file) 
    subclip = clip.subclip(frame_cut,(frame_cut * -1))
    subclip.write_videofile("temp.mp4") 
    
    cap = cv2.VideoCapture("temp.mp4")                
    while(cap.isOpened()):
        ret, image = cap.read()
        if ret == False:
            break                
    
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        #clahe_image = clahe.apply(gray)

        #detections = detector(clahe_image, 1)   
        #if len(detections) > 0:
        #    feature = extract_facial_feature(clahe_image)
        #    face_data.append(feature)
        #    face_label.append(emotion)
    
        h, w, c = image.shape                   
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)                  

        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                brect = calc_bounding_rect(image, face_landmarks)
                left_eye, right_eye = None, None

                #landmarks_extracted = []
                #for index in landmark_points_68:
                #    x_val = int(face_landmarks.landmark[index].x * w)
                #    y_val = int(face_landmarks.landmark[index].y * h)
                #    landmarks_extracted.append((x_val, y_val))
        
                #feature = extract_facial_feature(landmarks_extracted, image)
                feature = extract_facial_feature(face_landmarks, image, w, h)                           
                face_data.append(feature)
                face_label.append(emotion)
    
    
    face_data = [arr.tolist() for arr in face_data]
    batch_pred, proba = predict_batch(face_data, ort_facial_session)
    
    mean_array = np.mean(face_data, axis=0)
    mean_array = mean_array.reshape(1, -1) 
    temp = mean_array[0].tolist()  
    #print("MEAN:",temp)
    mean_pred,probability_rtn = predict(temp, ort_facial_session)
    
    print(batch_pred,"-", mean_pred, " : ", proba, "   proba=", probability_rtn)  

    result=0
    #if observed_emotions[batch_pred] == emotion:
    #if observed_emotions[batch_pred] == emotion or observed_emotions[mean_pred] == emotion:
    #if observed_emotions[batch_pred] == emotion:
    if observed_emotions[mean_pred] == emotion:
        result = 1
        print("match")
    
    #=========================================================================
    
    #probability_rtn = 0
    #proba = facial_model.predict_proba(face_data)
    #for i in range(len(proba)):
    #    for j in range(len(proba[i])): 
    #        probability_rtn[0][j] += round(proba[i][j] * 100)       
    #        #proba[i][j] = round(proba[i][j] * 100)          
    #for j in range(len(probability_rtn[0])): 
    #        probability_rtn[0][j] = round(probability_rtn[0][j] / len(proba),2)
    
    
    cap.release()        

    return result,probability_rtn

def process_audio(file, emotion):

    result = 0
    audio_data = []
    audio_label = []
    feature = extract_feature(file, mfcc=True, chroma=True, mel=True)           
    audio_data.append(feature)  
    audio_label.append(emotion) 
    if len(audio_data) > 0:
        
        #y_pred= audio_model.predict(audio_data)                     
        #result = audio_model.score(audio_data,audio_label)
        
        #proba = audio_model.predict_proba(audio_data)
        #for i in range(len(proba)):
        #    for j in range(len(proba[i])):      
        #        proba[i][j] = round(proba[i][j] * 100)       
        #print("Label:",audio_label,"AUDIO: ", proba)     

        file_pred = np.array(audio_data)
        file_pred = file_pred.reshape(1, -1)

        #=========================================================================
        # Batch Prediction
        audio_data = [arr.tolist() for arr in audio_data]
        #batch_pred, proba = predict_batch(face_data, ort_session)
        batch_pred, proba = predict_batch_alt(audio_data, ort_audio_session)
        
        #batch_pred,proba = predict(audio_data, ort_audio_session)
        
        #=========================================================================
        print(batch_pred, " - ", proba)
        result=0
        if observed_emotions[batch_pred] == emotion:
            result = 1
        
        
    return result,proba
    
def load_data(face_mesh):
    global x_full, y_full
    #x, y = [], []
    global file_count, file_correct  
    sub_count = 0
    sub_correct = 0
    combined_sub_correct = 0
    index = 0
    

    # Changed order from dataset>>emotion TO emotion>>dataset
    for emotion in observed_emotions:   

        print("Emotion=",emotion)     
        index = observed_emotions.index(emotion)                    
        
        for d_index,d_item in enumerate(dataset_directory):
            print("Dataset=",dataset_name[d_index])     
            sub_count = 0
            #sub_correct = 0         
            for file in glob.glob(d_item %emotion): 

                #now = datetime.datetime.now()
                #print ("START: " + now.strftime("%Y-%m-%d %H:%M:%S"))

                print("File=",file)
                file_count +=1
                sub_count += 1 

                # ==========================================================================
                # Perform facial prediction
                facial_accuracy = 0
                facial_accuracy,facial_probability = process_image(file, emotion, face_mesh)               
                if facial_accuracy > 0:
                    dataset_facial_correct[d_index][index] += 1;   
                else:
                    highest_proba = np.amax(facial_probability)  
                    if (facial_probability[index] >= highest_proba):
                        dataset_facial_correct[d_index][index] += 1;   

                
                #print("facial accuracy=",facial_accuracy)
                print("facial probability=",facial_probability)
                
                
                        
                
                # ==========================================================================
                # Perform audio prediction

                audio_accuracy = 0
                #audio_accuracy,audio_probability = process_audio(file, emotion)               
                #if audio_accuracy >= 0.5:
                #    dataset_audio_correct[d_index][index] += 1; 
                
                #print("audio probability=",audio_probability)
                    

               
                # ==========================================================================
                # Perform combined prediction

                #combined_proba = [0, 0, 0, 0, 0, 0, 0]              
                #for i in range(len(combined_proba)):
                #    combined_proba[i] = facial_probability[i] + audio_probability[i]
                #highest_proba = np.amax(combined_proba)  
                #print("(",highest_proba,") combined probability=",combined_proba)
                #if (combined_proba[index] >= highest_proba):
                #    dataset_combined_correct[d_index][index] += 1;   
                
                
                    
            # Calculate Percentage correct for dataset emotion
            dataset_facial_count[d_index][index] = sub_count; 
            #dataset_audio_count[d_index][index] = sub_count; 
            #dataset_combined_count[d_index][index] = sub_count;   
            #dataset_facial_accuracies[d_index][index] = 0
            #if sub_count > 0:
            #    dataset_facial_accuracies[d_index][index] = ((sub_correct / sub_count) * 100)
            #print("Correct=",dataset_facial_accuracies[d_index][index],"  (",file_count,")")          


    #cap.release()        
       
            
    #x_full = np.array(x)
    #y_full = y
    return

# make a class prediction for rows of data
def predict_batch(rows, ort_session):

    prediction_count = [0, 0, 0, 0, 0, 0, 0]
    values = []
    
    #print("Length=",len(rows))
    data = json.dumps({'data': rows})
    data = np.array(json.loads(data)['data']).astype('float32')
    temp = data #numpy.expand_dims(data, axis=0)
    print(temp.shape)

    ort_inputs = {input_name: temp}
    ort_outs = ort_session.run([output_name], ort_inputs)[0]
    
    for i, value in enumerate(ort_outs):
        int_value = np.argmax(value)  
        values.append(int_value) 
        prediction_count[int_value] += 1

    highest_index = 0
    for i in range(num_emotions):
        if prediction_count[i] > prediction_count[highest_index]:
            highest_index = i
    prediction = highest_index
    
    #print(prediction_count)
    #print("Prediction=",prediction)
    
    return prediction, prediction_count 

def predict_batch_alt(rows, ort_session):

    prediction_count = [0, 0, 0, 0, 0, 0, 0]
    values = []
    
    for i in range(len(rows)):    
        pred, proba = predict(rows[i], ort_session)
        print("predict_batch_alt: ", pred, " - ", proba)
        values.append(pred) 
        prediction_count[pred] += 1

    highest_index = 0
    for i in range(num_emotions):
        if prediction_count[i] > prediction_count[highest_index]:
            highest_index = i
    prediction = highest_index
    
    #print(prediction_count)
    #print("Prediction=",prediction)
    
    return prediction, proba 
    
# make a class prediction for one row of data
def predict(row, ort_session):

    data = json.dumps({'data': row})
    data = np.array(json.loads(data)['data']).astype('float32')
    temp = np.expand_dims(data, axis=0)
    #print(temp.shape)

    ort_inputs = {input_name: temp}
    ort_outs = ort_session.run([output_name], ort_inputs)

    prediction = np.argmax(ort_outs[0][0])  
    proba = ort_outs[0][0]
    for i in range(len(proba)):
        proba[i] = round(proba[i] * 100)       
    
    return prediction, proba

    
def main(): 

    args = get_args()
    max_num_faces = args.max_num_faces
    refine_landmarks = args.refine_landmarks
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    use_brect = args.use_brect

    print("=============================================================================================")
    print("Setup Face Mesh")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=max_num_faces,
        refine_landmarks=True,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )   

    print("=============================================================================================")
    print("Load data")
    load_data(face_mesh)

    print("=============================================================================================")
    print("= Facial")
    overall_percentage = 0
    overall_correct=0
    overall_count=0
    for emotion in observed_emotions: 
        index = observed_emotions.index(emotion) 
        cycle_percentage = 0   
        cycle_correct=0
        cycle_count=0
        for d_index,d_item in enumerate(dataset_directory):
            dataset_index = index
            cycle_correct += dataset_facial_correct[d_index][index]
            cycle_count += dataset_facial_count[d_index][index]
        if cycle_correct != 0 and cycle_count !=0:
            cycle_percentage = ((cycle_correct / cycle_count) * 100)
        overall_correct+=cycle_correct
        overall_count+=cycle_count
        print("Emotion=",emotion,"  (",cycle_percentage,"%) (files=",cycle_count,")")  
    if overall_correct != 0 and overall_count !=0:
        overall_percentage = ((overall_correct / overall_count) * 100)    
    print("Overall: (",overall_percentage,"%) (files=",overall_count,")")      

    #print("=============================================================================================")
    #print("= Audio")
    #overall_percentage = 0
    #overall_correct=0
    #overall_count=0
    #for emotion in observed_emotions: 
    #    index = observed_emotions.index(emotion)       
    #    cycle_percentage = 0   
    #    cycle_correct=0
    #    cycle_count=0
    #    for d_index,d_item in enumerate(dataset_directory):
    #        dataset_index = index
    #        cycle_correct += dataset_audio_correct[d_index][index]
    #        cycle_count += dataset_audio_count[d_index][index]
    #    if cycle_correct != 0 and cycle_count !=0:
    #        cycle_percentage = ((cycle_correct / cycle_count) * 100)
    #    overall_correct+=cycle_correct
    #    overall_count+=cycle_count
    #    print("Emotion=",emotion,"  (",cycle_percentage,"%) (files=",cycle_count,")")  
    #if overall_correct != 0 and overall_count !=0:
    #    overall_percentage = ((overall_correct / overall_count) * 100)    
    #print("Overall: (",overall_percentage,"%) (files=",overall_count,")")  

    #print("=============================================================================================")
    #print("= Combined")
    #overall_percentage = 0
    #overall_correct=0
    #overall_count=0
    #for emotion in observed_emotions: 
    #    index = observed_emotions.index(emotion)       
    #    cycle_percentage = 0   
    #    cycle_correct=0
    #    cycle_count=0
    #    for d_index,d_item in enumerate(dataset_directory):
    #        dataset_index = index
    #        cycle_correct += dataset_combined_correct[d_index][index]
    #        cycle_count += dataset_combined_count[d_index][index]
    #    if cycle_correct != 0 and cycle_count !=0:
    #        cycle_percentage = ((cycle_correct / cycle_count) * 100)
    #    overall_correct+=cycle_correct
    #    overall_count+=cycle_count
    #    print("Emotion=",emotion,"  (",cycle_percentage,"%) (files=",cycle_count,")")  
    #if overall_correct != 0 and overall_count !=0:
    #    overall_percentage = ((overall_correct / overall_count) * 100)    
    #print("Overall: (",overall_percentage,"%) (files=",overall_count,")")  


    for d_index,d_item in enumerate(dataset_directory):
        for emotion in observed_emotions: 
            index = observed_emotions.index(emotion)       
            cycle_correct=0
            cycle_count=0
        cycle_percentage = ((cycle_correct / cycle_count) * 100)
        print("Dataset=",dataset_name[d_index],"  (",cycle_percentage,"%) (files=",cycle_count,")")     


print("=============================================================================================")
# Setup runtime
EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
ort_facial_session = onnxruntime.InferenceSession(FACIAL_MODEL_PATH, providers=EP_list)
input_name = ort_facial_session.get_inputs()[0].name
output_name = ort_facial_session.get_outputs()[0].name
print("FACIAL:", input_name)
print("FACIAL:", output_name)  

#ort_audio_session = onnxruntime.InferenceSession(AUDIO_MODEL_PATH, providers=EP_list)
#input_name = ort_audio_session.get_inputs()[0].name
#output_name = ort_audio_session.get_outputs()[0].name
#print("AUDIO:", input_name)
#print("AUDIO:", output_name)

if __name__ == '__main__':
    main()

