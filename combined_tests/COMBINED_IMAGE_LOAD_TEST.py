import numpy as np
import os, glob, math, random, pickle
import cv2
import argparse
import mediapipe as mp
import dlib
#from sklearn.model_selection import train_test_split
#from sklearn.neural_network import MLPClassifier
#from sklearn.metrics import accuracy_score
#np.random.seed(4)

import onnx, onnxruntime

#import numpy 
import json
#from numpy import vstack

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

MODEL_PATH = 'MediaPipe_ImageClassifier_1.onnx'

# ==================================================================================
# conda activate py312
# cd d:\Documents\PhD\PythonDevelopment\EMPACTVR ONNX November2023\combined_tests
# python COMBINED_IMAGE_LOAD_TEST.py
# ==================================================================================

# Emotions to observe
observed_emotions=['neutral', 'happy', 'sad', 'anger', 'fear', 'disgust', 'surprise']
x_full, y_full = [], []

emotion_correct = [0, 0, 0, 0, 0, 0, 0]
emotion_count = [0, 0, 0, 0, 0, 0, 0]
file_count = 0
file_correct = 0

# ================================================================================================
# Jawline 0-15
# Right-eyebrow 16-21
# Left-eyebrow 22-26
# Nose 27-35
# Eyes 36-47
# Mouth 48-67
landmark_points_68 = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,
                        46,53,52,65,55,
                        285,295,282,283,276,
                        168,197, 5, 4, 44, 167, 2, 393, 274,
                        33,159,159,133,145,145,
                        362,386,386,263,374,374,
                        43,73,37, 0,267,303,273,404,314,17,84,180,76,38,12,268,306,317,14,87]
                                             
# ================================================================================================ 

#Datasets
dataset_name=['Yale','CK', 'ADFES', 'FEI', 'JAFFE', 'OulaCasia', 'VisGraf']
#dataset_name=['CK Train']
#dataset_directory=['D:\\Documents\\EmotionDatabases\\Image Libraries\\CK_Train\\%s\\*'] 

dataset_directory=[
'D:\\Documents\\EmotionDatabases\\Image Libraries\\Yale_Labelled\\%s\\*',
'D:\\Documents\\EmotionDatabases\\Image Libraries\\CK_Labelled\\%s\\*',
'D:\\Documents\\EmotionDatabases\\Image Libraries\\ADFES_Labelled\\%s\\*',
'D:\\Documents\\EmotionDatabases\\Image Libraries\\FEI_Labelled\\%s\\*',
'D:\\Documents\\EmotionDatabases\\Image Libraries\\JAFFE_Labelled\\%s\\*',
'D:\\Documents\\EmotionDatabases\\Image Libraries\\OulaCasia_Labelled\\%s\\*',
'D:\\Documents\\EmotionDatabases\\Image Libraries\\VisGraf_Labelled\\%s\\*']

#dataset_directory=['D:\\Documents\\EmotionDatabases\\Image Libraries\\CK_Labelled\\%s\\*']

dataset_accuracies = [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]
dataset_correct = [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]
dataset_count = [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]

#Set up some required objects
video_capture = cv2.VideoCapture(0) #Webcam object
detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever 
 
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
    #if ret < 0:
    #    ret = 0
    return ret      
    
def extract_feature(landmarks, image, w, h):
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
    #AU7_right = ((distance(x[37],y[37],x[41],y[41]) + distance(x[38],y[38],x[40],y[40])) / (2 * distance(x[36],y[36],x[39],y[39])))
    #AU7_left = ((distance(x[43],y[43],x[47],y[47]) + distance(x[44],y[44],x[46],y[46])) / (2 * distance(x[42],y[45],x[45],y[45])))
    
    AU7_right = ((distance(x[159],y[159],x[145],y[145]) + distance(x[159],y[159],x[145],y[145])) / (2 * distance(x[130],y[130],x[243],y[243])))
    AU7_left = ((distance(x[386],y[386],x[374],y[374]) + distance(x[386],y[386],x[374],y[374])) / (2 * distance(x[463],y[463],x[359],y[359])))
    
    AU7 = (AU7_right + AU7_left / 2)    
    
    #AU7 = round(AU7, 4)
    AU7 = lerp(AU7,0.02,0.7)

    # ===============================================================================
    # AU9: Max = 2.5 & Min = 1
    #       
    #AU9 = (math.atan2(y[167] - y[2], x[2] - x[167]) + math.atan2(y[393] - y[2],x[393] - x[2]))
    AU9 = (math.atan2(y[37] - y[2], x[2] - x[37]) + math.atan2(y[267] - y[2],x[267] - x[2]))
    
    #AU9 = round(AU9, 4)
    AU9 = lerp(AU9,1.1,2.7)
    
    # ===============================================================================
    # AU15: Max = 1 & Min = -0.5 (-0.8)
    #          
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
    
    #eye_width = distance(x[33],y[33],x[263],y[263])
    #mouth_height = (distance(x[81],y[81],x[178],y[178]) + distance(x[13],y[13],x[14],y[14]) + distance(x[311],y[311],x[402],y[402]))
    #AU26 = round((mouth_height / eye_width), 2)
	
    eye_width = distance(x[33],y[33],x[263],y[263])
    mouth_height = (distance(x[81],y[81],x[178],y[178]) + distance(x[13],y[13],x[14],y[14]) + distance(x[311],y[311],x[402],y[402]))
    AU26 = round((mouth_height / eye_width), 2)    
    AU26 = lerp(AU26,0,1)
    if AU26 > 0.1:
       AU25 = 1 

	
    #dlib_feature = Dlib_extract_facial_feature(image)

    #AU1 = dlib_feature[0] 
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

        #print("DL Co-ords:" + " (0)=" + str(x[0]) + "/" + str(y[0]) + " (16)=" + str(x[16]) + "/" + str(y[16]) + " (8)=" + str(x[8]) + "/" + str(y[8])) 
        #print("DL Co-ords:" + " (0)=" + str(x[0]) + "/" + str(y[0]) + " (16)=" + str(x[16]) + "/" + str(y[16]) + " (8)=" + str(x[8]) + "/" + str(y[8])) 

            
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

    result = np.array([AU1,AU2,AU4,AU6,AU7,AU9,AU15,AU20,AU23,AU25, AU26])   
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
    
def load_data():
    global x_full, y_full
    x, y = [], []
    global file_count, file_correct  
    sub_count = 0
    sub_correct = 0
    emotion_index = 0    
    
    # Changed order from dataset>>emotion TO emotion>>dataset
    for emotion in observed_emotions:
        emotion_index = observed_emotions.index(emotion)    
        print("Emotion=",emotion)     
        for d_index,d_item in enumerate(dataset_directory):   
            sub_count = 0
            sub_correct = 0         
            for file in glob.glob(d_item %emotion): 
            
                face_data = []
                face_label = []
        
                #print("File=",file)
                file_count +=1
                sub_count += 1              
                
                image = cv2.imread(file)
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
                
                        #feature = extract_feature(landmarks_extracted, image)  
                        feature = extract_feature(face_landmarks, image, w, h)   
                
                        x.append(feature)
                        y.append(emotion)
                        face_data.append(feature)
                        face_label.append(emotion)
                        
                        #print("feature=",feature)
                        
                        # File Prediction
                        #file_pred = np.array(face_data)
                        #file_pred = file_pred.reshape(1, -1)                        
                        #file_lin = model.score(file_pred, face_label)                                         
                        
                        #print(feature)
                        row = feature.tolist()
                        #print(row)
                        pred = predict(row, ort_session)
                        file_lin = 0 
                        if pred == emotion_index:
                            file_lin = 1                   
                        
                        
                        emotion_correct[emotion_index] += file_lin
                        emotion_count[emotion_index] += 1                   
                        file_correct += file_lin;
                        sub_correct += file_lin;
                    
            # Calculate Percentage correct for dataset emotion
            dataset_correct[d_index][emotion_index] = 0;   
            dataset_count[d_index][emotion_index] = 0;
            dataset_accuracies[d_index][emotion_index] = 0;         
            if sub_count > 0:
                dataset_correct[d_index][emotion_index] = sub_correct;   
                dataset_count[d_index][emotion_index] = sub_count;     
                dataset_accuracies[d_index][emotion_index] = ((sub_correct / sub_count) * 100)
            print(d_index,"/",emotion_index," Dataset=",dataset_name[d_index], "Correct=",dataset_accuracies[d_index][emotion_index],"  (",dataset_count[d_index][emotion_index],")")          
              
    print("=============================================")
    print("Yale neutral count=",dataset_count[0][0])
    print("Yale happy count=",dataset_count[0][1])
    print("Yale sadness count=",dataset_count[0][2])
    print("=============================================")       
            
    x_full = np.array(x)
    y_full = y
    return  
            
# make a class prediction for one row of data
def predict(row, ort_session):

    data = json.dumps({'data': row})
    data = np.array(json.loads(data)['data']).astype('float32')
    temp = np.expand_dims(data, axis=0)
    #print(temp.shape)

    ort_inputs = {input_name: temp}
    ort_outs = ort_session.run([output_name], ort_inputs)

    #prediction = int(ort_outs[0] + 0.5)
    prediction = np.argmax(ort_outs[0]) #ort_outs[0]
    
    return prediction
    

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

    
print("Load model")
path = MODEL_PATH
model = onnx.load(path)
onnx.checker.check_model(model)
print('ONNX model loaded')

# Setup runtime
EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
ort_session = onnxruntime.InferenceSession(MODEL_PATH, providers=EP_list)
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name
print(input_name)
print(output_name) 
    
print("Load data")
load_data()
                                 
#y_pred= model.predict(x_full)                     
#accuracy = accuracy_score(y_true=y_full, y_pred=y_pred)
#print(\"Accuracy: {:.2f}%\".format(accuracy*100))
#print(accuracy)

# =========================================================================================================
# Overall
#for emotion in observed_emotions: 
#   index = observed_emotions.index(emotion)       
#   for d_index,d_item in enumerate(dataset_directory):
#       dataset_index = index
#       cycle_correct += dataset_correct[d_index][index]
#       cycle_count += dataset_count[d_index][index]
#   cycle_percentage = ((cycle_correct / cycle_count) * 100)
#   print("Dataset=",dataset_name[dataset_index],"  (",cycle_percentage,"%) (files=",cycle_count) 

cycle_correct=0
cycle_count=0 
for d_index,d_item in enumerate(dataset_directory):
    print("=============================================")
    print("Dataset=",dataset_name[d_index])
    cycle_correct=0
    cycle_count=0
    for emotion in observed_emotions: 
        e_index = observed_emotions.index(emotion)
        if dataset_count[d_index][e_index] > 0:
            print("Emotion=",emotion," (files=",dataset_count[d_index][e_index],")")  
        cycle_correct += dataset_correct[d_index][e_index]
        cycle_count += dataset_count[d_index][e_index]
        #print("Emotion=",emotion,"  (",cycle_percentage,"%) (files=",cycle_count,")") 
    cycle_percentage = 0    
    if cycle_correct > 0 and cycle_count > 0:
        cycle_percentage = ((cycle_correct / cycle_count) * 100)
    print("Dataset=",dataset_name[d_index],"  (",cycle_percentage,"%) (files=",cycle_count,")")     

# =========================================================================================================
# Overall - Neutral
cycle_correct=0
cycle_count=0 
print("=============================================")
for d_index,d_item in enumerate(dataset_directory):
    print("=============================================")
    print("Dataset=",dataset_name[d_index])
    cycle_correct=0
    cycle_count=0
    for emotion in observed_emotions:
        e_index = observed_emotions.index(emotion)  
        if e_index > 0:      
            cycle_correct += dataset_correct[d_index][e_index]
            cycle_count += dataset_count[d_index][e_index]
    cycle_percentage = 0    
    if cycle_correct > 0 and cycle_count > 0:
        cycle_percentage = ((cycle_correct / cycle_count) * 100)
    print("O-N Dataset=",dataset_name[d_index],"  (",cycle_percentage,"%) (files=",cycle_count,")") 

    
print("=============================================")
cycle_correct=0
cycle_count=0 
for d_index,d_item in enumerate(dataset_directory):
    for emotion in observed_emotions:
        e_index = observed_emotions.index(emotion)  
        cycle_correct += dataset_correct[d_index][e_index]
        cycle_count += dataset_count[d_index][e_index]
cycle_percentage = 0
if cycle_correct > 0 and cycle_count > 0:
    cycle_percentage = ((cycle_correct / cycle_count) * 100)
print("Overall Accuracy=",cycle_percentage,"% (",cycle_count,")") 
print("=============================================")
    