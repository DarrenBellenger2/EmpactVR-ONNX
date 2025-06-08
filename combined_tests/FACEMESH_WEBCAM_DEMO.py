import datetime
import time
import argparse
import librosa
#import soundfile
import mediapipe as mp

#from mediapipe.tasks import python
#from mediapipe.tasks.python import vision

from moviepy.editor import *
from pydub import AudioSegment
import numpy as np
import subprocess
import sys
import os, glob, math, random, pickle
from pathlib import Path
import cv2
#import dlib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from tkinter import *
from tkinter import ttk
from ttkthemes import ThemedTk
from PIL import Image, ImageTk


np.random.seed(4)

KeyFrame = 4

# ==================================================================================
# conda info --envs
# conda activate py312
# cd d:\Documents\PhD\PythonDevelopment\EMPACTVR ONNX November2023\combined_tests
# python FACEMESH_WEBCAM_DEMO.py
# ==================================================================================

#Set up some required objects
#detector = dlib.get_frontal_face_detector() #Face detector
#predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever 

# ======================================================================================================================
# Face
# Right brow (70,63,105,66,55)
# Left brow (285,296,334,293,300)
# Nose (168,197,5,4,75,97,2,326,305)
# Right Eye (33,160,158,133,153,144)
# Left Eye ()
# Mouth

#landmark_points_68 = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,
#                        70,63,105,66,55,
#                        285,296,334,293,300,
#                        168,197,5,4,75,97,2,326,305,
#                        33,160,158,133,153,144,
#                       362,385,387,263,373,380,
#                        61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87]


# ======================================================================================================================
# Face (162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389)
# Right brow (70,63,105,66,55) - (46,53,52,65,55)
# Left brow (285,296,334,293,300)  - (285,295,282,283,276)  

#       27  28   29 30  31  32 33  34  35
# Nose (168,197, 5, 4,  75, 97, 2, 326, 305)  - 
#      (168,197, 5, 4, 218, 60, 2, 290, 438)
#      (168,197, 5, 4, 44, 167, 2, 393, 274)
#                      --   --     --    --


# Right Eye (33,160,158,133,153,144) - (33,159,159,133,145,145)
# Left Eye (362,385,387,263,373,380) - (362,386,386,263,374,374)



#        48 49 50 51 52  53  54  55  56  57 58 59  60 61 62 63  64  65  66 67
# Mouth (61,39,37, 0,267,269,291,404,314,17,84,181,78,82,13,312,308,317,14,87) -

# ================================================================================================ 
#       (43,73,37, 0,267,303,273,405,314,17,84,180,76,38,12,268,306,317,14,87)   
#                                                     -- -- ---
#                                                  --           --               AU15 XXX
# ================================================================================================ 

#        43,73,37, 0,267,303,273,404,315,16,85,180,76,38,12,268,306,316,15,86 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#                                404 315 16 85 180                  316 15 86           AU20xxxxxxxxxxxxxxxxxxx


landmark_points_68 = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,
                        46,53,52,65,55,
                        285,295,282,283,276,
                        168,197, 5, 4, 44, 167, 2, 393, 274,
                        33,159,159,133,145,145,
                        362,386,386,263,374,374,
                        43,73,37, 0,267,303,273,404,314,17,84,180,76,38,12,268,306,317,14,87]

#AU15 = (math.atan2(y[48] - y[60], x[60] - x[48]) + math.atan2(y[54] - y[64],x[54] - x[64]))

# ==============================================================================================
#AU20 = 0
#AU20 += pos_atan2(y[59] - y[65], x[65] - x[59]) + pos_atan2(y[55] - y[67],x[55] - x[67])
#AU20 += pos_atan2(y[59] - y[66],x[66] - x[59]) ??????????????????????????????????????????????????????????????
#AU20 += pos_atan2(y[59] - y[67],x[67] - x[59]) + pos_atan2(y[55] - y[65],x[55] - x[65])
# ==============================================================================================



# ==============================================================================================
#AU23 = 0
#AU23 += pos_atan2(y[49] - y[50], x[50] - x[49]) + pos_atan2(y[53] - y[52],x[53] - x[52]) 
#AU23 += pos_atan2(y[61] - y[49],x[61] - x[49]) + pos_atan2(y[63] - y[53],x[53] - x[63])
# ==============================================================================================

#AU23 += pos_atan2(y[58] - y[59],x[58] - x[59]) + pos_atan2(y[56] - y[55],x[55] - x[56]) 
#AU23 += pos_atan2(y[60] - y[51],x[51] - x[60]) + pos_atan2(y[64] - y[51],x[64] - x[51]) 
#AU23 += pos_atan2(y[57] - y[60],x[57] - x[60]) + pos_atan2(y[57] - y[64],x[64] - x[57]) 
#AU23 += pos_atan2(y[62] - y[49],x[62] - x[49]) + pos_atan2(y[62] - y[53],x[53] - x[62]) 
#AU23 += pos_atan2(y[57] - y[60],x[57] - x[60]) + pos_atan2(y[57] - y[64],x[64] - x[57]))

    
# ======================================================================================================================
                        
def distance(From_X, From_Y, To_X, To_Y): 
    dist = math.sqrt( math.pow(To_X - From_X,2) + math.pow(To_Y - From_Y,2) )
    return dist

def lerp(x, a, b): 
    ret = (x - a) / (b - a)
    if ret < 0:
        ret = 0
    return round(ret,2)
    
def pos_atan2(y, x): 
    ret = math.atan2(y, x)  
    #if ret < 0:
    #    ret = 0
    return ret  
 
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
    dist = math.sqrt( math.pow(To_X - From_X,2) + math.pow(To_Y - From_Y,2) )
    return dist

def lerp(x, a, b): 
    ret = (x - a) / (b - a)
    if ret < 0:
        ret = 0
    return round(ret,2)
    
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

def process_image(image, landmarks, all_landmarks):

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
 

    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #clahe_image = clahe.apply(gray)
    
    #landmark_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       
    landmark_image = image

    #detections = detector(clahe_image, 1)   
    #if len(detections) > 0:
    #    detections = detector(clahe_image, 1) 
    #    for k,d in enumerate(detections): #For all detected face instances individually
    #        shape = predictor(clahe_image, d) #Draw Facial Landmarks with the predictor class
    #        #print("d:",d.left())
    #        cv2.rectangle(landmark_image, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 1)
    #        x = []
    #        y = []

    x = []
    y = []
    #for i in range(0,68): #Store X and Y coordinates in two lists
    #    x.append(float(landmarks[i][0]))
    #    y.append(float(landmarks[i][1]))
    #    #cv2.circle(landmark_image, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), 1)
    #    cv2.circle(landmark_image, (int(landmarks[i][0]), int(landmarks[i][1])), 2, (0, 255, 0), 1)

    for i in range(0,468): #Store X and Y coordinates in two lists
        x.append(float(all_landmarks[i][0]))
        y.append(float(all_landmarks[i][1]))
        cv2.circle(landmark_image, (int(all_landmarks[i][0]), int(all_landmarks[i][1])), 1, (0, 0, 0), 1)


    
    #print("Co-ords:" + " (0)=" + str(x[0]) + "/" + str(y[0]) + " (16)=" + str(x[16]) + "/" + str(y[16]) + " (8)=" + str(x[8]) + "/" + str(y[8])) 

        
    # ===============================================================================
    # Evaluate Action Units

    # ===============================================================================
    # AU1: Max = 1.2 & Min = 0.1
    #
    #AU1 = (math.atan2(y[17] - y[20], x[20] - x[17]) + math.atan2(y[26] - y[23],x[26] - x[23]))
    
    #AU1 = (math.atan2(y[70] - y[66], x[66] - x[70]) + math.atan2(y[300] - y[296],x[300] - x[296]))
    #AU1 += (math.atan2(y[46] - y[65], x[65] - x[46]) + math.atan2(y[276] - y[295],x[276] - x[295]))
    #AU1 += (math.atan2(y[225] - y[222], x[222] - x[225]) + math.atan2(y[445] - y[442],x[445] - x[442]))

    AU1 = (math.atan2(y[71] - y[66], x[66] - x[71]) + math.atan2(y[301] - y[296],x[301] - x[296]))
    AU1 += (math.atan2(y[70] - y[65], x[65] - x[70]) + math.atan2(y[300] - y[295],x[300] - x[295]))
    
    AU1 = round(AU1, 4)
    #AU1 = lerp(AU1,0,1)




    #cv2.circle(landmark_image, (int(all_landmarks[70][0]), int(all_landmarks[70][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[66][0]), int(all_landmarks[66][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[300][0]), int(all_landmarks[300][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[296][0]), int(all_landmarks[296][1])), 2, (0, 255, 0), 1)

    #cv2.circle(landmark_image, (int(all_landmarks[46][0]), int(all_landmarks[46][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[65][0]), int(all_landmarks[65][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[276][0]), int(all_landmarks[276][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[295][0]), int(all_landmarks[295][1])), 2, (0, 255, 0), 1)

    #cv2.circle(landmark_image, (int(all_landmarks[225][0]), int(all_landmarks[225][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[222][0]), int(all_landmarks[222][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[445][0]), int(all_landmarks[445][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[442][0]), int(all_landmarks[442][1])), 2, (0, 255, 0), 1)

    
    
    #cv2.circle(landmark_image, (int(all_landmarks[105][0]), int(all_landmarks[105][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[52][0]), int(all_landmarks[52][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[223][0]), int(all_landmarks[223][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[334][0]), int(all_landmarks[334][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[282][0]), int(all_landmarks[282][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[443][0]), int(all_landmarks[443][1])), 2, (0, 255, 0), 1)



    
    # ===============================================================================
    # AU2: Max = 2.3 & Min = 0.9
    #
    #AU2 = (math.atan2(y[17] - y[18], x[18] - x[17]) + math.atan2(y[26] - y[25],x[26] - x[25]))
    AU2 = (math.atan2(y[70] - y[63], x[63] - x[70]) + math.atan2(y[300] - y[293],x[300] - x[293]))
    AU2 += (math.atan2(y[46] - y[53], x[53] - x[46]) + math.atan2(y[276] - y[283],x[276] - x[283]))
    AU2 += (math.atan2(y[225] - y[224], x[224] - x[225]) + math.atan2(y[445] - y[444],x[445] - x[444]))
    AU2 = round(AU2, 4) 
    #AU2 = lerp(AU2,0.5,2)
    
    # ===============================================================================
    # AU4: Max = 3 & Min = 1.2
    #
    #AU4 = (math.atan2(y[39] - y[21], x[21] - x[39]) + math.atan2(y[42] - y[22],x[42] - x[22]))
    AU4 = (math.atan2(y[133] - y[55], x[55] - x[133]) + math.atan2(y[362] - y[285],x[362] - x[285]))
    AU4 += (math.atan2(y[153] - y[55], x[55] - x[153]) + math.atan2(y[380] - y[285],x[380] - x[285]))
    AU4 += (math.atan2(y[158] - y[55], x[55] - x[158]) + math.atan2(y[385] - y[285],x[385] - x[285]))
    AU4 = round(AU4, 4)
    #AU4 = lerp(AU4,1.5,3)
    
    
    #cv2.circle(landmark_image, (int(all_landmarks[153][0]), int(all_landmarks[153][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[380][0]), int(all_landmarks[380][1])), 2, (0, 255, 0), 1)

    #cv2.circle(landmark_image, (int(all_landmarks[133][0]), int(all_landmarks[133][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[55][0]), int(all_landmarks[55][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[362][0]), int(all_landmarks[362][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[285][0]), int(all_landmarks[285][1])), 2, (0, 255, 0), 1)
    
    
    # ===============================================================================
    # AU6: Max = 1.16 & Min = 0.7
    #
    #AU6 = (distance(x[48],y[48],x[66],y[66]) + distance(x[66],y[66],x[54],y[54])) / (distance(x[48],y[48],x[51],y[51]) + distance(x[51],y[51],x[54],y[54]))
    #AU6 = (distance(x[76],y[76],x[14],y[14]) + distance(x[14],y[14],x[306],y[306])) / (distance(x[76],y[76],x[0],y[0]) + distance(x[0],y[0],x[306],y[306]))
    AU6 = (distance(x[216],y[216],x[12],y[12]) + distance(x[12],y[12],x[436],y[436])) / (distance(x[74],y[74],x[13],y[13]) + distance(x[13],y[13],x[304],y[304]))
    AU6 = round(AU6, 4)

    #cv2.circle(landmark_image, (int(all_landmarks[216][0]), int(all_landmarks[216][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[436][0]), int(all_landmarks[436][1])), 2, (0, 255, 0), 1)

    #cv2.circle(landmark_image, (int(all_landmarks[12][0]), int(all_landmarks[12][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[74][0]), int(all_landmarks[74][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[13][0]), int(all_landmarks[13][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[304][0]), int(all_landmarks[304][1])), 2, (0, 255, 0), 1)


    #AU6 = lerp(AU6,0.7,1.3)
    
    # ===============================================================================
    # AU7: Max = 0.5 & Min = 0.05
    #       
    #AU7_right = ((distance(x[37],y[37],x[41],y[41]) + distance(x[38],y[38],x[40],y[40])) / (2 * distance(x[36],y[36],x[39],y[39])))
    #AU7_left = ((distance(x[43],y[43],x[47],y[47]) + distance(x[44],y[44],x[46],y[46])) / (2 * distance(x[42],y[45],x[45],y[45])))
    AU7_right = ((distance(x[159],y[159],x[145],y[145]) + distance(x[159],y[159],x[145],y[145])) / (2 * distance(x[33],y[33],x[133],y[133])))
    AU7_left = ((distance(x[386],y[386],x[374],y[374]) + distance(x[386],y[386],x[374],y[374])) / (2 * distance(x[362],y[362],x[263],y[263])))
    AU7 = (AU7_right + AU7_left / 2)    
    
    AU7 = round(AU7, 4)
    #AU7 = lerp(AU7,0.05,0.8)

    # ===============================================================================
    # AU9: Max = 2.5 & Min = 1
    #
    # DLIB AU9 = (math.atan2(y[50] - y[33], x[33] - x[50]) + math.atan2(y[52] - y[33],x[52] - x[33]))   
    #      AU9 = (math.atan2(y[32] - y[33], x[33] - x[32]) + math.atan2(y[34] - y[33],x[34] - x[33]))
    # 0.2     AU9 = (math.atan2(y[167] - y[2], x[2] - x[167]) + math.atan2(y[393] - y[2],x[393] - x[2]))
    
    # 0.5
    AU9 = (math.atan2(y[37] - y[2], x[2] - x[37]) + math.atan2(y[267] - y[2],x[267] - x[2]))
    
    
    # 0.3     AU9 = (math.atan2(y[39] - y[2], x[2] - x[39]) + math.atan2(y[269] - y[2],x[269] - x[2]))
    
    AU9 = round(AU9, 4)
    #AU9 = lerp(AU9,0.3,1.45)
    
    #cv2.circle(landmark_image, (int(all_landmarks[167][0]), int(all_landmarks[167][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[2][0]), int(all_landmarks[2][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[393][0]), int(all_landmarks[393][1])), 2, (0, 255, 0), 1)

    #cv2.circle(landmark_image, (int(all_landmarks[141][0]), int(all_landmarks[141][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[370][0]), int(all_landmarks[370][1])), 2, (0, 255, 0), 1)
    
    
    # ===============================================================================
    # AU15: Max = 1 & Min = -0.5 (-0.8)
    #          
    #AU15 = (math.atan2(y[48] - y[60], x[60] - x[48]) + math.atan2(y[54] - y[64],x[54] - x[64]))
    AU15 = (math.atan2(y[43] - y[76], x[76] - x[43]) + math.atan2(y[273] - y[306],x[273] - x[306]))
    AU15 = round(AU15, 4)
    #AU15 = lerp(AU15,-0.6,1)

    #cv2.circle(landmark_image, (int(all_landmarks[43][0]), int(all_landmarks[43][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[76][0]), int(all_landmarks[76][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[273][0]), int(all_landmarks[273][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[306][0]), int(all_landmarks[306][1])), 2, (0, 255, 0), 1)
    
    
    
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

    #cv2.circle(landmark_image, (int(all_landmarks[180][0]), int(all_landmarks[180][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[317][0]), int(all_landmarks[317][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[87][0]), int(all_landmarks[87][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[404][0]), int(all_landmarks[404][1])), 2, (0, 255, 0), 1)

    #cv2.circle(landmark_image, (int(all_landmarks[73][0]), int(all_landmarks[73][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[267][0]), int(all_landmarks[267][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[303][0]), int(all_landmarks[303][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[37][0]), int(all_landmarks[37][1])), 2, (0, 255, 0), 1)

    
    AU20 = round(AU20, 4)
    #AU20 = lerp(AU20,0.7,3.3)

    # ===============================================================================
    # AU23: Max = 9 & Min = 2
    #          
    #AU23 = (pos_atan2(y[49] - y[50], x[50] - x[49]) + pos_atan2(y[53] - y[52],x[53] - x[52]) + pos_atan2(y[61] - y[49],x[61] - x[49]) + pos_atan2(y[63] - y[53],x[53] - x[63]) + pos_atan2(y[58] - y[59],x[58] - x[59]) + pos_atan2(y[56] - y[55],x[55] - x[56]) + pos_atan2(y[60] - y[51],x[51] - x[60]) + pos_atan2(y[64] - y[51],x[64] - x[51]) + pos_atan2(y[57] - y[60],x[57] - x[60]) + pos_atan2(y[57] - y[64],x[64] - x[57]) + pos_atan2(y[62] - y[49],x[62] - x[49]) + pos_atan2(y[62] - y[53],x[53] - x[62]) + pos_atan2(y[57] - y[60],x[57] - x[60]) + pos_atan2(y[57] - y[64],x[64] - x[57]))
    AU23 = (pos_atan2(y[73] - y[37], x[37] - x[73]) + pos_atan2(y[303] - y[267],x[303] - x[267]) + pos_atan2(y[38] - y[73],x[38] - x[73]) + pos_atan2(y[268] - y[303],x[303] - x[268]) + pos_atan2(y[84] - y[180],x[84] - x[180]) + pos_atan2(y[314] - y[404],x[404] - x[314]) + pos_atan2(y[76] - y[0],x[0] - x[76]) + pos_atan2(y[306] - y[0],x[306] - x[0]) + pos_atan2(y[17] - y[76],x[17] - x[76]) + pos_atan2(y[17] - y[306],x[306] - x[17]) + pos_atan2(y[12] - y[73],x[12] - x[73]) + pos_atan2(y[12] - y[303],x[303] - x[12]) + pos_atan2(y[17] - y[76],x[17] - x[76]) + pos_atan2(y[17] - y[306],x[306] - x[17]))
    AU23 = round(AU23, 4)
    #AU23 = lerp(AU23,3,10)

    #cv2.circle(landmark_image, (int(all_landmarks[73][0]), int(all_landmarks[73][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[37][0]), int(all_landmarks[37][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[303][0]), int(all_landmarks[303][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[267][0]), int(all_landmarks[267][1])), 2, (0, 255, 0), 1)

    #cv2.circle(landmark_image, (int(all_landmarks[38][0]), int(all_landmarks[38][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[268][0]), int(all_landmarks[268][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[84][0]), int(all_landmarks[84][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[180][0]), int(all_landmarks[180][1])), 2, (0, 255, 0), 1)

    #cv2.circle(landmark_image, (int(all_landmarks[314][0]), int(all_landmarks[314][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[404][0]), int(all_landmarks[404][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[76][0]), int(all_landmarks[76][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[0][0]), int(all_landmarks[0][1])), 2, (0, 255, 0), 1)

    #cv2.circle(landmark_image, (int(all_landmarks[306][0]), int(all_landmarks[306][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[17][0]), int(all_landmarks[17][1])), 2, (0, 255, 0), 1)
    #cv2.circle(landmark_image, (int(all_landmarks[12][0]), int(all_landmarks[12][1])), 2, (0, 255, 0), 1)
    
    # ===============================================================================
    # AU26: Max = 1 & Min = 0
    #                 
    #AU26 = (distance(x[61],y[61],x[67],y[67]) + distance(x[62],y[62],x[66],y[66]) + distance(x[63],y[63],x[65],y[65])) / (distance(x[36],y[36],x[45],y[45]))
    
    #AU26 = (distance(x[38],y[38],x[86],y[86]) + distance(x[12],y[12],x[15],y[15]) + distance(x[268],y[268],x[316],y[316])) / (distance(x[33],y[33],x[263],y[263]))
    
    eye_width = 0
    lip_height = round((3 * distance(x[11],y[11],x[12],y[12])),0)
    #mouth_height = round((distance(x[72],y[72],x[85],y[85]) + distance(x[11],y[11],x[16],y[16]) + distance(x[302],y[302],x[315],y[315])),0)
    mouth_height = round((distance(x[38],y[38],x[86],y[86]) + distance(x[12],y[12],x[15],y[15]) + distance(x[268],y[268],x[316],y[316])),0)
    if mouth_height > lip_height:
        eye_width = round((2 * distance(x[33],y[33],x[263],y[263])),0)
        AU26 = mouth_height / eye_width
        AU26 = round(AU26, 2)
        #AU26 = lerp(AU26,0,1)        
        if AU26 > 0.1:
           AU25 = 1
   
    cv2.circle(landmark_image, (int(all_landmarks[38][0]), int(all_landmarks[38][1])), 2, (0, 255, 0), 1)
    cv2.circle(landmark_image, (int(all_landmarks[86][0]), int(all_landmarks[86][1])), 2, (0, 255, 0), 1)
    cv2.circle(landmark_image, (int(all_landmarks[12][0]), int(all_landmarks[12][1])), 2, (0, 255, 0), 1)
    cv2.circle(landmark_image, (int(all_landmarks[15][0]), int(all_landmarks[15][1])), 2, (0, 255, 0), 1)
    cv2.circle(landmark_image, (int(all_landmarks[268][0]), int(all_landmarks[268][1])), 2, (0, 255, 0), 1)
    cv2.circle(landmark_image, (int(all_landmarks[316][0]), int(all_landmarks[316][1])), 2, (0, 255, 0), 1)
    cv2.circle(landmark_image, (int(all_landmarks[33][0]), int(all_landmarks[33][1])), 2, (0, 255, 0), 1)
    cv2.circle(landmark_image, (int(all_landmarks[263][0]), int(all_landmarks[263][1])), 2, (0, 255, 0), 1)

    print("AU26: ", AU26, " width:", eye_width, " height:", mouth_height, "lip height:", lip_height )
    

    result = np.array([AU1,AU2,AU4,AU6,AU7,AU9,AU15,AU20,AU23,AU25,AU26])   
    
    return result, landmark_image

    
# ========================================================================================================  
args = get_args()
max_num_faces = args.max_num_faces
refine_landmarks = args.refine_landmarks
min_detection_confidence = args.min_detection_confidence
min_tracking_confidence = args.min_tracking_confidence
use_brect = args.use_brect

print("=============================================================================================")
print("Setup Face Mesh")
mp_face_mesh = mp.solutions.face_mesh
#face_mesh = mp_face_mesh.FaceMesh(
#    max_num_faces=max_num_faces,
#    refine_landmarks=refine_landmarks,
#    min_detection_confidence=min_detection_confidence,
#    min_tracking_confidence=min_tracking_confidence,
#)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=max_num_faces,
    refine_landmarks=True,
    min_detection_confidence=min_detection_confidence,
    min_tracking_confidence=min_tracking_confidence,
)
    
# ========================================================================================================  

face_data = []
face_label = []
keyframe_face_data = []
keyframe_face_label = []    

# Create an instance of TKinter Window or frame
win = Toplevel #Tk()
win = ThemedTk(theme="scidgrey")


win.title("Webcam")
win.geometry("1024x768")

# Create a Label to capture the Video frames
label =Label(win)
label.grid(row=0, column=0)
cap= cv2.VideoCapture(0)

#text_box = Text(win,height = 5, width = 25, bg = "light yellow")

# Define function to show frame
def show_frames():

    # Get the latest frame and convert into Image
    cv2image= cv2.cvtColor(cap.read()[1],cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    ret, image = cap.read()


    r = 1024.0 / cv2image.shape[1]
    dim = (1024, int(cv2image.shape[0] * r))
    # perform the actual resizing of the image using cv2 resize
    resized = cv2.resize(cv2image, dim, interpolation=cv2.INTER_AREA)

    
    h, w, c = resized.shape                   
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(resized)    

    if results.multi_face_landmarks is not None:
        for face_landmarks in results.multi_face_landmarks:
            brect = calc_bounding_rect(resized, face_landmarks)
            left_eye, right_eye = None, None

            landmarks_extracted = []
            for index in landmark_points_68:
                x_val = float(face_landmarks.landmark[index].x * w)
                y_val = float(face_landmarks.landmark[index].y * h)
                landmarks_extracted.append((x_val, y_val))

            full_landmarks_extracted = []
            for i in range(0,468):
                x_val = float(face_landmarks.landmark[i].x * w)
                y_val = float(face_landmarks.landmark[i].y * h)
                full_landmarks_extracted.append((x_val, y_val))

                
            #feature = extract_facial_feature(landmarks_extracted)                        
            #face_data.append(feature)
            #face_label.append(emotion)
    
            feature, landmarks_img = process_image(resized, landmarks_extracted, full_landmarks_extracted) 
            #print("AU1: ", feature[0], "   AU2: ", feature[1], "   AU4: ", feature[2])
            #print("AU6: ", feature[3], "   AU7: ", feature[4], "   AU9: ", feature[5])
            #print("AU15: ", feature[6], "   AU20: ", feature[7], "   AU23: ", feature[8])
            #print("AU20: ", feature[7])
            print("AU7: ", feature[4], "    AU26: ", feature[10])
   
    # Convert image to PhotoImage
    imgtk = ImageTk.PhotoImage(image=Image.fromarray(landmarks_img))   
    #imgtk = ImageTk.PhotoImage(image = img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
   
    #text_box.insert(END, "some text...")

    # Repeat after an interval to capture continiously
    label.after(20, show_frames)

#text_box.pack()

show_frames()
win.mainloop()

