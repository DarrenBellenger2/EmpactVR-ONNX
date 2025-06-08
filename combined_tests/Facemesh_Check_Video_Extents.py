import datetime
import time
#import librosa
#import soundfile
import argparse
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
import csv
np.random.seed(4)

import onnx, onnxruntime

import json
from statistics import mode
from pandas import read_csv
np.set_printoptions(threshold=sys.maxsize)

# ==================================================================================
# conda info --envs
# conda activate py312
# cd d:\Documents\PhD\PythonDevelopment\EMPACTVR ONNX November2023\combined_tests
# python Facemesh_Check_Video_Extents.py
# ==================================================================================

# Emotions to observe
#observed_emotions=['neutral', 'happy', 'sad', 'anger', 'fear', 'disgust', 'surprise']
#mlp_emotions=['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
observed_emotions=['neutral', 'happy', 'sad', 'anger', 'fear', 'disgust', 'surprise']
num_emotions = 7
x_full, y_full = [], []

#csv_header = ['File', 'Frame', 'AU1_low', 'AU1_high', 'AU2_low', 'AU2_high', 'AU4_low', 'AU4_high', 'AU6_low', 'AU6_high', 'AU9_low', 'AU9_high', 'AU15_low', 'AU15_high', 'AU20_low', 'AU20_high', 'AU23_low', 'AU23_high', 'AU26_low', 'AU26_high']
csv_header = ['File', 'Frame', 'AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU9', 'AU15', 'AU20', 'AU23', 'AU25', 'AU26']

emotion_correct = [0, 0, 0, 0, 0, 0, 0]
emotion_count = [0, 0, 0, 0, 0, 0, 0]
combined_emotion_correct = [0, 0, 0, 0, 0, 0, 0]
combined_emotion_count = [0, 0, 0, 0, 0, 0, 0]
file_count = 0
file_correct = 0

frame_cut = 0 # 1 #0.5

#Datasets
dataset_name=['ADFES TEST']
#'ADFES', 'Ryerson', 'eNTERFACE05', 'RAVDESS', 'SAVEE', 'CREMA-D']

#dataset_directory=['E:\\Documents\\EmotionDatabases\\Video Libraries\\RAVDESS_640_Labelled\\%s\\*.mp4']
dataset_directory=['E:\\Documents\\EmotionDatabases\\Video Libraries\\RAVDESS_640_Labelled\\%s\\01-01-07-02-02-01-01_1.mp4']


#dataset_directory=['E:\\Documents\\EmotionDatabases\\Video Libraries\\ADFES_Labelled\\%s\\*.m4v']
#dataset_directory=['E:\\Documents\\EmotionDatabases\\Video Libraries\\ADFES_Labelled_MP4_Short\\%s\\*.mp4']

AU1_high = 0
AU1_low = 100

AU2_high = 0
AU2_low = 100

AU4_high = 0
AU4_low = 100

AU6_high = 0
AU6_low = 100

AU7_high = 0
AU7_low = 100

AU9_high = 0
AU9_low = 100

AU15_high = 0
AU15_low = 100

AU20_high = 0
AU20_low = 100

AU23_high = 0
AU23_low = 100

AU26_high = 0
AU26_low = 100

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
                        61,39,37,0,267,269,291,405,314,17,84,181,62,38,12,268,292,317,14,87]                   
# ================================================================================================    

                    
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

def pos_atan2(y, x): 
    ret = math.atan2(y, x)  
    if ret < 0:
        ret = 0
    return ret  
    
def extract_feature(landmarks):
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
    global AU1_high
    global AU1_low
    global AU2_high
    global AU2_low
    global AU4_high
    global AU4_low
    global AU6_high
    global AU6_low  
    global AU7_high
    global AU7_low
    global AU9_high
    global AU9_low
    global AU15_high
    global AU15_low
    global AU20_high
    global AU20_low
    global AU23_high
    global AU23_low
    global AU26_high
    global AU26_low 
    
    x = []
    y = []
    for i in range(0,68): #Store X and Y coordinates in two lists
        #print(landmarks[i][0])
        x.append(float(landmarks[i][0]))
        y.append(float(landmarks[i][1]))    

    #print("Co-ords:" + " (0)=" + str(x[0]) + "/" + str(y[0]) + " (16)=" + str(x[16]) + "/" + str(y[16]) + " (8)=" + str(x[8]) + "/" + str(y[8])) 

    
    # ===============================================================================
    # Evaluate Action Units
    
    
    
    # ===============================================================================
    # AU1: Max = 1.2 & Min = 0.1
    #
    AU1 = (math.atan2(y[17] - y[20], x[20] - x[17]) + math.atan2(y[26] - y[23],x[26] - x[23]))
    if AU1 > AU1_high:
        AU1_high = AU1
    if AU1 < AU1_low:
        AU1_low = AU1
    
    #AU1 = lerp(AU1,0.1,1.2)
    
    # ===============================================================================
    # AU2: Max = 2.3 & Min = 0.9
    #
    AU2 = (math.atan2(y[17] - y[18], x[18] - x[17]) + math.atan2(y[26] - y[25],x[26] - x[25]))
    if AU2 > AU2_high:
        AU2_high = AU2
    if AU2 < AU2_low:
        AU2_low = AU2
    
    
    #AU2 = lerp(AU2,0.9,2.3)
    
    # ===============================================================================
    # AU4: Max = 3 & Min = 1.2
    #
    AU4 = (math.atan2(y[39] - y[21], x[21] - x[39]) + math.atan2(y[42] - y[22],x[42] - x[22]))
    if AU4 > AU4_high:
        AU4_high = AU4
    if AU4 < AU4_low:
        AU4_low = AU4
    
    #AU4 = lerp(AU4,1.2,3)
    
    # ===============================================================================
    # AU6: Max = 1.16 & Min = 0.7
    #
    AU6 = (distance(x[48],y[48],x[66],y[66]) + distance(x[66],y[66],x[54],y[54])) / (distance(x[48],y[48],x[51],y[51]) + distance(x[51],y[51],x[54],y[54]))
    if AU6 > AU6_high:
        AU6_high = AU6
    if AU6 < AU6_low:
        AU6_low = AU6
    
    #AU6 = lerp(AU6,0.7,1.16)
    
    # ===============================================================================
    # AU7: Max = 0.5 & Min = 0.05
    #       
    #AU7 = (distance(x[37],y[37],x[41],y[41]) + distance(x[62],y[62],x[66],y[66]) + distance(x[63],y[63],x[65],y[65])) / (2 * distance(x[36],y[36],x[45],y[45]))
    AU7_right = ((distance(x[37],y[37],x[41],y[41]) + distance(x[38],y[38],x[40],y[40])) / (2 * distance(x[36],y[36],x[39],y[39])))
    AU7_left = ((distance(x[43],y[43],x[47],y[47]) + distance(x[44],y[44],x[46],y[46])) / (2 * distance(x[42],y[45],x[45],y[45])))
    AU7 = (AU7_right + AU7_left / 2)      
    if AU7 > AU7_high:
        AU7_high = AU7
    if AU7 < AU7_low:
        AU7_low = AU7
    
    
    
    #AU7 = lerp(AU7,0.05,0.5)

    # ===============================================================================
    # AU9: Max = 2.5 & Min = 1
    #    
    #AU9 = (math.atan2(y[50] - y[33], x[33] - x[50]) + math.atan2(y[52] - y[33],x[52] - x[33]))
    AU9 = (math.atan2(y[32] - y[33], x[33] - x[32]) + math.atan2(y[34] - y[33],x[34] - x[33]))  
    if AU9 > AU9_high:
        AU9_high = AU9
    if AU9 < AU9_low:
        AU9_low = AU9
    
    
    #AU9 = lerp(AU9,1,2.5)
    
    # ===============================================================================
    # AU15: Max = 1 & Min = -0.5 (-0.8)
    #          
    AU15 = (math.atan2(y[48] - y[60], x[60] - x[48]) + math.atan2(y[54] - y[64],x[54] - x[64]))
    if AU15 > AU15_high:
        AU15_high = AU15
    if AU15 < AU15_low:
        AU15_low = AU15
    
    #AU15 = lerp(AU15,-0.5,1)

    # ===============================================================================
    # AU20: Max = 3 & Min = 0
    #         
    AU20 = (pos_atan2(y[59] - y[65], x[65] - x[59]) + pos_atan2(y[55] - y[67],x[55] - x[67]) + pos_atan2(y[59] - y[66],x[66] - x[59]) + pos_atan2(y[59] - y[67],x[67] - x[59]) + pos_atan2(y[55] - y[65],x[55] - x[65]))
    if AU20 > AU20_high:
        AU20_high = AU20
    if AU20 < AU20_low:
        AU20_low = AU20
    
    
    #AU20 = lerp(AU20,0,3)

    # ===============================================================================
    # AU23: Max = 9 & Min = 2
    #          
    AU23 = (pos_atan2(y[49] - y[50], x[50] - x[49]) + pos_atan2(y[53] - y[52],x[53] - x[52]) + pos_atan2(y[61] - y[49],x[61] - x[49]) + pos_atan2(y[63] - y[53],x[53] - x[63]) + pos_atan2(y[58] - y[59],x[58] - x[59]) + pos_atan2(y[56] - y[55],x[55] - x[56]) + pos_atan2(y[60] - y[51],x[51] - x[60]) + pos_atan2(y[64] - y[51],x[64] - x[51]) + pos_atan2(y[57] - y[60],x[57] - x[60]) + pos_atan2(y[57] - y[64],x[64] - x[57]) + pos_atan2(y[62] - y[49],x[62] - x[49]) + pos_atan2(y[62] - y[53],x[53] - x[62]) + pos_atan2(y[57] - y[60],x[57] - x[60]) + pos_atan2(y[57] - y[64],x[64] - x[57]))
    if AU23 > AU23_high:
        AU23_high = AU23
    if AU23 < AU23_low:
        AU23_low = AU23
    
    
    #AU23 = lerp(AU23,2,9)
    
    # ===============================================================================
    # AU26: Max = 1 & Min = 0
    #                 
    AU26 = (distance(x[61],y[61],x[67],y[67]) + distance(x[62],y[62],x[66],y[66]) + distance(x[63],y[63],x[65],y[65])) / (distance(x[36],y[36],x[45],y[45]))
    if AU26 > AU26_high:
        AU26_high = AU26
    if AU26 < AU26_low:
        AU26_low = AU26
    
    
    
    #AU26 = lerp(AU26,0,1)        
    #if AU26 > 0.1:
    #    AU25 = 1
    
    result = np.array([AU1,AU2,AU4,AU6,AU7,AU9,AU15,AU20,AU23,AU25,AU26])
    #result = np.array([AU1_low, AU1_high, AU2_low, AU2_high, AU4_low, AU4_high, AU6_low, AU6_high, AU9_low, AU9_high, AU15_low, AU15_high, AU20_low, AU20_high, AU23_low, AU23_high, AU26_low, AU26_high])  
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
    
   
def process_image(file, emotion):

    clip = VideoFileClip(file) 
    subclip = clip.subclip(frame_cut,(frame_cut * -1))
    subclip.write_videofile("temp.mp4") 
   
    cap = cv2.VideoCapture("temp.mp4")                
    while(cap.isOpened()):
        ret, image = cap.read()
        if ret == False:
            break                
  
        h, w, c = image.shape                   
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)                  

        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                brect = calc_bounding_rect(image, face_landmarks)
                left_eye, right_eye = None, None
                #if refine_landmarks:
                #    left_eye, right_eye = calc_iris_min_enc_losingCircle(
                #        debug_image,
                #        face_landmarks,
                #    )

                landmarks_extracted = []
                for index in landmark_points_68:
                    #x_val = int(face_landmarks.landmark[index].x * (w / 1.5))
                    #y_val = int(face_landmarks.landmark[index].y * 480)
                    x_val = float(face_landmarks.landmark[index].x * w)
                    y_val = float(face_landmarks.landmark[index].y * h)
                    landmarks_extracted.append((x_val, y_val))

                #feature = extract_feature(clahe_image)
                feature = extract_feature(landmarks_extracted)   
        
    cap.release()        

    return
    
def load_data():
    global x_full, y_full
    #x, y = [], []
    global file_count, file_correct  
    sub_count = 0
    sub_correct = 0
    combined_sub_correct = 0
    index = 0
    frame_no = 0

    # Changed order from dataset>>emotion TO emotion>>dataset
    with open('mediapipe_video_extents.csv', 'w', encoding='UTF8', newline='') as f:

        writer = csv.writer(f)
        writer.writerow(csv_header)
        
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

                    #process_image(file, emotion)    
                    clip = VideoFileClip(file) 
                    subclip = clip #clip.subclip(frame_cut,(frame_cut * -1))
                    subclip.write_videofile("temp.mp4")                     
                
                    cap = cv2.VideoCapture("temp.mp4")   
                    frame_no = 0                    
                    while(cap.isOpened()):
                        frame_no += 1
                        ret, image = cap.read()
                        if ret == False:
                            break                
                  
                        h, w, c = image.shape                   
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        results = face_mesh.process(image)                  

                        if results.multi_face_landmarks is not None:
                            for face_landmarks in results.multi_face_landmarks:
                                brect = calc_bounding_rect(image, face_landmarks)
                                left_eye, right_eye = None, None
                                #if refine_landmarks:
                                #    left_eye, right_eye = calc_iris_min_enc_losingCircle(
                                #        debug_image,
                                #        face_landmarks,
                                #    )

                                landmarks_extracted = []
                                for index in landmark_points_68:
                                    #x_val = int(face_landmarks.landmark[index].x * (w / 1.5))
                                    #y_val = int(face_landmarks.landmark[index].y * 480)
                                    x_val = float(face_landmarks.landmark[index].x * w)
                                    y_val = float(face_landmarks.landmark[index].y * h)
                                    landmarks_extracted.append((x_val, y_val))

                                #feature = extract_feature(clahe_image)
                                feature = extract_feature(landmarks_extracted)   

                                csv_data_row = feature
                                csv_data_row = np.insert(csv_data_row, 0, observed_emotions.index(emotion), axis=0)
                                #csv_data_row = np.insert(csv_data_row, 0, file_count, axis=0)
           
                                writer.writerow((file,frame_no,csv_data_row[1],csv_data_row[2],csv_data_row[3],csv_data_row[4],csv_data_row[5],csv_data_row[6],csv_data_row[7],csv_data_row[8],csv_data_row[9],csv_data_row[10],csv_data_row[11]))                               

                    cap.release()        
                    
    return

args = get_args()
max_num_faces = args.max_num_faces
refine_landmarks = args.refine_landmarks
min_detection_confidence = args.min_detection_confidence
min_tracking_confidence = args.min_tracking_confidence
use_brect = args.use_brect  
   
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=max_num_faces,
    refine_landmarks=refine_landmarks,
    min_detection_confidence=min_detection_confidence,
    min_tracking_confidence=min_tracking_confidence,
)   
    
print("Load data")
load_data()

print("AU Values")
print("AU1 Low: ", AU1_low, "   AU1_high: ", AU1_high)
print("AU2 Low: ", AU2_low, "   AU2_high: ", AU2_high)
print("AU4 Low: ", AU4_low, "   AU4_high: ", AU4_high)
print("AU6 Low: ", AU6_low, "   AU6_high: ", AU6_high)
print("AU7 Low: ", AU7_low, "   AU7_high: ", AU7_high)
print("AU9 Low: ", AU9_low, "   AU9_high: ", AU9_high)
print("AU15 Low: ", AU15_low, "   AU15_high: ", AU15_high)
print("AU20 Low: ", AU20_low, "   AU20_high: ", AU20_high)
print("AU23 Low: ", AU23_low, "   AU23_high: ", AU23_high)
print("AU26 Low: ", AU26_low, "   AU26_high: ", AU26_high)
