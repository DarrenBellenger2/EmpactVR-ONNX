import numpy as np
import os, glob, math, random, pickle
import cv2
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

MODEL_PATH = '..\models\ImageClassifier_Tanh.onnx'

# ==================================================================================
# conda activate py37
# cd d:\Documents\PhD\PythonDevelopment\EMPACTVR ONNX November2023\model_testing
# python IMAGE_LOAD_TEST_TANH_V1_OHE.py
# ==================================================================================

# Emotions to observe
observed_emotions=['neutral', 'happy', 'sad', 'anger', 'fear', 'disgust', 'surprise']
x_full, y_full = [], []

emotion_correct = [0, 0, 0, 0, 0, 0, 0]
emotion_count = [0, 0, 0, 0, 0, 0, 0]
file_count = 0
file_correct = 0

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

def distance(From_X, From_Y, To_X, To_Y): 
    dist = math.sqrt( math.pow(To_X - From_X,2) + math.pow(To_Y - From_Y,2) )
    return dist

def lerp(x, a, b): 
    ret = (x - a) / (b - a)
    if ret < 0:
        ret = 0
    return round(ret,2)
    
def extract_feature(image):
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

    detections = detector(image, 1) 
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
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
        AU1 = lerp(AU1,0.1,1.2)
        
        # ===============================================================================
        # AU2: Max = 2.3 & Min = 0.9
        #
        AU2 = (math.atan2(y[17] - y[18], x[18] - x[17]) + math.atan2(y[26] - y[25],x[26] - x[25]))
        AU2 = lerp(AU2,0.9,2.3)
        
        # ===============================================================================
        # AU4: Max = 3 & Min = 1.2
        #
        AU4 = (math.atan2(y[39] - y[21], x[21] - x[39]) + math.atan2(y[42] - y[22],x[42] - x[22]))
        AU4 = lerp(AU4,1.2,3)
        
        # ===============================================================================
        # AU6: Max = 1.16 & Min = 0.7
        #
        AU6 = (distance(x[48],y[48],x[66],y[66]) + distance(x[66],y[66],x[54],y[54])) / (distance(x[48],y[48],x[51],y[51]) + distance(x[51],y[51],x[54],y[54]))
        AU6 = lerp(AU6,0.7,1.16)
        
        # ===============================================================================
        # AU7: Max = 0.5 & Min = 0.05
        #       
        #AU7 = (distance(x[37],y[37],x[41],y[41]) + distance(x[62],y[62],x[66],y[66]) + distance(x[63],y[63],x[65],y[65])) / (2 * distance(x[36],y[36],x[45],y[45]))
        #AU7 = lerp(AU7,0.05,0.5)
        AU7_right = ((distance(x[37],y[37],x[41],y[41]) + distance(x[38],y[38],x[40],y[40])) / (2 * distance(x[36],y[36],x[39],y[39])))
        AU7_left = ((distance(x[43],y[43],x[47],y[47]) + distance(x[44],y[44],x[46],y[46])) / (2 * distance(x[42],y[45],x[45],y[45])))
        AU7 = (AU7_right + AU7_left / 2)        
        AU7 = lerp(AU7,0.05,0.8)		

        # ===============================================================================
        # AU9: Max = 2.5 & Min = 1
        #       
        AU9 = (math.atan2(y[50] - y[33], x[33] - x[50]) + math.atan2(y[52] - y[33],x[52] - x[33]))
        AU9 = lerp(AU9,1,2.5)
        
        # ===============================================================================
        # AU15: Max = 1 & Min = -0.5 (-0.8)
        #          
        AU15 = (math.atan2(y[48] - y[60], x[60] - x[48]) + math.atan2(y[54] - y[64],x[54] - x[64]))
        AU15 = lerp(AU15,-0.5,1)

        # ===============================================================================
        # AU20: Max = 3 & Min = 0
        #         
        AU20 = (math.atan2(y[59] - y[65], x[65] - x[59]) + math.atan2(y[55] - y[67],x[55] - x[67]) + math.atan2(y[59] - y[66],x[66] - x[59]) + math.atan2(y[59] - y[67],x[67] - x[59]) + math.atan2(y[55] - y[65],x[55] - x[65]))
        AU20 = lerp(AU20,0,3)

        # ===============================================================================
        # AU23: Max = 9 & Min = 2
        #          
        AU23 = (math.atan2(y[49] - y[50], x[50] - x[49]) + math.atan2(y[53] - y[52],x[53] - x[52]) + math.atan2(y[61] - y[49],x[61] - x[49]) + math.atan2(y[63] - y[53],x[53] - x[63]) + math.atan2(y[58] - y[59],x[58] - x[59]) + math.atan2(y[56] - y[55],x[55] - x[56]) + math.atan2(y[60] - y[51],x[51] - x[60]) + math.atan2(y[64] - y[51],x[64] - x[51]) + math.atan2(y[57] - y[60],x[57] - x[60]) + math.atan2(y[57] - y[64],x[64] - x[57]) + math.atan2(y[62] - y[49],x[62] - x[49]) + math.atan2(y[62] - y[53],x[53] - x[62]) + math.atan2(y[57] - y[60],x[57] - x[60]) + math.atan2(y[57] - y[64],x[64] - x[57]))
        AU23 = lerp(AU23,2,9)
        
        # ===============================================================================
        # AU26: Max = 1 & Min = 0
        #                 
        AU26 = (distance(x[61],y[61],x[67],y[67]) + distance(x[62],y[62],x[66],y[66]) + distance(x[63],y[63],x[65],y[65])) / (distance(x[36],y[36],x[45],y[45]))
        AU26 = lerp(AU26,0,1)        
        if AU26 > 0.1:
            AU25 = 1
        
        break

    result = np.array([AU1,AU2,AU4,AU6,AU7,AU9,AU15,AU20,AU23,AU25,AU26])   
    return result

def load_data():
    global x_full, y_full
    x, y = [], []
    global file_count, file_correct  
    sub_count = 0
    sub_correct = 0
    index = 0    
    
    # Changed order from dataset>>emotion TO emotion>>dataset
    for emotion in observed_emotions:
        index = observed_emotions.index(emotion)    
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
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                clahe_image = clahe.apply(gray)

                detections = detector(clahe_image, 1)                    
                if len(detections) > 0:
                    feature = extract_feature(clahe_image)
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
                    if pred == index:
                        file_lin = 1                   
                    
                    
                    emotion_correct[index] += file_lin
                    emotion_count[index] += 1                   
                    file_correct += file_lin;
                    sub_correct += file_lin;
                    
            # Calculate Percentage correct for dataset emotion
            dataset_correct[d_index][index] = 0;   
            dataset_count[d_index][index] = 0;
            dataset_accuracies[d_index][index] = 0;         
            if sub_count > 0:
                dataset_correct[d_index][index] = sub_correct;   
                dataset_count[d_index][index] = sub_count;     
                dataset_accuracies[d_index][index] = ((sub_correct / sub_count) * 100)
            print(d_index,"/",index," Dataset=",dataset_name[d_index], "Correct=",dataset_accuracies[d_index][index],"  (",dataset_count[d_index][index],")")          
              
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
    