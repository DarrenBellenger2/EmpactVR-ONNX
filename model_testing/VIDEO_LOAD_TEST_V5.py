import datetime
import time
import librosa
import soundfile
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
# conda info --envs
# conda activate py37
# cd d:\Documents\PhD\PythonDevelopment\EMPACTVR ONNX November2023\model_testing
# python VIDEO_LOAD_TEST_V5.py
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

frame_cut = 0.5

#Datasets
dataset_name=['SAVEE_Labelled']
#'ADFES', 'Ryerson', 'eNTERFACE05', 'RAVDESS', 'SAVEE', 'CREMA-D']

#dataset_directory=['E:\\Documents\\EmotionDatabases\\Video Libraries\\QUICK_SAVEE\\%s\\*.avi']
#dataset_directory=['E:\\Documents\\EmotionDatabases\\Video Libraries\\QUICK_RAVDESS\\%s\\*.mp4']
#dataset_directory=['E:\\Documents\\EmotionDatabases\\Video Libraries\\QUICK\\%s\\*.mp4']


dataset_directory=['E:\\Documents\\EmotionDatabases\\Video Libraries\\SAVEE_Labelled\\%s\\*.avi']

#dataset_directory=['E:\\Documents\\EmotionDatabases\\Video Libraries\\RAVDESS_320_Labelled\\%s\\*.mp4']

#dataset_directory=['E:\\Documents\\EmotionDatabases\\Video Libraries\\ADFES_Labelled\\%s\\*.m4v']

#dataset_directory=['C:\\Users\\bellengerd\\Documents\\EmotionDatabases\\Video Libraries\\Ryerson\\Ryerson_EN_Labelled\\%s\\*']
#dataset_directory=['C:\\Users\\bellengerd\\Documents\\EmotionDatabases\\Video Libraries\\CREMA-D_Labelled\\%s\\*']
#dataset_directory=['E:\\Documents\\EmotionDatabases\\Video Libraries\\CREMA-D_Labelled\\%s\\*Hi*']

#'E:\\Documents\\EmotionDatabases\\Video Libraries\\ADFES_Labelled\\%s\\*'
#'E:\\Documents\\EmotionDatabases\\Video Libraries\\Ryerson\\Ryerson_EN_Labelled\\%s\\*',
#'E:\\Documents\\EmotionDatabases\\Video Libraries\\eNTERFACE05\\eNTERFACE05_Labelled\\%s\\*',
#'E:\\Documents\\EmotionDatabases\\Video Libraries\\RAVDESS_Labelled\\%s\\*',
#'E:\\Documents\\EmotionDatabases\\Video Libraries\\SAVEE_Labelled\\%s\\*',
#'E:\\Documents\\EmotionDatabases\\Video Libraries\\CREMA-D_Labelled\\%s\\*']

#dataset_name=['ADFES(V)']
#dataset_directory=['c:\\Users\\bellengerd\\Documents\\EmotionDatabases\\Video Libraries\\ADFES_Labelled\\%s\\*']

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


#Set up some required objects
#video_capture = cv2.VideoCapture(0) #Webcam object
detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever 

print("Load model")
#facial_model = pickle.load(open("..\models\Facial_1.mlp", 'rb'))
#audio_model = pickle.load(open("..\models\Audio_1.mlp", 'rb'))
#print("Audio Classes:",audio_model.classes_)
#print("Facial Classes:",facial_model.classes_)

MODEL_PATH = '..\models\ImageClassifier_12K.onnx'
path = MODEL_PATH
model = onnx.load(path)
onnx.checker.check_model(model)
print('ONNX model loaded')

def distance(From_X, From_Y, To_X, To_Y): 
    dist = math.sqrt( math.pow(To_X - From_X,2) + math.pow(To_Y - From_Y,2) )
    return dist

def lerp(x, a, b): 
    ret = (x - a) / (b - a)
    if ret < 0:
        ret = 0
    return round(ret,2)
    
def extract_facial_feature(image):
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
        AU7 = (distance(x[37],y[37],x[41],y[41]) + distance(x[62],y[62],x[66],y[66]) + distance(x[63],y[63],x[65],y[65])) / (2 * distance(x[36],y[36],x[45],y[45]))
        AU7 = lerp(AU7,0.05,0.5)

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
    
def process_image(file, emotion):

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
    
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_image = clahe.apply(gray)

        detections = detector(clahe_image, 1)   
        if len(detections) > 0:
            feature = extract_facial_feature(clahe_image)
            face_data.append(feature)
            face_label.append(emotion)
    
    #=========================================================================
    # Batch Prediction
    face_data = [arr.tolist() for arr in face_data]
    #batch_pred, proba = predict_batch(face_data, ort_session)
    batch_pred, proba = predict_batch_alt(face_data, ort_session)
    
    #=========================================================================
    # Mean Prediction
    mean_array = np.mean(face_data, axis=0)
    mean_array = mean_array.reshape(1, -1) 
    temp = mean_array[0].tolist()  
    #print("MEAN:",temp)
    mean_pred = predict(temp, ort_session)

    #=========================================================================
    # Median Prediction
    median_array = np.median(face_data, axis=0)
    median_array = mean_array.reshape(1, -1) 
    temp = median_array[0].tolist()  
    #print("MEAN:",temp)
    median_pred = predict(temp, ort_session)
  
    print(batch_pred,"-", mean_pred, " : ", median_pred)  

    result=0
    #if observed_emotions[batch_pred] == emotion:
    #if observed_emotions[batch_pred] == emotion or observed_emotions[mean_pred] == emotion:
    #if observed_emotions[batch_pred] == emotion:
    #if observed_emotions[median_pred] == emotion:
    if observed_emotions[batch_pred] == emotion:
        result = 1
        print("match")
    
    #=========================================================================
    
    probability_rtn = 0
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
        
        y_pred= audio_model.predict(audio_data)                     
        #result = accuracy_score(y_true=audio_label, y_pred=y_pred) 
        result = audio_model.score(audio_data,audio_label)
        
        proba = audio_model.predict_proba(audio_data)
        for i in range(len(proba)):
            for j in range(len(proba[i])):      
                proba[i][j] = round(proba[i][j] * 100)       
        #print("Label:",audio_label,"AUDIO: ", proba)        
        
    return result,proba
    
def load_data():
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
                facial_accuracy,facial_probability = process_image(file, emotion)               
                if facial_accuracy > 0:
                    dataset_facial_correct[d_index][index] += 1;   
                
                #elif facial_accuracy >= 0.1:
                #    arr = np.array(facial_probability)
                #    if index == arr.argmax():
                #        dataset_facial_correct[d_index][index] += 1;   
                        
                        
                # ==========================================================================
                # Perform audio prediction

                #now = datetime.datetime.now()
                #print ("AFTER FACE: " + now.strftime("%Y-%m-%d %H:%M:%S"))

                audio_accuracy = 0
                #audio_accuracy,audio_probability = process_audio(file, emotion)               
                if audio_accuracy >= 0.5:
                    dataset_audio_correct[d_index][index] += 1; 

                #now = datetime.datetime.now()
                #print ("AFTER AUDIO: " + now.strftime("%Y-%m-%d %H:%M:%S"))
                    

               
                # ==========================================================================
                # Perform combined prediction
                
                """
                highest_prediction = 0
                highest_index = -1
                for i in range(len(facial_probability[0])):
                    temp = round(facial_probability[0][i]) + round(audio_probability[0][i])
                    if temp > highest_prediction:
                        highest_prediction = temp
                        highest_index = i
                if highest_index == index:
                    dataset_combined_correct[d_index][index] += 1;   
                """
                   
                #print("File=",file, " Face=", facial_accuracy, " Audio=", audio_accuracy)                   
                #print("INDEX: ", index, " HIGH:", highest_index, "HIGH PRED:", highest_prediction, "standard accuracy=", facial_accuracy, "File=",file,)   
                #print("Face: ", facial_probability)                    
                #print("AUDIO: ", audio_probability)  
                
                
                    
            # Calculate Percentage correct for dataset emotion
            dataset_facial_count[d_index][index] = sub_count; 
            dataset_audio_count[d_index][index] = sub_count; 
            dataset_combined_count[d_index][index] = sub_count;   
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
        int_value = int(abs(value) + 0.5)
        if int_value > 6:
            int_value = 6
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
        pred = predict(rows[i], ort_session)
        values.append(pred) 
        prediction_count[pred] += 1

    highest_index = 0
    for i in range(num_emotions):
        if prediction_count[i] > prediction_count[highest_index]:
            highest_index = i
    prediction = highest_index
    
    print(prediction_count)
    print("Prediction=",prediction)
    
    return prediction, prediction_count 

    
# make a class prediction for one row of data
def predict(row, ort_session):

    data = json.dumps({'data': row})
    data = np.array(json.loads(data)['data']).astype('float32')
    temp = np.expand_dims(data, axis=0)
    #print(temp.shape)

    ort_inputs = {input_name: temp}
    ort_outs = ort_session.run([output_name], ort_inputs)

    prediction = int(ort_outs[0] + 0.5)
    if prediction >= num_emotions:
        prediction = (num_emotions - 1) 
    #prediction = ort_outs[0]

    #print(prediction, " - ", ort_outs[0])
    
    return prediction

    

    
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

print("=============================================================================================")
print("= Audio")
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
        cycle_correct += dataset_audio_correct[d_index][index]
        cycle_count += dataset_audio_count[d_index][index]
    if cycle_correct != 0 and cycle_count !=0:
        cycle_percentage = ((cycle_correct / cycle_count) * 100)
    overall_correct+=cycle_correct
    overall_count+=cycle_count
    print("Emotion=",emotion,"  (",cycle_percentage,"%) (files=",cycle_count,")")  
if overall_correct != 0 and overall_count !=0:
    overall_percentage = ((overall_correct / overall_count) * 100)    
print("Overall: (",overall_percentage,"%) (files=",overall_count,")")  

print("=============================================================================================")
print("= Combined")
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
        cycle_correct += dataset_combined_correct[d_index][index]
        cycle_count += dataset_combined_count[d_index][index]
    if cycle_correct != 0 and cycle_count !=0:
        cycle_percentage = ((cycle_correct / cycle_count) * 100)
    overall_correct+=cycle_correct
    overall_count+=cycle_count
    print("Emotion=",emotion,"  (",cycle_percentage,"%) (files=",cycle_count,")")  
if overall_correct != 0 and overall_count !=0:
    overall_percentage = ((overall_correct / overall_count) * 100)    
print("Overall: (",overall_percentage,"%) (files=",overall_count,")")  


for d_index,d_item in enumerate(dataset_directory):
    for emotion in observed_emotions: 
        index = observed_emotions.index(emotion)       
        cycle_correct=0
        cycle_count=0
    cycle_percentage = ((cycle_correct / cycle_count) * 100)
    print("Dataset=",dataset_name[d_index],"  (",cycle_percentage,"%) (files=",cycle_count,")")     





