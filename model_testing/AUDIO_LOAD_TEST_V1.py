from moviepy.editor import *
from pydub import AudioSegment
from scipy.io import wavfile
import noisereduce as nr
import numpy as np
import os, glob, pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import librosa
import soundfile

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

np.random.seed(4)
file_count = 0
file_correct = 0
emotion_correct = [0, 0, 0, 0, 0, 0, 0, 0]
emotion_count = [0, 0, 0, 0, 0, 0, 0, 0]

# ==================================================================================
# conda activate py37
# cd d:\Documents\PhD\PythonDevelopment\EMPACTVR ONNX November2023\model_testing
# python AUDIO_LOAD_TEST_V1.py
# ==================================================================================

# =========================================================
# Emotions to observe
observed_emotions=['neutral', 'happy', 'sad', 'anger', 'fear', 'disgust', 'surprise']
num_emotions = 7
x_check, y_check = [], []

#Datasets

dataset_name=['Oreau']
dataset_directory=['f:\Documents\EmotionDatabases\AudioLibrariesForUse\Oreau\Oreau_Labelled\\%s\\*']

"""
dataset_name=['Shemo', 'SAVEE', 'EmoDB', 'RAVDESS', 'CREMA-D-HI', 'TESS', 'AESDD', 'Oreau', 'CaFE', 'Emovo', 'JLCorpus', 'SUBESCO', 'ESD-CN', 'MESD', 'URDU'] 
dataset_directory=[
'f:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\Shemo\\combined_labelled\\%s\\*', 
'f:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\SAVEE\\SAVEE_Audio_Testing\\%s\\*', 
'f:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\EmoDB\\EMODB_labelled\\%s\\*', 
'f:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\RAVDESS\\RAVDESS_labelled_audio\\%s\\*', 
'f:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\CREMA-D\\CREMAD_Labelled_44K\\%s\\*Hi*', 
'f:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\Toronto Emotional Speech Set (TESS)\\dataverse_files\\%s\\*','f:\Documents\EmotionDatabases\AudioLibrariesForUse\\AESDD\\AESDD_Labelled\\%s\\*',
'f:\Documents\EmotionDatabases\AudioLibrariesForUse\Oreau\Oreau_Labelled\\%s\\*',
'f:\Documents\EmotionDatabases\AudioLibrariesForUse\CaFE\CaFE_Labelled\\%s\\*',
'f:\Documents\EmotionDatabases\AudioLibrariesForUse\Emovo\Emovo_Labelled\\%s\\*',
'f:\Documents\EmotionDatabases\AudioLibrariesForUse\JLCorpus_archive\JLCorpus_Labelled\\%s\\*',
'f:\Documents\EmotionDatabases\AudioLibrariesForUse\SUBESCO\SUBESCO_Labelled\\%s\\*',
'f:\Documents\EmotionDatabases\AudioLibrariesForUse\\ESD\\ESD_CN_Labelled\\%s\\*',
'f:\Documents\EmotionDatabases\AudioLibrariesForUse\MESD\MESD_Labelled\\%s\\*',
'f:\Documents\EmotionDatabases\AudioLibrariesForUse\\URDU\\URDU_Labelled\\%s\\*']
"""

dataset_accuracies = [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]
dataset_correct = [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]
dataset_count = [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]


print("Load model")
MODEL_PATH = '..\models\AudioClassifier.onnx'
path = MODEL_PATH
model = onnx.load(path)
onnx.checker.check_model(model)
print('ONNX model loaded')

def extract_feature(file_name, mfcc, chroma, mel):

    if file_name.endswith('.mp4') or file_name.endswith('.m4v') or file_name.endswith('.avi'):
       
        wav_file = "TEMP.wav"
        clip = AudioFileClip(file_name)
        clip.write_audiofile(wav_file, fps=44100, nbytes=4, buffersize=50000, codec='pcm_f32le', ffmpeg_params=["-ac", "1"], verbose=False, logger=None)

        y, sample_rate = librosa.load(wav_file, sr=44100)
        X = librosa.to_mono(y)
    else:
        y, sample_rate = librosa.load(file_name, sr=44100)
        X = librosa.to_mono(y)
   
        
    if chroma:
        stft = np.abs(librosa.stft(X))
        result = np.array([])
    if mfcc:        
        full_mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=34)
        subset_mfccs = full_mfccs[1:]       
        norm_mfccs = np.subtract(subset_mfccs,np.mean(subset_mfccs))
        my_mfccs = np.mean(norm_mfccs.T, axis=0)            
        result = np.hstack((result, my_mfccs))      
    
    #if chroma:
    #    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis = 0)
    #    result = np.hstack((result, chroma))
    #if mel:
    #    #mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    #    mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate, n_mels=40).T, axis=0)
    #    result = np.hstack((result, mel))

    return result

def load_data(test_size = 0.2):
    global x_check, y_check
    global file_count, file_correct
    global emotion_correct, emotion_count   
    x, y = [], []
    sub_count = 0
    sub_correct = 0
    
    for emotion in observed_emotions: 
        index = observed_emotions.index(emotion)    
        print("==============================")    
        print("Emotion=",emotion)   
        for d_index,d_item in enumerate(dataset_directory): 
            sub_count = 0
            sub_correct = 0     
            print("Dataset=",dataset_name[d_index]) 
            for file in glob.glob(d_item %emotion):    
                #print("File=",file)
                audio_data = []
                audio_label = []
                feature = extract_feature(file, mfcc=True, chroma=True, mel=True)  
                #print("shape=",feature.shape)
                x.append(feature)
                y.append(emotion) 
                audio_data.append(feature)  
                audio_label.append(emotion) 
                if len(audio_data) > 0:
                    file_pred = np.array(audio_data)
                    file_pred = file_pred.reshape(1, -1)
                    
                    
                    #file_lin = loaded_model.score(file_pred, audio_label)
                    #print("Label=%s:", audio_label, " %s:" %file , " %s" %file_lin)
                    #=========================================================================
                    # Batch Prediction
                    audio_data = [arr.tolist() for arr in audio_data]
                    #batch_pred, proba = predict_batch(face_data, ort_session)
                    batch_pred, proba = predict_batch_alt(audio_data, ort_session)
                    #=========================================================================
                    #print(batch_pred,"-", emotion)  
                    result=0
                    if observed_emotions[batch_pred] == emotion:
                        result = 1
                        #print("match")
                    
                    
                    index = observed_emotions.index(emotion)                    
                    
                    emotion_correct[index] += result
                    emotion_count[index] += 1                   
                    file_correct += result;
                    file_count += 1
                    sub_count += 1                  
                    sub_correct += result;                    
 

            # Calculate Percentage correct for dataset emotion
            dataset_correct[d_index][index] = 0;   
            dataset_count[d_index][index] = 0;
            dataset_accuracies[d_index][index] = 0;         
            if sub_count > 0:
                dataset_correct[d_index][index] = sub_correct;   
                dataset_count[d_index][index] = sub_count;     
                dataset_accuracies[d_index][index] = ((sub_correct / sub_count) * 100)
            print(d_index,"/",index," Dataset=",dataset_name[d_index], "Correct=",dataset_accuracies[d_index][index],"  (",dataset_count[d_index][index],")") 

 
    x_check = np.array(x)
    y_check = y
    return

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
    
    #print(prediction_count)
    #print("Prediction=",prediction)
    
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
load_data(test_size=0.2)
              
#full_pred= loaded_model.predict(x_check)                      
#accuracy = accuracy_score(y_true=y_check, y_pred=full_pred)
#print("=============================")
#print(accuracy)
#print("=============================")
#print("Single testing")
#print(file_count)
#print(file_correct)
#print((file_correct / file_count) * 100)
#print("=============================")

for emotion in observed_emotions:
    index = observed_emotions.index(emotion)
    #Total_correct += emotion_correct[index]
    #Total_incorrect += emotion_count[index] - Total_correct
    emotion_accuracy = 0
    if emotion_correct[index] > 0:
        emotion_accuracy = ((emotion_correct[index] / emotion_count[index]) * 100)
    print("emotion =%s" %emotion)
    print("Correct = %i" %emotion_accuracy)
    #print("Incorrect = %i" %emotion_incorrect[index])


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
        cycle_correct += dataset_correct[d_index][e_index]
        cycle_count += dataset_count[d_index][e_index]
    cycle_percentage = ((cycle_correct / cycle_count) * 100)
    print("Dataset=",dataset_name[d_index],"  (",cycle_percentage,"%) (files=",cycle_count,")")    
    
    
    
    
    