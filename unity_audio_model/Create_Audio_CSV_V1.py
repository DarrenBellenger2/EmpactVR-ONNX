import numpy as np
import os, glob, math, random, pickle
import cv2
import librosa
import soundfile
import csv

# conda activate py37
# d:
# cd d:\Documents\PhD\PythonDevelopment\EMPACTVR ONNX November2023\unity_audio_model
# python Create_Audio_CSV_V1.py

# Emotions to observe
observed_emotions=['neutral', 'happy', 'sad', 'anger', 'fear', 'disgust', 'surprise']
x_full, y_full = [], []
csv_header = ['Emotion', '1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85']
#csv_header = ['Emotion', 'Input']
#csv_data_row = [0,0,0,0,0,0,0,0,0,0,0,0]

#Datasets
#dataset_name=['SAVEE', 'RAVDESS', 'Oreau', 'CaFE'] 
#dataset_directory=['f:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\SAVEE\\SAVEE_Audio_Testing\\%s\\*', 
#'f:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\RAVDESS\\RAVDESS_labelled_audio\\%s\\*', 
#'f:\Documents\EmotionDatabases\AudioLibrariesForUse\Oreau\Oreau_Labelled\\%s\\*',
#'f:\Documents\EmotionDatabases\AudioLibrariesForUse\CaFE\CaFE_Labelled\\%s\\*']


#Datasets
dataset_name=['Shemo', 'SAVEE', 'EmoDB', 'RAVDESS', 'CREMA-D-HI', 'CREMA-D-XX', 'TESS', 'AESDD', 'URDU', 'Oreau', 'CaFE', 'Emovo', 'JLCorpus', 'SUBESCO', 'MESD', 'ESD-CN'] # 'ESD-EN'] 

dataset_directory=['f:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\Shemo\\combined_labelled\\%s\\*', 
'f:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\SAVEE\\SAVEE_Audio_Testing\\%s\\*', 
'f:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\EmoDB\\EMODB_labelled\\%s\\*', 
'f:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\RAVDESS\\RAVDESS_labelled_audio\\%s\\*', 
'f:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\CREMA-D\\CREMAD_Labelled_44K\\%s\\*Hi*', 
'f:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\CREMA-D\\CREMAD_Labelled_44K\\%s\\*Xx*',
'f:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\Toronto Emotional Speech Set (TESS)\\dataverse_files\\%s\\*',
'f:\Documents\EmotionDatabases\AudioLibrariesForUse\\AESDD\\AESDD_Labelled\\%s\\*',
'f:\Documents\EmotionDatabases\AudioLibrariesForUse\\URDU\\URDU_Labelled\\%s\\*',
'f:\Documents\EmotionDatabases\AudioLibrariesForUse\Oreau\Oreau_Labelled\\%s\\*',
'f:\Documents\EmotionDatabases\AudioLibrariesForUse\CaFE\CaFE_Labelled\\%s\\*',
'f:\Documents\EmotionDatabases\AudioLibrariesForUse\Emovo\Emovo_Labelled\\%s\\*',
'f:\Documents\EmotionDatabases\AudioLibrariesForUse\JLCorpus_archive\JLCorpus_Labelled\\%s\\*',
'f:\Documents\EmotionDatabases\AudioLibrariesForUse\SUBESCO\SUBESCO_Labelled\\%s\\*',
'f:\Documents\EmotionDatabases\AudioLibrariesForUse\MESD\MESD_Labelled\\%s\\*',
'f:\Documents\EmotionDatabases\AudioLibrariesForUse\\ESD\\ESD_CN_Labelled\\%s\\*']

#'f:\Documents\EmotionDatabases\AudioLibrariesForUse\\ESD\\ESD_EN_Labelled\\%s\\*']


#dataset_name=['RAVDESS']
#dataset_directory=['f:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\RAVDESS\\RAVDESS_labelled_audio\\%s\\*']


rows, cols = (15, 6)
dataset_accuracies = [[0]*cols]*rows

def extract_feature(file_name, mfcc, chroma, mel):
    
    n_fft = 1024
    win_length = None
    hop_length = 512
    n_mels = 40
    n_mfcc = 34

    y, sample_rate = librosa.load(file_name, sr=44100)
    X = librosa.to_mono(y)
    
    
    #with soundfile.SoundFile(file_name) as mySoundFile:
    #    X = mySoundFile.read(dtype="float32")
    #    sample_rate = mySoundFile.samplerate
        
    if chroma:
        stft = np.abs(librosa.stft(X))
        result = np.array([])
    if mfcc:
        # ======================================================================
        # Overall normalisation
        
        #full_mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=34) 
        
        #melspec = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_fft=1024,win_length=win_length, hop_length=hop_length,n_mels=n_mels, htk=True, center=False, norm=None)
        #full_mfccs = librosa.feature.mfcc(S=librosa.core.spectrum.power_to_db(melspec),n_mfcc=n_mfcc, dct_type=2, norm='ortho')

        full_mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=n_mfcc, dct_type=2, norm='ortho', window='hamming', htk=False, n_mels=40, fmin=100, fmax=8000, n_fft=1024, hop_length=int(0.010*sample_rate), center=False)
        
        subset_mfccs = full_mfccs[1:]        
        norm_mfccs = np.subtract(subset_mfccs,np.mean(subset_mfccs))
        my_mfccs = np.mean(norm_mfccs.T, axis=0)            

        temp = np.mean(subset_mfccs)
        print("subset_mfccs shape:", subset_mfccs.shape, "    mean value:",temp) 
        print("check subtraction:", subset_mfccs[0][0], " - ", norm_mfccs[0][0])

        result = np.hstack((result, my_mfccs))
    
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis = 0)
        #result = np.hstack((result, chroma))
    if mel:
        #mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
        mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate, n_mels=40).T, axis=0)
        #result = np.hstack((result, mel))
    return result

def load_data(test_size = 0.2):
    global x_full, y_full
    x, y = [], []
    filecount = 0   

    # Changed order from dataset>>emotion TO emotion>>dataset
    with open('output.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        #writer.writerow(csv_header)
        for emotion in observed_emotions:   
            print("Emotion=",emotion)     
            for d_index,d_item in enumerate(dataset_directory):
                print("Dataset=",dataset_name[d_index])     
                filecount = 0       
                for file in glob.glob(d_item %emotion): 
                    print("File=",file)
                    filecount +=1   
                    
                    feature = extract_feature(file, mfcc=True, chroma=True, mel=True)

                    x.append(feature)
                    y.append(emotion)
                        
                    csv_data_row = feature
                    csv_data_row = np.insert(csv_data_row, 0, observed_emotions.index(emotion), axis=0)

                    writer.writerow(csv_data_row)  
                print("Filecount=",filecount)               
       
            
    #split = train_test_split(np.array(x), y, test_size = test_size, random_state = 9)
    #split = train_test_split(np.array(x), y, test_size = 0.1, random_state = 9)
    #x_full = np.array(x)
    #y_full = y
    return

print("Load data")
load_data(test_size=0.2)
               
                      







