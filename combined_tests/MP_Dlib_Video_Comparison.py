import numpy as np
import os, glob, math, random, pickle
import matplotlib.pyplot as plt
import cv2
import csv
import dlib
from pathlib import Path
import argparse
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
np.random.seed(4)

from moviepy.editor import *
from pydub import AudioSegment


# conda activate py312
# d:
# cd d:\Documents\PhD\PythonDevelopment\EMPACTVR ONNX November2023\combined_tests
# python MP_Dlib_Video_Comparison.py

# Emotions to observe
observed_emotions=['neutral', 'happy', 'sad', 'anger', 'fear', 'disgust', 'surprise']
x_full, y_full = [], []
csv_header = ['Emotion', 'MP_AU1','MP_AU2','MP_AU4','MP_AU6','MP_AU7','MP_AU9','MP_AU15','MP_AU20','MP_AU23','MP_AU26', 'Dlib_AU1','Dlib_AU2','Dlib_AU4','Dlib_AU6','Dlib_AU7','Dlib_AU9','Dlib_AU15','Dlib_AU20','Dlib_AU23','Dlib_AU26']
#csv_data_row = [0,0,0,0,0,0,0,0,0,0,0,0]

#Datasets
video_name=['Anger_Video']
#video_url=['E:\\Documents\\EmotionDatabases\\Video Libraries\\RAVDESS_640_Labelled\\fear\\01-01-06-02-02-01-09_1.mp4']

video_url=['E:\\Documents\\EmotionDatabases\\Video Libraries\\RAVDESS_640_Labelled\\disgust\\01-01-07-02-02-01-01_1.mp4']


#video_url=['E:\\Documents\\EmotionDatabases\\Enlarged Video Libraries\\RAVDESS_640_Labelled\\anger\\Video_Man01.mp4']
#video_url=['E:\\Documents\\EmotionDatabases\\Video Libraries\\ADFES_Labelled_MP4_Short\\anger\\F01-Anger-Face Forward-1.mp4']
emotion_attribute=['anger']
frame_cut = 0 #0.5

rows, cols = (6, 6)
dataset_accuracies = [[0]*cols]*rows


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


#Set up some required objects
video_capture = cv2.VideoCapture(0) #Webcam object
detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor("..\\facial_model_training\shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever 

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

    
def extract_feature(landmarks, image):
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
    
    dlib_feature = Dlib_extract_facial_feature(image)
    
    landmark_image = image
    x = []
    y = []
    for i in range(0,68): #Store X and Y coordinates in two lists
        #print(landmarks[i][0])
        x.append(float(landmarks[i][0]))
        y.append(float(landmarks[i][1]))   
        cv2.circle(landmark_image, (int(landmarks[i][0]), int(landmarks[i][1])), 2, (0, 255, 0), 1)     

    #print("MP Co-ords:" + " (0)=" + str(x[0]) + "/" + str(y[0]) + " (16)=" + str(x[16]) + "/" + str(y[16]) + " (8)=" + str(x[8]) + "/" + str(y[8])) 

        
    # ===============================================================================
    # Evaluate Action Units

    # ===============================================================================
    # AU1: Max = 1.2 & Min = 0.1
    #
    AU1 = (math.atan2(y[17] - y[20], x[20] - x[17]) + math.atan2(y[26] - y[23],x[26] - x[23]))
    AU1 = round(AU1, 4)
    #AU1 = lerp(AU1,0,1)
    
    # ===============================================================================
    # AU2: Max = 2.3 & Min = 0.9
    #
    AU2 = (math.atan2(y[17] - y[18], x[18] - x[17]) + math.atan2(y[26] - y[25],x[26] - x[25]))
    AU2 = round(AU2, 4) 
    #AU2 = lerp(AU2,0.5,2.3)
    
    # ===============================================================================
    # AU4: Max = 3 & Min = 1.2
    #
    AU4 = (math.atan2(y[39] - y[21], x[21] - x[39]) + math.atan2(y[42] - y[22],x[42] - x[22]))
    AU4 = round(AU4, 4)
    #AU4 = lerp(AU4,1.5,3)
    
    # ===============================================================================
    # AU6: Max = 1.16 & Min = 0.7
    #
    AU6 = (distance(x[48],y[48],x[66],y[66]) + distance(x[66],y[66],x[54],y[54])) / (distance(x[48],y[48],x[51],y[51]) + distance(x[51],y[51],x[54],y[54]))
    AU6 = round(AU6, 4)
    #AU6 = lerp(AU6,0.7,1.3)
    
    # ===============================================================================
    # AU7: Max = 0.5 & Min = 0.05
    #       
    #AU7 = (distance(x[37],y[37],x[41],y[41]) + distance(x[62],y[62],x[66],y[66]) + distance(x[63],y[63],x[65],y[65])) / (2 * distance(x[36],y[36],x[45],y[45]))
    AU7_right = ((distance(x[37],y[37],x[41],y[41]) + distance(x[38],y[38],x[40],y[40])) / (2 * distance(x[36],y[36],x[39],y[39])))
    AU7_left = ((distance(x[43],y[43],x[47],y[47]) + distance(x[44],y[44],x[46],y[46])) / (2 * distance(x[42],y[45],x[45],y[45])))
    AU7 = (AU7_right + AU7_left / 2)    
    
    AU7 = round(AU7, 4)
    #AU7 = lerp(AU7,0,0.48)
    #AU7 = lerp(AU7,0.05,0.8)

    # ===============================================================================
    # AU9: Max = 2.5 & Min = 1
    #       
    #AU9 = (math.atan2(y[50] - y[33], x[33] - x[50]) + math.atan2(y[52] - y[33],x[52] - x[33]))
    AU9 = (math.atan2(y[32] - y[33], x[33] - x[32]) + math.atan2(y[34] - y[33],x[34] - x[33]))
    AU9 = round(AU9, 4)
    #AU9 = lerp(AU9,1.2,2.5)
    
    # ===============================================================================
    # AU15: Max = 1 & Min = -0.5 (-0.8)
    #          
    #AU15 = (pos_atan2(y[48] - y[60], x[60] - x[48]) + pos_atan2(y[54] - y[64],x[54] - x[64]))
    #AU15 = round(AU15, 4)
    
    AU15 = (math.atan2(y[48] - y[60], x[60] - x[48]) + math.atan2(y[54] - y[64],x[54] - x[64])) 
    AU15 = round(AU15, 4)
    
    print("MP AU15:" + " (0)=" + str(AU15))
    #AU15 = lerp(AU15,-1,1)

    # ===============================================================================
    # AU20: Max = 3 & Min = 0
    #         
    AU20 = (pos_atan2(y[59] - y[65], x[65] - x[59]) + pos_atan2(y[55] - y[67],x[55] - x[67]) + pos_atan2(y[59] - y[66],x[66] - x[59]) + pos_atan2(y[59] - y[67],x[67] - x[59]) + pos_atan2(y[55] - y[65],x[55] - x[65]))
    AU20 = round(AU20, 4)
    #AU20 = lerp(AU20,0.7,3.5)

    # ===============================================================================
    # AU23: Max = 9 & Min = 2
    #          
    AU23 = (pos_atan2(y[49] - y[50], x[50] - x[49]) + pos_atan2(y[53] - y[52],x[53] - x[52]) + pos_atan2(y[61] - y[49],x[61] - x[49]) + pos_atan2(y[63] - y[53],x[53] - x[63]) + pos_atan2(y[58] - y[59],x[58] - x[59]) + pos_atan2(y[56] - y[55],x[55] - x[56]) + pos_atan2(y[60] - y[51],x[51] - x[60]) + pos_atan2(y[64] - y[51],x[64] - x[51]) + pos_atan2(y[57] - y[60],x[57] - x[60]) + pos_atan2(y[57] - y[64],x[64] - x[57]) + pos_atan2(y[62] - y[49],x[62] - x[49]) + pos_atan2(y[62] - y[53],x[53] - x[62]) + pos_atan2(y[57] - y[60],x[57] - x[60]) + pos_atan2(y[57] - y[64],x[64] - x[57]))
    AU23 = round(AU23, 4)
    #AU23 = lerp(AU23,3,11)
    
    # ===============================================================================
    # AU26: Max = 1 & Min = 0
    #                 
    AU26 = (distance(x[61],y[61],x[67],y[67]) + distance(x[62],y[62],x[66],y[66]) + distance(x[63],y[63],x[65],y[65])) / (distance(x[36],y[36],x[45],y[45]))
    AU26 = round(AU26, 4)
    #AU26 = lerp(AU26,0,1.4)        
    #if AU26 > 0.1:
    #   AU25 = 1
      
    
    result = np.array([AU1,AU2,AU4,AU6,AU7,AU9,AU15,AU20,AU23, AU26, dlib_feature[0],dlib_feature[1],dlib_feature[2],dlib_feature[3],dlib_feature[4],dlib_feature[5],dlib_feature[6],dlib_feature[7],dlib_feature[8],dlib_feature[9]])   
    return result, landmark_image

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
        AU7 = (distance(x[37],y[37],x[41],y[41]) + distance(x[62],y[62],x[66],y[66]) + distance(x[63],y[63],x[65],y[65])) / (2 * distance(x[36],y[36],x[45],y[45]))
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

    result = np.array([AU1,AU2,AU4,AU6,AU7,AU9,AU15,AU20,AU23,AU26])   
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
    
def load_data(face_mesh):
    global x_full, y_full
    x, y = [], []
    filecount = 0  
    output_video_path = 'mp_output.avi' 
    output_video_size = 640,480

    # Changed order from dataset>>emotion TO emotion>>dataset
    with open('MP_Dlib_video_comparison.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        for d_index,d_item in enumerate(video_url):
            print("Video=",video_name[d_index])     
            filecount = 0       
            for file in glob.glob(d_item): 
                print("File=",file)
                filecount +=1   
                
                # ================================================================================================
                # Cycle through video frames

                clip = VideoFileClip(file) 
                subclip = clip #clip.subclip(frame_cut,(frame_cut * -1))
                subclip.write_videofile("temp.mp4") 
                
                landmarks_video = cv2.VideoWriter(output_video_path,cv2.VideoWriter_fourcc(*'DIVX'), 30, output_video_size)  
                
                cap = cv2.VideoCapture("temp.mp4")                
                while(cap.isOpened()):
                    ret, image = cap.read()
                    if ret == False:
                        break 
                
                    #image = cv2.imread(file) 
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
                                x_val = float(face_landmarks.landmark[index].x * w)
                                y_val = float(face_landmarks.landmark[index].y * h)
                                landmarks_extracted.append((x_val, y_val))
                    
                            feature, output_image = extract_feature(landmarks_extracted, image)                        
                            writer.writerow([emotion_attribute[d_index], feature[0], feature[1], feature[2], feature[3], feature[4], feature[5], feature[6], feature[7], feature[8], feature[9], feature[10], feature[11], feature[12], feature[13], feature[14], feature[15], feature[16], feature[17], feature[18], feature[19]])
                            landmarks_video.write(output_image)

                            
                landmarks_video.close()        

            print("Filecount=",filecount)               
       
            
    return

    
def main():
    
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
    load_data(face_mesh)

if __name__ == '__main__':
    main()




