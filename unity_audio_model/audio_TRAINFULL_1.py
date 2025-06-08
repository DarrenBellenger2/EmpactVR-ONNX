# pytorch mlp for binary classification
import sys
import numpy
import pandas as pd
from numpy import vstack
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
from torch.nn import CrossEntropyLoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F

# conda activate py37
# conda activate py312
# cd D:\Documents\PhD\PythonDevelopment\EMPACTVR ONNX November2023\unity_audio_model
# python audio_TRAINFULL_1.py

#======================================================
#TRAIN Accuracy: 0.925
#TEST Accuracy: 0.929
#======================================================

# dataset definition
MODEL_PATH = 'audio_model.pth'
CLASS_LABELS  = ['neutral', 'happy', 'sad', 'anger', 'fear', 'disgust', 'surprise']
NUM_CLASSES = len(CLASS_LABELS)
numpy.set_printoptions(threshold=sys.maxsize)

class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        df = read_csv(path, header=None)
        df = df.sample(frac = 1)
        # store the inputs and outputs
        #self.X = df.values[:, :-1]
        #self.y = df.values[:, -1]
        
        self.X = df.values[:,1:34]
        self.y = df.values[:,0]   
        print("X:",self.X.shape)
        
        # ensure input data is floats
        self.X = self.X.astype('float32')
        # label encode target and ensure the values are floats
        #self.y = LabelEncoder().fit_transform(self.y)
        #self.y = self.y.astype('float32')
        #self.y = self.y.reshape((len(self.y), 1))
        
        # ======================================================
        # Hot Encode
        self.y = pd.get_dummies(self.y).to_numpy()
        self.y = self.y.astype('float32')       

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.2):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])

class MLP(Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        #print("input_size=",input_size)
        self.fc1 = Linear(input_size, 100)
        self.act1 = ReLU()
        self.dropout1 = Dropout(0.02)
        self.fc2 = Linear(100, 200)
        self.act2 = ReLU()
        self.dropout2 = Dropout(0.02)        
        self.fc3 = Linear(200, 10)
        self.act3 = ReLU() 
        #self.dropout3 = Dropout(0.2)        
        self.fc4 = Linear(10, NUM_CLASSES)
        
    def forward(self, x):
        #==============================================
        #out = self.fc2(F.relu(self.fc1(x)))
        #==============================================
        #x = self.fc1(x)
        #x = self.act1(x)
        #x = self.fc2(x) 
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x) 
        x = self.act2(x)
        x = self.dropout2(x)        
        x = self.fc3(x) 
        x = self.act3(x)
        #x = self.dropout3(x)        
        x = self.fc4(x)  
        x = F.softmax(x, dim=1)  
        return x       
        

#Function to Convert to ONNX 
def Convert_ONNX(): 

    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor
    input_size = 33 
    batch_size=128
    dummy_input = torch.randn(1, input_size, requires_grad=True)  

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "AudioClassifier_V2.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX')

def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)
    
def resume(model, filename):
    model.load_state_dict(torch.load(filename)) 
        
# prepare the dataset
def prepare_data(path):
    # load the dataset
    dataset = CSVDataset(path)
    # calculate split
    train, test = dataset.get_splits()
    #print("train=",train.shape)
    
    # prepare data loaders
    #train_dl = DataLoader(train, batch_size=128, shuffle=False)
    train_dl = DataLoader(dataset, batch_size=128, shuffle=False)
    test_dl = DataLoader(test, batch_size=128, shuffle=False)
    return train_dl, test_dl

# train the model
def train_model(train_dl, model):
    # define the optimization
    criterion = CrossEntropyLoss() #L1Loss() #MSELoss() 
    #criterion = MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay=0.0001) #lr = 0.001, weight_decay=0.0001
 
    n_epochs = 5000
    early_stop_thresh = 2000
    best_loss = -1
    best_epoch = -1
 
    # enumerate epochs
    for epoch in range(5000):
        # enumerate mini batches
        cum_loss = 0
        for i, (inputs, targets) in enumerate(train_dl):

            #print("inputs=",inputs.shape)
            # compute the model output
            yhat = model(inputs)
            #print(yhat.shape, "     -     ",targets.shape)
            # calculate loss
            loss = criterion(yhat, targets)

            # clear the gradients
            optimizer.zero_grad()

            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
            
            cum_loss += loss.item()

        if cum_loss < best_loss or best_loss == -1:
            best_loss = cum_loss
            best_epoch = epoch
            checkpoint(model, "best_model.pth")
        elif epoch - best_epoch > early_stop_thresh:
            print("Early stopped training at epoch %d" % epoch)
            break 
        
        print ('Epoch [%d/%d], Loss: %.4f' %(epoch+1, n_epochs, cum_loss))  			
			
        #if epoch % 10 == 0:
        #    print ('Epoch [%d/%d], Loss: %.4f' 
        #              %(epoch+1, 5000, cum_loss))
            
            
# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        #print("inputs=",inputs.shape)
        yhat = model(inputs)
        #print(yhat.shape, "     -     ",targets.shape)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), NUM_CLASSES))
        # round to class values
        yhat = yhat.round()
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc

# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat

# prepare the data
path = 'audio_output.csv' #'audio_output.csv'
train_dl, test_dl = prepare_data(path)
print(len(train_dl.dataset), len(test_dl.dataset))
#print("training shape=",train_dl.shape)
# define the network
model = MLP(33)
# train the model
train_model(train_dl, model)

# evaluate the model
acc = evaluate_model(train_dl, model)
print('TRAIN Accuracy: %.3f' % acc)

# evaluate the model
acc = evaluate_model(test_dl, model)
print('TEST Accuracy: %.3f' % acc)
# make a single prediction (expect class=1)
#row = [102.4898529,-38.81188965,52.5655899,-20.51604652,-7.544162273,-6.737011909,-24.28520012,11.7255621,-26.17083931,-5.696669102,4.078910828,-23.02308464,-4.219229221,-13.84540939,-2.476534128,-1.803974152,-21.5403347,-6.417791843,-3.749093771,-3.127588272,-8.999363899,-5.305860996,6.680866241,-0.504951894,8.638659477,7.102868557,4.604849339,8.565843582,-1.735134721,5.854863644,3.333282709,0.808232963,10.06078815] #0
#yhat = predict(row, model)
#print('Predicted: %.3f (class=%d)' % (yhat, yhat.round()))
#print("Predicted:", yhat)

#torch.save(model, MODEL_PATH)
torch.save(model.state_dict(), MODEL_PATH)
print("Model saved")

# Conversion to ONNX 
Convert_ONNX() 