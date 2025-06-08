# pytorch mlp for binary classification
import sys
import numpy
import pandas as pd
from numpy import vstack
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch
from torch import Tensor
from torch.nn import Linear
from torch.nn import BatchNorm1d
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
from torchvision import transforms
import matplotlib.pyplot as plt

# conda activate py37
# conda activate py312

# cd d:\Documents\PhD\PythonDevelopment\EMPACTVR ONNX November2023\mediapipe_facial_model_training
# python COMBINED_MLP_TRAINFULL_WITH_AU25.py

#======================================================

#Best epoch 4935
#TRAIN Accuracy: 0.976
#TEST Accuracy: 0.898
#Model saved

#Best epoch 4999
#TRAIN Accuracy: 0.973
#TEST Accuracy: 0.910

#Best loss 0
#Best epoch 4939
#TRAIN Accuracy: 0.975
#TEST Accuracy: 0.886

#======================================================

val_losses = []
train_losses = []

# dataset definition
MODEL_PATH = 'mediapipe_facial_model.pth'
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
        self.X = df.values[:,1:12]
        self.y = df.values[:,0]        
        
        # ensure input data is floats
        self.X = self.X.astype('float32')
        
        # ======================================================
        # Hot Encode
        self.y = pd.get_dummies(self.y).to_numpy()
        self.y = self.y.astype('float32')
        #print("=====================================")
        
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

# model definition
class MLP(Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()

        # Dropout version
        #self.fc1 = Linear(input_size, 200)
        #self.act1 = Tanh()
        #self.dropout1 = Dropout(0.2)
        #self.fc2 = Linear(200, 100)
        #self.act2 = Tanh()
        #self.dropout2 = Dropout(0.2)        
        #self.fc3 = Linear(100, 10)
        #self.act3 = Tanh() 
        #self.fc4 = Linear(10, NUM_CLASSES)

        # Batchnorm version
        #self.fc1 = Linear(input_size, 200)
        #self.act1 = Tanh()
        #self.bn1 = BatchNorm1d(200)
        #self.fc2 = Linear(200, 100)
        #self.act2 = Tanh()
        #self.bn2 = BatchNorm1d(100)   
        #self.fc3 = Linear(100, 10)
        #self.act3 = Tanh() 
        #self.bn3 = BatchNorm1d(10)   
        #self.fc4 = Linear(10, NUM_CLASSES)

        # Batchnorm + Dropout version
        self.fc1 = Linear(input_size, 200)
        self.bn1 = BatchNorm1d(200)
        self.act1 = Tanh()
        self.dropout1 = Dropout(0.2)
        self.fc2 = Linear(200, 100)
        self.bn2 = BatchNorm1d(100)
        self.act2 = Tanh()
        self.dropout2 = Dropout(0.2)        
        self.fc3 = Linear(100, 10)
        self.bn3 = BatchNorm1d(10)   
        self.act3 = Tanh() 
        self.fc4 = Linear(10, NUM_CLASSES)
		
        
    def forward(self, x):

        # Dropout version
        #x = self.fc1(x)
        #x = self.act1(x)
        #x = self.dropout1(x)
        #x = self.fc2(x) 
        #x = self.act2(x)
        #x = self.dropout2(x)        
        #x = self.fc3(x) 
        #x = self.act3(x)
        #x = self.fc4(x) 
        #x = F.softmax(x, dim=1)    

		# Batchnorm version
        #x = self.fc1(x)
        #x = self.act1(x)
        #x = self.bn1(x)
        #x = self.fc2(x) 
        #x = self.act2(x)
        #x = self.bn2(x)        
        #x = self.fc3(x) 
        #x = self.act3(x)
        #x = self.bn3(x)        
        #x = self.fc4(x) 
        #x = F.softmax(x, dim=1)    

		# Batchnorm + Dropout version
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x) 
        x = self.bn2(x)        
        x = self.act2(x)
        x = self.dropout2(x)        
        x = self.fc3(x) 
        x = self.bn3(x)        
        x = self.act3(x)
        x = self.fc4(x) 
        x = F.softmax(x, dim=1)    
		
        
        return x       

#Function to Convert to ONNX 
def Convert_ONNX(): 

    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor
    input_size = 11 
    batch_size=64
    dummy_input = torch.randn(1, input_size, requires_grad=True)  

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "MediaPipe_ImageClassifier_1.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=11,    # the ONNX version to export the model to 
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
    
    # prepare data loaders
    #train_dl = DataLoader(train, batch_size=64, shuffle=False)
    train_dl = DataLoader(dataset, batch_size=64, shuffle=False)
    test_dl = DataLoader(test, batch_size=64, shuffle=False)
    return train_dl, test_dl

# train the model
def train_model(train_dl, model):
    # define the optimization
    criterion = MSELoss() #CrossEntropyLoss() #L1Loss() #MSELoss() 
    #optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay=0.0001) #lr = 0.001, weight_decay=0.0001
    optimizer = torch.optim.Adam(model.parameters()) #lr = 0.001, weight_decay=0.0001

    n_epochs = 5000  # 2000 for dropout=0.1 5K for dropout=0.2
    early_stop_thresh = 1500  #1500
    best_loss = -1
    best_epoch = -1
    
    # enumerate epochs
    for epoch in range(n_epochs):
        # enumerate mini batches
        cum_loss = 0
        #print("EPOCH=",epoch)
        
        for i, (inputs, targets) in enumerate(train_dl):
            
            #print("**",i)
            
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
            train_losses.append(loss.item())
            val_losses.append(cum_loss)
            
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
        #              %(epoch+1, n_epochs, cum_loss))

    print("Best loss %d" % best_loss)
    print("Best epoch %d" % best_epoch)       
    resume(model, "best_model.pth")            
            
# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        #print(yhat.shape, "     -     ",targets.shape)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        #print("actual:",actual.shape)
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
path = 'mediapipe_output.csv'
train_dl, test_dl = prepare_data(path)
print(len(train_dl.dataset), len(test_dl.dataset))
# define the network
model = MLP(11)
# train the model
train_model(train_dl, model)

# evaluate the model
acc = evaluate_model(train_dl, model)
print('TRAIN Accuracy: %.3f' % acc)

# evaluate the model
acc = evaluate_model(test_dl, model)
print('TEST Accuracy: %.3f' % acc)

# make a single prediction (expect class=1)
#row = [0.34,0.46,0.49,0.54,0.02,0.5,0.62,0.85,0.48,0,0.02] #0
#yhat = predict(row, model)
#print('Predicted: %.3f (class=%d)' % (yhat, yhat.round()))
#print("Predicted:", yhat)

#torch.save(model, MODEL_PATH)
torch.save(model.state_dict(), MODEL_PATH)
print("Model saved")


# Conversion to ONNX 
Convert_ONNX()  

#======================================================

plt.figure(figsize=(10,5))
plt.title("Training and Validation Loss")
plt.plot(val_losses,label="val")
plt.plot(train_losses,label="train")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

