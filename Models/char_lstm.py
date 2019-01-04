!pip install torch torchvision

import os
import re
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.utils import shuffle
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from torch.optim import lr_scheduler

torch.manual_seed(42)

torch.cuda.is_available()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pd.read_csv('../Datasets/namesdata.csv',sep=',')

data['names'] = data['names'].str.lower()

corpus = list(data['names'].as_matrix())

print(corpus[0:2])

labels = data['labels'].as_matrix()

print(labels[0:2])

avg_len = 0
for sen in corpus:
  avg_len+=len(sen)
print(avg_len/len(corpus))

maxwordlength = 8

vocablength = 27

vocab['<PAD>'] = 0

vocab = {}
for i in range(1,vocablength):
  vocab[chr(i+97)] = i

print(vocab)

length = len(corpus)

traindata,testdata,trainlabels,testlabels = train_test_split(corpus,labels,test_size=0.2,random_state=42)

traindata,valdata,trainlabels,vallabels = train_test_split(traindata,trainlabels,test_size=0.1,random_state=42)

trainlength = len(traindata)
vallength = len(valdata)
testlength = len(testdata)

print(trainlength)
print(vallength)
print(testlength)

def convert_to_vector(name):
  vector = torch.zeros(maxwordlength,vocablength)
  for i in range(min(maxwordlength,len(name))):
    vec = torch.zeros(vocablength)
    vec[vocab[name[i]]] = 1
    vector[i,:] = vec
  return vector

trainvectors = torch.zeros(trainlength,maxwordlength,vocablength).to(device)

for index in range(0,trainlength):
  trainvectors[index,:,:] = convert_to_vector(traindata[index])

valvectors = torch.zeros(vallength,maxwordlength,vocablength).to(device)

for index in range(0,vallength):
  valvectors[index,:,:] = convert_to_vector(valdata[index])

testvectors = torch.zeros(testlength,maxwordlength,vocablength).to(device)

for index in range(0,testlength):
  testvectors[index,:,:] = convert_to_vector(testdata[index])

hiddenlayer1 = 32
hiddenlayer2 = 8
numclasses = 2

class gender_prediction(nn.Module):
  def __init__(self):
    super(gender_prediction,self).__init__()
    self.lstm1 = nn.LSTM(vocablength,hiddenlayer1,batch_first=True)
    self.lstm2 = nn.LSTM(hiddenlayer1,hiddenlayer2,batch_first=True)
    self.dense1 = nn.Linear(hiddenlayer2,numclasses)
  def forward(self,x):
    out,(h0,c0) = self.lstm1(x,None)
    out,_ = self.lstm2(out,None)
    out = self.dense1(out[:,-1,:])
    return F.log_softmax(out,dim=1)

model = gender_prediction().to(device)

test = torch.zeros(4,maxwordlength,vocablength).to(device)
test[0,:,:] = convert_to_vector('pytorch')
test[1,:,:] = convert_to_vector('torchvison')
test[2,:,:] = convert_to_vector('torchtext')
test[3,:,:] = convert_to_vector('fair')

output = model(test)
print(output.shape)

optimizer = optim.Adam(model.parameters())

batchsize = 32

train_labels = torch.from_numpy(trainlabels).long().to(device)
val_labels = torch.from_numpy(vallabels).long().to(device)
test_labels = torch.from_numpy(testlabels).long().to(device)

traindataset = torch.utils.data.TensorDataset(trainvectors,train_labels)
trainloader = torch.utils.data.DataLoader(traindataset,batch_size=batchsize)

valdataset = torch.utils.data.TensorDataset(valvectors,val_labels)
valloader = torch.utils.data.DataLoader(valdataset,batch_size=batchsize)

testdataset = torch.utils.data.TensorDataset(testvectors,test_labels)
testloader = torch.utils.data.DataLoader(testdataset,batch_size=batchsize)

def get_accuracy(net,loader):
  net.eval()
  with torch.no_grad():
    avg_loss = 0.0
    acc = 0
    count = 0
    for i,(name,label) in enumerate(loader):
      outputs = net(name)
      avg_loss+=F.nll_loss(outputs,label).item() * name.size(0)
      _,preds = torch.max(outputs.data,1)
      acc+=torch.sum(preds==label.data).item()
      count+=name.size(0)
    return (avg_loss/count),((acc/count)*100)

numepochs = 30

valid_loss_min = np.Inf

best_model_wts = copy.deepcopy(model.state_dict())
for epoch in range(numepochs):
  model.train()
  print("Epoch {}".format(epoch+1))
  for i,(Xtrain,ytrain) in enumerate(trainloader):
    optimizer.zero_grad()
    output = model(Xtrain)
    loss = F.nll_loss(output,ytrain)
    loss.backward()
    optimizer.step()
    
  trainloss,trainacc = get_accuracy(model,trainloader)
  valloss,valacc = get_accuracy(model,valloader)
  print("Train Loss {} Train Accuracy {}".format(trainloss,trainacc))
  print("Validation Loss {} Validation Accuracy {}".format(valloss,valacc))
  if(valloss<valid_loss_min):
    best_model_wts = copy.deepcopy(model.state_dict())
    valid_loss_min = valloss
model.load_state_dict(best_model_wts)

test_loss,test_acc = get_accuracy(model,testloader)

print(test_acc)

torch.save(model.state_dict(),'model_weights.pt')
model_gn = gender_prediction()
model_gn.load_state_dict(torch.load('model_weights.pt'))
model_gn =  model_gn.to(device)

test_loss,test_acc = get_accuracy(model_gn,testloader)

print(test_acc)
