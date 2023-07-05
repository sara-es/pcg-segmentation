import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))

from DataManipulation.AudioDataset import *
from DataManipulation.DataPresentation import DataPresentation
from Utilities.constants import * 
import torch
from torch.utils.data import DataLoader
from MurmurPrediction import *


def accuracy(model, data):
    _, yhat= torch.max(model(data.audio_tensor), 1)
    return (yhat == dataset.murmurs).numpy().mean()

def training(epochs):
    LOSS = [] 
    ACC = [] 
    for epoch in range(epochs):
        for x,y in trainloader:
            yhat = model(x)
            loss = criterion(yhat, y)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
        LOSS.append(loss)
        ACC.append(accuracy(model, dataset))
    return LOSS, ACC 


dataset = AudioData("training_data.csv")
train, test = split_data(dataset, 0.7)

trainloader = DataLoader(train, batch_size=32, shuffle=True)
testloader = DataLoader(test, batch_size=len(test), shuffle=False)

model = DNN([NUM_FRAMES, 50, 50, 50, 3])
model.train()
criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model.parameters(), lr=0.05)

loss, acc = training(100)


dp = DataPresentation()

dp.plot_loss_and_accuracy(loss, acc)

model.eval()
for data,label in testloader:
    _, yhat= torch.max(model(data), 1)
    test_accuracy = (yhat == label).numpy().mean()
    print(test_accuracy)
