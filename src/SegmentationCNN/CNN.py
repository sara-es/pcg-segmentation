
from doctest import OutputChecker
from turtle import back, backward
import torch
import torch.nn as nn
from utils import get_wavs_and_tsvs
from DataPreprocessing import DataPreprocessing

class CNN(nn.Module):

    IN_CHANNELS = 4 # Corresponding to number of envelopes
    OUT_CHANNELS = 4  # Coresponding to FHS 
    
    def __init__(self):
        super(CNN, self).__init__()
        
        # Defining objects for convolutional layer 1 
        self.conv1 = nn.Conv1d(in_channels=self.IN_CHANNELS, out_channels=8, padding="same", kernel_size=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=8, padding="same", kernel_size=3)
        self.relu2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        
        # # Defining objects for convolutional layer 2
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=16, padding="same", kernel_size=3)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv1d(in_channels=16, out_channels=16, padding="same", kernel_size=3)
        self.relu4 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)

        self.conv5 = nn.Conv1d(in_channels=16, out_channels=32, padding="same", kernel_size=3)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv1d(in_channels=32, out_channels=32, padding="same", kernel_size=3)
        self.relu6 = nn.ReLU()
        self.maxpool3 = nn.MaxPool1d(kernel_size=2)

        self.conv7 = nn.Conv1d(in_channels=32, out_channels=64, padding="same", kernel_size=3)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv1d(in_channels=64, out_channels=64, padding="same", kernel_size=3)
        self.relu8 = nn.ReLU()
        self.maxpool4 = nn.MaxPool1d(kernel_size=2)

        self.conv9 = nn.Conv1d(in_channels=64, out_channels=128, padding="same", kernel_size=3)
        self.relu9 = nn.ReLU()
        self.conv10 = nn.Conv1d(in_channels=128, out_channels=128, padding="same", kernel_size=3)
        self.relu10 = nn.ReLU()

        # self.upsample1 = nn.Upsample()

        self.deconv1 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=5)
        self.relu11 = nn.ReLU()
        self.uconv8 = nn.Conv1d(in_channels=64, out_channels=64, padding="same", kernel_size=3)
        self.relu12 = nn.ReLU()
        self.uconv7 = nn.Conv1d(in_channels=64, out_channels=64, padding="same", kernel_size=3)
        self.relu13 = nn.ReLU()

        # self.deconv2 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3)
        # self.relu14 = nn.ReLU()
        # self.uconv6 = nn.Conv1d(in_channels=32, out_channels=32, padding="same", kernel_size=3)
        # self.relu15 = nn.ReLU()
        # self.uconv5 = nn.Conv1d(in_channels=32, out_channels=32, padding="same", kernel_size=3)
        # self.relu16 = nn.ReLU()

        # self.deconv3 = nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=3)
        # self.relu17 = nn.ReLU()
        # self.uconv4 = nn.Conv1d(in_channels=16, out_channels=16, padding="same", kernel_size=3)
        # self.relu18 = nn.ReLU()
        # self.uconv3 = nn.Conv1d(in_channels=16, out_channels=16, padding="same", kernel_size=3)
        # self.relu19 = nn.ReLU()

        # self.deconv4 = nn.ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=3)
        # self.relu20 = nn.ReLU()
        # self.uconv2 = nn.Conv1d(in_channels=8, out_channels=8, padding="same", kernel_size=3)
        # self.relu21 = nn.ReLU()
        # self.uconv1 = nn.Conv1d(in_channels=8, out_channels=8, padding="same", kernel_size=3)
        # self.relu22 = nn.ReLU()

        # self.conv_to_output = nn.Conv1d(in_channels=8, out_channels=4, padding="same", kernel_size=3)
        # self.fc = nn.Linear(4*64, 4)

        # # the first parameter is number of out_channels x dimensions of resulting activation map 
        # # the second parameter represents the number of classes
        # self.fc1 = nn.Linear(out_2 * 4 * 4, 10)
        
    
    def forward(self, x):
        # first convolutional layer
        layer1 = self.conv1(x)
        layer1 = self.relu1(layer1)
        layer1 = self.conv2(layer1)
        layer1 = self.relu2(layer1)

        print(layer1.shape)  

        layer2 = self.maxpool1(layer1)

        layer2 = self.conv3(layer2)
        layer2 = self.relu3(layer2)
        layer2 = self.conv4(layer2)
        layer2 = self.relu4(layer2)

        print(layer2.shape)  

        layer3 = self.maxpool2(layer2)

        layer3 = self.conv5(layer3)
        layer3 = self.relu5(layer3)
        layer3 = self.conv6(layer3)
        layer3 = self.relu6(layer3)

        print(layer3.shape)  

        layer4 = self.maxpool3(layer3)

        layer4 = self.conv7(layer4)
        layer4 = self.relu7(layer4)
        layer4 = self.conv8(layer4)
        layer4 = self.relu8(layer4)

        print(layer4.shape) 

        layer5 = self.maxpool4(layer4)

        layer5 = self.conv9(layer5)
        layer5 = self.relu9(layer5)
        layer5 = self.conv10(layer5)
        layer5 = self.relu10(layer5)

        print(layer5.shape) 

        backwards_layer5 = self.deconv1(layer5)   

        print(backwards_layer5.shape)
        backwards_layer5 = self.relu11(backwards_layer5)
        concat_layer4 = torch.cat([backwards_layer5, layer4], dim=1)

        print(concat_layer4.shape)

        backwards_layer4 = self.uconv8(concat_layer4)
        backwards_layer4 = self.relu12(backwards_layer4)
        backwards_layer4 = self.uconv7(backwards_layer4)
        backwards_layer4 = self.relu13(backwards_layer4)

        print(backwards_layer4.shape)
   

        # backwards_layer4 = self.deconv2(backwards_layer4)     
        # backwards_layer4 = self.relu14(backwards_layer4)
        # concat_layer3 = torch.cat([backwards_layer4, layer3], dim=1)
        # backwards_layer3 = self.uconv6(concat_layer3)
        # backwards_layer3 = self.relu15(backwards_layer3)
        # backwards_layer3 = self.uconv5(backwards_layer3)
        # backwards_layer3 = self.relu16(backwards_layer3)

        # backwards_layer3 = self.deconv3(backwards_layer3)     
        # backwards_layer3 = self.relu17(backwards_layer3)
        # concat_layer2 = torch.cat([backwards_layer3, layer2], dim=1)
        # backwards_layer2 = self.uconv4(concat_layer2)
        # backwards_layer2 = self.relu18(backwards_layer2)
        # backwards_layer2 = self.uconv3(backwards_layer2)
        # backwards_layer2 = self.relu19(backwards_layer2)

        # backwards_layer2 = self.deconv4(backwards_layer2)     
        # backwards_layer2 = self.relu20(backwards_layer2)
        # concat_layer1 = torch.cat([backwards_layer2, layer1], dim=1)
        # backwards_layer1 = self.uconv2(concat_layer1)
        # backwards_layer1 = self.relu21(backwards_layer1)
        # backwards_layer1 = self.uconv1(backwards_layer1)
        # backwards_layer1 = self.relu22(backwards_layer1)

        # out = self.conv_to_output(backwards_layer1)
        
        # out = out.view(out.size(0), -1)
        # out = self.fc(out)
 
        return None


wavs, tsvs, names = get_wavs_and_tsvs("/Users/serenahuston/GitRepos/python-classifier-2022/physionet.org/files/circor-heart-sound/1.0.3/training_data",
                                return_names=True)


dp = DataPreprocessing(wavs[0], tsvs[0], names[0])
dp.extract_patches()

model = CNN()

tensor = torch.from_numpy(dp.input_patches[0]).type(torch.float32)

print(tensor.dtype)
print(model.forward(tensor))