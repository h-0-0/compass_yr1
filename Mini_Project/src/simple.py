import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Class for fully connected feed forward neural network
class FC_FF_NN(nn.Module):
    def __init__(self, data_name, encoder_name=False):
        # We create the network architecture depending on the dataset
        if(data_name == "MNIST"):
            super(FC_FF_NN, self).__init__()
            self.flatten = nn.Flatten()
            self.network = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
            )
            self.device = None

        elif(data_name == "CIFAR10"):
            # We customize the input size depending on the encoder used,
            # if no encoder is used we use the original input size of the dataset
            if(encoder_name == False):
                in_features = 32*32*3
            else:
                if(encoder_name == "RN50_clip"):
                    in_features = 1024
                else:
                    raise Exception("Not given valid encoder name must be: RN50_clip")
            
            super(FC_FF_NN, self).__init__()
            self.flatten = nn.Flatten()
            self.network = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
            self.device = None

        elif(data_name == "CIFAR100"):
            # We customize the input size depending on the encoder used,
            # if no encoder is used we use the original input size of the dataset
            if(encoder_name == False):
                in_features = 32*32*3
            else:
                if(encoder_name == "RN50_clip"):
                    in_features = 1024
                else:
                    raise Exception("Not given valid encoder name must be: RN50_clip")
            
            super(FC_FF_NN, self).__init__()
            self.flatten = nn.Flatten()
            self.network = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 100)
            )
            self.device = None
        else:
            raise Exception("Not given valid dataset name must be: MNIST, CIFAR10 or CIFAR100")

    # Compute forward pass
    def forward(self, x):
        x = self.flatten(x)
        out = self.network(x)
        return out


# Class for convolutional neural network
class CNN(nn.Module):
    def __init__(self, data_name, encoder_name=False):
        # We create the network architecture depending on the dataset
        if(data_name == "MNIST"):
            super(CNN, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.fc = nn.Sequential(
                nn.Linear(32 * 5 * 5, 10)
            )
            self.flatten = nn.Flatten()
            self.device = None

        elif(data_name == "CIFAR10"):
            # We customize the input size depending on the encoder used,
            # if no encoder is used we use the original input size of the dataset
            if (encoder_name == False):
                in_channels = 3
            elif(encoder_name == "RN50_clip"):
                in_channels = 1
            else:
                raise Exception("Not given valid encoder name must be: RN50_clip")
            super(CNN, self).__init__()
            # Based on VGG16 architecture
            self.conv = nn.Sequential(
                # Block 1
                nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(3,3), stride=1, padding="same"),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding="same"),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2,2)),
                # Block 2
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding="same"),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding="same"),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2,2)),
                # Block 3
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=1, padding="same"),
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding="same"),
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding="same"),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2,2)),
                # Block 4
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=1, padding="same"),
                nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding="same"),
                nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding="same"),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2,2)),
                # Block 5
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding="same"),
                nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding="same"),
                nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding="same"),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2,2))
            )
            self.fc = nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
            self.flatten = nn.Flatten()
            self.device = None

        elif(data_name == "CIFAR100"):
            if (encoder_name == False):
                in_channels = 3
            elif(encoder_name == "RN50_clip"):
                in_channels = 1
            else:
                raise Exception("Not given valid encoder name must be: RN50_clip")
            super(CNN, self).__init__()

            # 3 VGG blocks
            self.conv = nn.Sequential(
                # Block 1
                nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3,3), stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2,2)),
                # Block 2
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2,2))
                # Block 3
                # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding=0),
                # nn.ReLU(),
                # nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=0),
                # nn.ReLU(),
                # nn.MaxPool2d(kernel_size=(2,2))
            )
            self.fc = nn.Sequential(
                nn.Linear(1600, 256),
                nn.ReLU(),
                nn.Linear(256, 100)
            )
            self.flatten = nn.Flatten()
            self.device = None
        else:
            raise Exception("Not given valid dataset name must be: MNIST, CIFAR10 or CIFAR100")

    # Compute forward pass
    def forward(self, x):
        out = self.conv(x)
        out = self.flatten(out)
        out = self.fc(out)
        return out

# TODO: check VGG big results, maybe batch nomralization? maybe make fully connected layers in CNN and FC_FF_NN with encoder the same?