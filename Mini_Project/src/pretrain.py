import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from encoding.encoders import VGG, AutoEncoder
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights

# Class for pretrained encoder with fully connected feed forward network on top
class encoder_FC_FF_NN(nn.Module):
    def __init__(self, data_name, encoder_name, device):
        super(encoder_FC_FF_NN, self).__init__()
        self.encoder_name = encoder_name
        # We initialize the encoder for the network depending on the encoder name
        if encoder_name == "nfVGG":
            if data_name == "CIFAR10":
                model = VGG("CIFAR100")
                print("Loading weights for VGG onto device: ", device)
                weights = torch.load("encoding/trained_encoders/CIFAR100_fVGG.pth", map_location=torch.device("cpu"))
                model.encoder.load_state_dict(weights)
                self.encoder = model.encoder
                self.preprocess = lambda x: x
            else:
                raise Exception("Not given valid dataset name must be: CIFAR10")
        elif encoder_name == "nfResnet18":
            if data_name == "CIFAR10":
                weights = ResNet18_Weights.DEFAULT
                model = resnet18(weights=weights)
                self.encoder = torch.nn.Sequential(*list(model.children())[:-1])
                self.preprocess = weights.transforms()
            else:
                raise Exception("Not given valid dataset name must be: CIFAR10")
        elif encoder_name == "nfResnet50":
            if data_name == "CIFAR10":
                weights = ResNet50_Weights.DEFAULT
                model = resnet50(weights=weights)
                self.encoder = torch.nn.Sequential(*list(model.children())[:-1])
                self.preprocess = weights.transforms()
            else:
                raise Exception("Not given valid dataset name must be: CIFAR10")
        elif(encoder_name == "nfAE"):
            if(data_name == "CIFAR10"):
                model = AutoEncoder(256, num_input_channels=3)
                print("The device is: ", device)
                weights = torch.load("encoding/trained_encoders/CIFAR100_fAE.pth")
                model.encoder.load_state_dict(weights)
                self.encoder = model.encoder
                self.preprocess = lambda x: x
            else:
                raise Exception("Not given valid dataset name must be: CIFAR10")
        else:
            raise Exception("Not given valid encoder name must be: nfVGG, nfResnet18, nfResnet50 or nfAE")
        
        # We create the network architecture for fully connected layer depending on the dataset
        if(data_name == "CIFAR10"):
            # We customize the input size depending on the encoder used, if no encoder is used we use the original input size of the dataset
            if(encoder_name == False):
                in_features = 32*32*3
            else:
                if(encoder_name == "nfVGG"):
                    in_features = 512
                elif(encoder_name == "nfResnet18"):
                    in_features = 512
                elif(encoder_name == "nfAE"):
                    in_features = 256
                else:
                    raise Exception("Not given valid encoder name must be: nfVGG, nfResnet18 or nfAE")
            self.flatten = nn.Flatten()
            self.fc = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 10),
            )
            self.device = None

        elif(data_name == "CIFAR100"):
            # We customize the input size depending on the encoder used, if no encoder is used we use the original input size of the dataset
            if(encoder_name == False):
                in_features = 32*32*3
            else:
                if(encoder_name == "nfVGG"):
                    in_features = 512
                else:
                    raise Exception("Not given valid encoder name must be: nfVGG")
            self.flatten = nn.Flatten()
            self.fc = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 100),
            )
            self.device = None
        else:
            raise Exception("Not given valid dataset name must be: CIFAR10 or CIFAR100")

    # Compute forward pass
    def forward(self, x):
        x = self.preprocess(x)
        x = self.encoder(x)
        x = self.flatten(x)
        out = self.fc(x)
        return out