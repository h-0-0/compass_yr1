# "Investigating the Effect of Latent Representations on Continual Learning Performance"
Project conducted by me (Henry Bourne) and supervised by Rihuan Ke at the univeristy of bristol as part of the assessed Mini-Project unit in the compass CDT programme.  

The project investigated how latent representations could affect the amount of catastrophic forgetting exhibited by a network. This was investigated by tessting networks made up of different pre-trained encoders on the continual learning split-CIFAR10 dataset (with 5 tasks - 2 classes for each task). Please refer to "Report/main.pdf" for more information on what this repositry was used for.  

Below is the rough structure of this repositry with information on what directories/files contain:  
.  
├── Proposal/  
│   └── Folder containing pdf (and latex files) for the initial proposal for this project  
├── Report/  
│   └── Folder containing pdf (and latex files) for the finished report for this project, go here if you would like further information on the results of my experiments aswell as background information in the area of continal learning and catastrophic forgetting  
└── src/  
    ├── data/  
    │   └── Where data to be trained on was stored, such as the MNIST and CIFAR10 datasets  
    ├── encoding/  
    │   ├── This directory contains all the files that were used to pretrain VGG16, the AutoRncoder networks and then encode the datasets to be used when training the networks with frozen encoders (fVGG, fAE)  
    │   ├── encode.py/  
    │   │   └── Contains code used to pretrain and generate encodings  
    │   ├── encoders.py/  
    │   │   └── Contains code for the networks used (VGG16 and AutoEncoder)  
    │   └── job.sh/  
    │       └── Used to run encode.py using the SLURM scheduling system  
    ├── latent_CL/  
    │   └── Uses code from (https://github.com/oleksost/latent_CL) with minor adjustments to make it run with the rest of my code, this was used to generate encoded data using the pretrained (on Imagenet) Resnet18 and Resnet50 networks  
    ├── plots/  
    │   └── Contains all the final plots of various metrics recorded during training of the networks  
    ├── results/  
    │   └── Contains the csv files with the data used to generate the plots  
    ├── data.py/  
    │   └── Code used to create dataloaders for training and testing the network  
    ├── main.py/  
    │   └── Code used to run experiments, can take arguments from the command line  
    ├── opy.py/  
    │   └── Contains code for running the testing and training loops  
    ├── plot.py/  
    │   └── Code for generating plots from results saved to csv files with results.py  
    ├── results.py/  
    │   └── Code for saving metrics tracked during training and testing to csv files for plot.py to plot  
    ├── simple.py/  
    │   └── Code for all models used apart from those in pretrain.py  
    ├── pretrain.py/  
    │   └── Code for all pre-trained models with non-frozen encoders (all models with "nf" at the start)  
    └── MNIST.sh, CIFAR10.sh, CIFAR100.sh/  
        └── Bash scripts used to run experiments using the SLURM scheduling system  
