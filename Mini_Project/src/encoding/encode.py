from argparse import ArgumentParser
from continuum.datasets import MNIST, CIFAR10, CIFAR100
import torch
import sys 
sys.path.append('..')
import data 
import opt
import matplotlib.pyplot as plt
import h5py
from encoders import VGG
import os
import numpy as np


# Returns model for encoder
def detect_model(encoder_name, data_name):
    if(encoder_name == "fVGG"):
        return VGG(data_name)
    else:
        raise Exception("Not given valid encoder name must be: fVGG")

# Function that puts model on device and adds device to model
def handle_device(model, device):
    model.to(device)
    model.device = device

# Used to create model, optimizer and prev_epoch, if load_model is True then load model from saved_models folder
def get_model(load_model, encoder_name, data_name, learning_rate, epochs, device, is_CL=False, optimizer_type="SGD"):
    if is_CL:
        CL_ext = "_CL"
    else:
        CL_ext = ""
    # Load model if load_model is True
    if(load_model == True):
        # Check if saved_models folder exists
        if not os.path.exists("saved_models"):
            raise Exception("No saved_models folder found, cannot load model")
        # Load model
        model = detect_model(encoder_name, data_name)
        checkpoint = torch.load("saved_models/"+ data_name+ "_"+encoder_name+CL_ext+".pth", map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])

        # Create optimizer
        if optimizer_type == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        elif optimizer_type == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:
            raise Exception("Not a valid optimizer type, must be SGD or Adam")
    
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if is_CL:
            prev_epoch = False
        else:
            prev_epoch = checkpoint['epoch']
            if(prev_epoch > epochs):
                raise Exception("Cannot load model from epoch: "+str(prev_epoch)+" and train for a total of: "+str(epochs)+" epochs")

        handle_device(model, device)
        print("Loaded PyTorch Model State from saved_models/"+ data_name + "_" + encoder_name+ CL_ext+".pth")
    elif(load_model == False):
        model = detect_model(encoder_name, data_name)

        # Create optimizer
        if optimizer_type == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        elif optimizer_type == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:
            raise Exception("Not a valid optimizer type, must be SGD or Adam")
        
        handle_device(model, device)
        if is_CL:
            prev_epoch = False
        else:
            prev_epoch = 0
    else:
        raise Exception("load_model must be a boolean, got: "+str(load_model)+" of type: "+str(type(load_model)))
    return model, optimizer, prev_epoch

# Saves the training and testing loss values to a file and plots a graph.
def save_loss(train, test, file_path):
    # Create a new figure and axes for the plot
    fig, ax = plt.subplots()
    
    # Plot the loss and accuracy values on their respective axes
    ax.plot(train, label="Training Loss")
    ax.plot(test, label="Test Loss")
    
    # Add a legend and set the title
    ax.legend()
    ax.set_title("Training and Testing Loss")

    # Save the plot to a file
    fig.savefig(file_path)
    
    # Save the loss and accuracy values to a file
    with open(file_path + ".txt", "w") as file:
        file.write("Training Loss\tTesting Loss\n")
        for tr, te in zip(train, test):
            file.write(str(tr) + "\t" + str(te) + "\n")

# Saves the training and testing accuracy values to a file and plots a graph.
def save_acc(train, test, file_path):
    # Create a new figure and axes for the plot
    fig, ax = plt.subplots()
    
    # Plot the loss and accuracy values on their respective axes
    ax.plot(train, label="Training Accuracy")
    ax.plot(test, label="Test Accuracy")
    
    # Add a legend and set the title
    ax.legend()
    ax.set_title("Training and Testing Accuracy")

    # Save the plot to a file
    fig.savefig(file_path)
    
    # Save the loss and accuracy values to a file
    with open(file_path + ".txt", "w") as file:
        file.write("Training Accuracy\tTesting Accuracy\n")
        for tr, te in zip(train, test):
            file.write(str(tr) + "\t" + str(te) + "\n")
        
# Encodes the data using the given encoder and saves it to a HDF5 file
def encode_and_save(encoder, data_loader, device, file_path):
    # Set the encoder to evaluation mode
    encoder.eval()

    # Create a new HDF5 file for storing the encoded data
    with h5py.File(file_path, "w") as h5_file:
        N = len(data_loader.dataset)
        data = h5_file.create_dataset('data', shape=(N, 512, 1, 1), dtype=np.float32, fillvalue=0)
        targets = h5_file.create_dataset('targets', shape=(N, 1), dtype=np.float32, fillvalue=0)
        for i, (X, y, *ignore) in enumerate(data_loader):
            # Encode the inputs using the encoder
            X  = X.to(device)
            encoded = encoder(X).cpu().detach().numpy()

            y = y.cpu().detach().numpy()

            # Save the encoded data to the HDF5 file
            for j in range(len(encoded)):
                data[i*len(encoded)+j] = encoded[j]
                targets[i*len(encoded)+j] = y[j]

# A function to encode data, when not in the CL scenario
def run_encoding(data_name, encoder_name, batch_size=128, learning_rate=0.001, epochs=100, optimizer="Adam", load_model=False):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get training and testing data
    train_dataloader, test_dataloader = data.get_data_loader(data_name, batch_size, device)

    # Get model
    model, optimizer, prev_epoch = get_model(load_model, encoder_name, data_name, learning_rate, epochs, device, is_CL=False, optimizer_type=optimizer)

    # Train model if not loading pre-trained or if we want to train pre-trained for more epochs
    if (load_model == False) or (prev_epoch < epochs) :
        # Train model
        train_losses, test_losses, train_accs, test_accs, times, optimizer = opt.train(
            model, 
            train_dataloader, 
            test_dataloader, 
            optimizer,
            learning_rate=learning_rate, 
            epochs=(prev_epoch, epochs)
        )
        # Save losses and accuracies
        if not os.path.exists("train_results"):
            os.makedirs("train_results")
        save_loss(train_losses, test_losses, "train_results/"+data_name+ "_"+ encoder_name + "_loss", training=True)
        save_acc(train_accs, test_accs, "train_results/"+data_name+ "_"+ encoder_name + "_acc", training=False)

        # Save model
        if not os.path.exists("saved_models"):
            os.makedirs("saved_models")
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, 
            "saved_models/"+data_name+ "_"+ encoder_name + ".pth"
        )

    # Extract encoder from model
    encoder = model.encoder

    # Save encoder
    torch.save(encoder.state_dict(), "trained_encoders/"+data_name+ "_"+ encoder_name + ".pth")

    # Encode data and save to hdf5 file
    encode_and_save(encoder, train_dataloader, device, "../data/EncodedDatasets/"+data_name+ "_"+ encoder_name + "_train.hdf5")
    encode_and_save(encoder, test_dataloader, device, "../data/EncodedDatasets/"+data_name+ "_"+ encoder_name + "_test.hdf5")
    print("Saved encoded data to file")

    
    

# Main function, encodes data according to arguments
def main(args, is_CL: bool=False):
    if is_CL:
        raise Exception("Not implemented CL scenario yet")
    else:
        if(args.data_name == "CIFAR10"):
            run_encoding(args.data_name, args.encoder_name, batch_size=128, learning_rate=0.0001, epochs=100, optimizer="Adam", load_model=args.load_model)
        elif(args.data_name == "CIFAR100_sub"):
            run_encoding(args.data_name, args.encoder_name, batch_size=128, learning_rate=0.0001, epochs=100, optimizer="Adam", load_model=args.load_model)
        else:
            raise Exception("Not implemented args for other datasets yet")
    print("Done")

if __name__ == "__main__":
    # Create argument parser
    parser = ArgumentParser()

    # Argment for data name
    parser.add_argument("--data_name", type=str, default="CIFAR10", help="Name of data set to use")

    # Argument for encoder name
    parser.add_argument("--encoder_name", type=str, default="fVGG", help="Name of encoder to use")

    # Argument for loading model
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--no-load_model', dest='load_model', action='store_false')
    parser.set_defaults(load_model=False)
    
    # Arguments related to CL, if n-tasks given will run CL experiment
    parser.add_argument("--n_tasks", type=int, help="Number of tasks", default=-1)
    parser.add_argument("--init_inc", type=int, help="Number of classes for first task", default=2)

    # Parse arguments and check if CL experiment
    args = parser.parse_args()
    is_CL = (args.n_tasks != -1)

    # Encode data
    main(args, is_CL)

# TODO: CL case