from argparse import ArgumentParser
from continuum.datasets import MNIST, CIFAR10, CIFAR100
import torch
import sys 
sys.path.append('..')
import data 
import opt
import matplotlib.pyplot as plt
import h5py
from encoders import VGG, AutoEncoder
import os
import numpy as np
from continuum.scenarios import encode_scenario
from continuum import ClassIncremental


# Returns model for encoder
def detect_model(encoder_name, data_name):
    if(encoder_name == "fVGG"):
        return VGG(data_name)
    elif(encoder_name == "fAE"):
        return AutoEncoder(256, num_input_channels=3)
    else:
        raise Exception("Not given valid encoder name must be: fVGG")

# Function that puts model on device and adds device to model
def handle_device(model, device):
    model.to(device)
    model.device = device

# Used to create model, optimizer and prev_epoch, if load_model is True then load model from saved_models folder
def get_model(load_model, encoder_name, data_name, learning_rate, epochs, device, optimizer_type="SGD"):
    # Load model if load_model is True
    if(load_model == True):
        if encoder_name == "fAE":
            # Check if saved_models folder exists
            if not os.path.exists("saved_models"):
                raise Exception("No saved_models folder found, cannot load model")
            # Load model
            print("Loading PyTorch Model State from saved_models/autoencoder_256.ckpt")
            model = detect_model(encoder_name, data_name)
            optimizer = None
            prev_epoch = epochs
            print("Loaded PyTorch Model State from saved_models/autoencoder_256.ckpt")
        else:
            # Check if saved_models folder exists
            if not os.path.exists("saved_models"):
                raise Exception("No saved_models folder found, cannot load model")
            # Load model
            print("Loading PyTorch Model State from saved_models/"+ data_name + "_" + encoder_name+".pth")
            model = detect_model(encoder_name, data_name)
            checkpoint = torch.load("saved_models/"+ data_name+ "_"+encoder_name+".pth", map_location=torch.device(device))
            model.load_state_dict(checkpoint['model_state_dict'])

            # Create optimizer
            if optimizer_type == "SGD":
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            elif optimizer_type == "Adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            else:
                raise Exception("Not a valid optimizer type, must be SGD or Adam")
        
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            prev_epoch = checkpoint['epoch']
            if(prev_epoch > epochs):
                raise Exception("Cannot load model from epoch: "+str(prev_epoch)+" and train for a total of: "+str(epochs)+" epochs")

            handle_device(model, device)
            print("Loaded PyTorch Model State from saved_models/"+ data_name + "_" + encoder_name+".pth")
    elif(load_model == False):
        if encoder_name == "fAE":
            raise Exception("Dont't support training autoencoder from scratch, must be trained and stored in saved_models folder")
        
        model = detect_model(encoder_name, data_name)

        # Create optimizer
        if optimizer_type == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        elif optimizer_type == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:
            raise Exception("Not a valid optimizer type, must be SGD or Adam")
        
        handle_device(model, device)
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
def encode_and_save(encoder_name, encoder, data_loader, device, file_path):
    # Set the encoder to evaluation mode
    encoder.eval()

    # Create a new HDF5 file for storing the encoded data
    with h5py.File(file_path, "w") as h5_file:
        N = len(data_loader.dataset)
        if encoder_name == "fVGG":
            lat_dim = 512
        elif encoder_name == "fAE":
            lat_dim = 256
        else:
            raise Exception("Not a valid encoder name, must be fVGG or fAE")
        data = h5_file.create_dataset('data', shape=(N, lat_dim, 1, 1), dtype=np.float32, fillvalue=0)
        targets = h5_file.create_dataset('targets', shape=(N, 1), dtype=np.float32, fillvalue=0)
        for i, (X, y, *ignore) in enumerate(data_loader):
            # Encode the inputs using the encoder
            X  = X.to(device)
            encoded = encoder(X).cpu().detach().numpy()

            y = y.cpu().detach().numpy()

            # Save the encoded data to the HDF5 file
            for j in range(len(encoded)):
                data[i*len(encoded)+j] = encoded[j].reshape((lat_dim, 1, 1))
                targets[i*len(encoded)+j] = y[j]

# A function to encode data, when not in the CL scenario
def run_encoding(train_data_name, encode_data_name, encoder_name, batch_size=128, learning_rate=0.001, epochs=100, optimizer="Adam", load_model=False):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Print some information about the run
    print("Running encoding with the following parameters:")
    print("train_data_name:", train_data_name)
    print("encode_data_name:", encode_data_name)
    print("encoder_name:", encoder_name)
    print("batch_size:", batch_size)
    print("learning_rate:", learning_rate)
    print("epochs:", epochs)
    print("optimizer:", optimizer)
    print("load_model:", load_model)
    print("device:", device)

    # Get model
    model, optimizer, prev_epoch = get_model(load_model, encoder_name, train_data_name, learning_rate, epochs, device, optimizer_type=optimizer)

    # Train model if not loading pre-trained or if we want to train pre-trained for more epochs
    if (load_model == False) or (prev_epoch < epochs) :
        # Get training and testing data for training the model
        train_dataloader, test_dataloader = data.get_data_loader(train_data_name, batch_size, device)
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
        save_loss(train_losses, test_losses, "train_results/"+train_data_name+ "_"+ encoder_name + "_loss")
        save_acc(train_accs, test_accs, "train_results/"+train_data_name+ "_"+ encoder_name + "_acc")

        # Save model
        if not os.path.exists("saved_models"):
            os.makedirs("saved_models")
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, 
            "saved_models/"+train_data_name+ "_"+ encoder_name + ".pth"
        )
    print("Model ready")
    # Extract encoder from model
    encoder = model.encoder

    # Save encoder
    torch.save(encoder.state_dict(), "trained_encoders/"+train_data_name+ "_"+ encoder_name + ".pth")
    print("Encoder ready! and saved!")
    # Get training and testing data to encode
    train_dataloader, test_dataloader = data.get_data_loader(encode_data_name, batch_size, device)
    
    # Encode data and save to hdf5 file
    encode_and_save(encoder_name, encoder, train_dataloader, device, "../data/EncodedDatasets/" + encode_data_name +"_by_" +train_data_name+ "_"+ encoder_name + "_train.hdf5")
    encode_and_save(encoder_name, encoder, test_dataloader, device, "../data/EncodedDatasets/" + encode_data_name +"_by_" +train_data_name+ "_"+ encoder_name + "_test.hdf5")
    print("Saved encoded data to file")

# A function to encode data, when in a CL scenario
def run_encoding_CL(train_data_name, encode_data_name, encoder_name, n_tasks, init_inc=2, batch_size=128, learning_rate=0.001, epochs=100, optimizer="Adam", load_model=False):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Print some information about the run
    print("Running encoding with the following parameters:")
    print("train_data_name:", train_data_name)
    print("encode_data_name:", encode_data_name)
    print("encoder_name:", encoder_name)
    print("n_tasks:", n_tasks)
    print("init_inc:", init_inc)
    print("batch_size:", batch_size)
    print("learning_rate:", learning_rate)
    print("epochs:", epochs)
    print("optimizer:", optimizer)
    print("load_model:", load_model)
    print("device:", device)

    # Get model
    model, optimizer, prev_epoch = get_model(load_model, encoder_name, train_data_name, learning_rate, epochs, device, optimizer_type=optimizer)
    # Train model if not loading pre-trained or if we want to train pre-trained for more epochs
    if (load_model == False) or (prev_epoch < epochs) :
        # Get training and testing data
        train_dataloader, test_dataloader = data.get_data_loader(train_data_name, batch_size, device)
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
        save_loss(train_losses, test_losses, "train_results/"+train_data_name+ "_"+ encoder_name + "_loss")
        save_acc(train_accs, test_accs, "train_results/"+train_data_name+ "_"+ encoder_name + "_acc")

        # Save model
        if not os.path.exists("saved_models"):
            os.makedirs("saved_models")
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, 
            "saved_models/"+train_data_name+ "_"+ encoder_name + ".pth"
        )
    print("Model ready!")
    # Extract encoder from model
    encoder = model.encoder
    # Save encoder
    torch.save(encoder.state_dict(), "trained_encoders/"+train_data_name+ "_"+ encoder_name + ".pth")
    print("Encoder ready! and saved!")

    # Get training and testing scenarios to encode 
    train_scenario, test_scenario = data.get_data_loader_CL(encode_data_name, batch_size, device, n_tasks, init_inc=init_inc)

    # Encode data and save to hdf5 file
    encode_scenario(train_scenario, encoder, batch_size, "../data/EncodedDatasets/"+ encode_data_name +"_by_"+train_data_name+ "_"+ encoder_name + "_train_CL.hdf5")
    encode_scenario(test_scenario, encoder, batch_size, "../data/EncodedDatasets/"+ encode_data_name +"_by_"+train_data_name+ "_"+ encoder_name + "_test_CL.hdf5")
    print("Saved encoded data to file")
    
    

# Main function, encodes data according to arguments
def main(args, is_CL: bool=False):
    if is_CL:
        if(args.encode_data_name == "CIFAR10"):
            """
            Will use specified encoder trained on CIFAR100 to encode CIFAR10 data (what the first three arguments specify), 
            n_tasks: is the number of tasks to use in the CL scenario for the encoding,
            init_inc: is the number of classes to add in the first task for the encoding,
            batch_size: is the batch size to use for the encoding and training,
            learning_rate: is the learning rate to use for the training,
            epochs: is the number of epochs to train for if not loading a pre-trained model or if we want to train for more epochs (will load pre-trained model and will pick up from where it left off in terms of epochs),
            optimizer: is the optimizer to use for the training,
            load_model: is a boolean to indicate if we want to load a pre-trained model or not
            """
            run_encoding_CL(args.train_data_name, args.encode_data_name, args.encoder_name, n_tasks=args.n_tasks, init_inc=args.init_inc, batch_size=64, learning_rate=0.0001, epochs=200, optimizer="Adam", load_model=args.load_model)
        else:
            raise Exception("Not implemented args for datasets other than encoding CIFAR10 yet")
    else:
        if(args.encode_data_name == "CIFAR10"):
            """
            Will use specified encoder trained on CIFAR100 to encode CIFAR10 data (what the first three arguments specify),
            batch_size: is the batch size to use for the encoding and training,
            learning_rate: is the learning rate to use for the training,
            epochs: is the number of epochs to train for if not loading a pre-trained model or if we want to train for more epochs (will load pre-trained model and will pick up from where it left off in terms of epochs),
            optimizer: is the optimizer to use for the training,
            load_model: is a boolean to indicate if we want to load a pre-trained model or not
            """
            run_encoding(args.train_data_name, args.encode_data_name, args.encoder_name, batch_size=64, learning_rate=0.0001, epochs=200, optimizer="Adam", load_model=args.load_model)
        else:
            raise Exception("Not implemented args for datasets other than encoding CIFAR10 yet")
    print("Done")

if __name__ == "__main__":
    # Create argument parser
    parser = ArgumentParser()

    # Argments for data set to encode and train encoder on
    parser.add_argument("--train_data_name", type=str, default="CIFAR100", help="Name of data set used to train encoder")
    parser.add_argument("--encode_data_name", type=str, default="CIFAR10", help="Name of data set we want to encode")

    # Argument for encoder name
    parser.add_argument("--encoder_name", type=str, default="fVGG", help="Name of encoder to use")

    # Argument for loading model
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--no-load_model', dest='load_model', action='store_false')
    parser.set_defaults(load_model=True)
    
    # Arguments related to CL, if n-tasks given will run CL experiment
    parser.add_argument("--n_tasks", type=int, help="Number of tasks", default=-1)
    parser.add_argument("--init_inc", type=int, help="Number of classes for first task", default=2)

    # Parse arguments and check if CL experiment
    args = parser.parse_args()
    is_CL = (args.n_tasks != -1)

    # Encode data
    main(args, is_CL)

# TODO: If you pretrain on CIFAR 100, encode CIFAR 10, etc.