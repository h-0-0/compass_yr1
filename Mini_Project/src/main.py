import os
import torch
import data
import simple
import opt
import results

# Given model_name detects what model to create
def get_model(model_name):
    if("FC_FF_NN" in model_name):
        return simple.FC_FF_NN()
    elif("CNN" in model_name):
        return simple.CNN()
    else:
        raise Exception("No valid models found in: "+model_name) 

# Function that puts model on device and adds device to model
def handle_device(model, device):
    model.to(device)
    model.device = device

# Main function
def main(data_name="MNIST", model_name="FF_FC_NN", batch_size=64, learning_rate=1e-3, epochs=5, load_model=False, save_model=False):
    # Use GPU if available else use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # Load data
    train_dataloader, test_dataloader = data.get_data_loader("MNIST", 64, device)

    # Create model or load model
    if(load_model!=False):
        # Check if saved_models folder exists
        if not os.path.exists("saved_models"):
            raise Exception("No saved_models folder found, cannot load model")
        # Load model
        model = torch.load("saved_models/"+load_model+".pth")
        handle_device(model, device)
        print("Loaded PyTorch Model State from saved_models/"+load_model+".pth")
    else:
        model = get_model(model_name)
        handle_device(model, device)

    # Train model
    train_losses, test_losses, train_accs, test_accs = opt.train(
        model, 
        train_dataloader, 
        test_dataloader, 
        learning_rate=learning_rate, 
        epochs=epochs
    )
    
    # We add the results to the results dataframe
    results.update_results(model_name, epochs, batch_size, learning_rate, train_losses, test_losses, train_accs, test_accs)

    # Save model if save_name is given
    if(save_model==True):
        # Check if saved_models folder exists
        if not os.path.exists("saved_models"):
            os.makedirs("saved_models")
        # Save model
        torch.save(model, "saved_models/"+model_name+".pth")
        print("Saved PyTorch Model State to saved_models/"+model_name+".pth")

main(
    data_name="MNIST", 
    model_name="simple_FC_FF_NN",
    batch_size=50, 
    learning_rate=0.00005, 
    epochs=20, 
    load_model=False,
    save_model=True
    )

import plot
plot.plot_loss(results.load_results(), "simple_FC_FF_NN")
plot.plot_loss(results.load_results(), "simple_CNN")

# TODO: change results so is for certain dataset, or add column to dataframe containing dataset name