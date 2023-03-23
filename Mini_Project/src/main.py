import os
import torch
import data
import simple
import opt
import results
import plot

# Given model_name detects what model to create
def get_model(model_name, data_name, encoder_name):
    if("FC_FF_NN" in model_name):
        return simple.FC_FF_NN(data_name, encoder_name)
    elif("CNN" in model_name):
        return simple.CNN(data_name, encoder_name)
    else:
        raise Exception("No valid models found in: "+model_name) 

# Given model_name detects if it contains the name of one of our encoders
def is_encoder(model_name):
    if("RN50_clip" in model_name):
        return "RN50_clip"
    else:
        return False

# Function that puts model on device and adds device to model
def handle_device(model, device):
    model.to(device)
    model.device = device

# Run an experiment
def run_exp(data_name="MNIST", model_name="RN50_clip_FF_FC_NN", batch_size=64, learning_rate=1e-3, epochs=5, load_model=False, save_model=False):
    print("\n \n \n---------------------------- New Experiment ----------------------------")
    print("Data: "+data_name, "Model: "+model_name, "Batch size: "+str(batch_size), "Learning rate: "+str(learning_rate), "Epochs: "+str(epochs), "Load model: "+str(load_model), "Save model: "+str(save_model), sep="\n")
    # Use GPU if available else use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # Load data
    encoder_name = is_encoder(model_name)
    if(encoder_name!=False):
        train_dataloader, test_dataloader = data.get_data_loader_encoder(data_name, encoder_name, batch_size, device)
    else:
        train_dataloader, test_dataloader = data.get_data_loader(data_name, batch_size, device)

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
        model = get_model(model_name, data_name, encoder_name)
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
    results.update_results(data_name, model_name, epochs, batch_size, learning_rate, train_losses, test_losses, train_accs, test_accs)

    # Save model if save_name is given
    if(save_model==True):
        # Check if saved_models folder exists
        if not os.path.exists("saved_models"):
            os.makedirs("saved_models")
        # Save model
        torch.save(model, "saved_models/"+model_name+".pth")
        print("Saved PyTorch Model State to saved_models/"+model_name+".pth")
    return data_name, model_name

# Main function
def main():
    # Run experiments
    data_name, model_name = run_exp(
        data_name="CIFAR100", 
        model_name="FC_FF_NN",
        batch_size=64, 
        learning_rate=0.01, 
        epochs=100, 
        load_model=False,
        save_model=True
        )
    plot.plot_default(data_name, model_name)

    data_name, model_name = run_exp(
        data_name="CIFAR100", 
        model_name="CNN",
        batch_size=64, 
        learning_rate=0.008, 
        epochs=100, 
        load_model=False,
        save_model=True
        )
    plot.plot_default(data_name, model_name)

    data_name, model_name = run_exp(
        data_name="CIFAR100", 
        model_name="RN50_clip_FC_FF_NN",
        batch_size=64, 
        learning_rate=0.01, 
        epochs=100, 
        load_model=False,
        save_model=True
        )
    plot.plot_default(data_name, model_name)

main()