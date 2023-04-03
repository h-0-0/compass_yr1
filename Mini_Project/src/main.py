import os
import torch
import data
import simple
import opt
import results
import plot
from argparse import ArgumentParser

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
    if(load_model == True):
        # Check if saved_models folder exists
        if not os.path.exists("saved_models"):
            raise Exception("No saved_models folder found, cannot load model")
        # Load model
        model = get_model(model_name, data_name, encoder_name)
        checkpoint = torch.load("saved_models/"+ data_name+ "_"+model_name+".pth")
        model.load_state_dict(checkpoint['model_state_dict'])

        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        prev_epoch = checkpoint['epoch']
        if(prev_epoch > epochs):
            raise Exception("Cannot load model from epoch: "+str(prev_epoch)+" and train for a total of: "+str(epochs)+" epochs")

        handle_device(model, device)
        print("Loaded PyTorch Model State from saved_models/"+ data_name + "_" + model_name+".pth")
    elif(load_model == False):
        model = get_model(model_name, data_name, encoder_name)
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        handle_device(model, device)
        prev_epoch = 0
    else:
        raise Exception("load_model must be a boolean, got: "+str(load_model)+" of type: "+str(type(load_model)))

    # Train model
    train_losses, test_losses, train_accs, test_accs, times, optimizer = opt.train(
        model, 
        train_dataloader, 
        test_dataloader, 
        optimizer,
        learning_rate=learning_rate, 
        epochs=(prev_epoch, epochs)
    )
    
    # We add the results to the results dataframe
    results.update_results(data_name, model_name, (prev_epoch, epochs), batch_size, learning_rate, train_losses, test_losses, train_accs, test_accs, times)

    # Save model if save_name is given
    if(save_model==True):
        # Check if saved_models folder exists
        if not os.path.exists("saved_models"):
            os.makedirs("saved_models")
        # Save model
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, 
            "saved_models/"+data_name+ "_"+ model_name+".pth"
            )
        print("Saved PyTorch Model State to saved_models/"+data_name+ "_"+model_name+".pth")
    return data_name, model_name

# Run an experiment in CL scenario
def run_exp_CL(data_name="MNIST", model_name="RN50_clip_FF_FC_NN", n_tasks=5, init_inc=2, batch_size=64, learning_rate=1e-3, epochs=5, load_model=False, save_model=False):
    print("\n \n \n---------------------------- New Experiment ----------------------------")
    print("Data: "+data_name, "Model: "+model_name, "Number of tasks: "+str(n_tasks), "Number of classes in first task: " + str(init_inc), "Batch size: "+str(batch_size), "Learning rate: "+str(learning_rate), "Epochs: "+str(epochs), "Load model: "+str(load_model), "Save model: "+str(save_model), sep="\n")

    # Use GPU if available else use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # Load data
    encoder_name = is_encoder(model_name)
    if(encoder_name!=False):
        train_scenario, test_scenario = data.get_data_loader_encoder_CL(data_name, encoder_name, batch_size, device, n_tasks)
    else:
        train_scenario, test_scenario = data.get_data_loader_CL(data_name, batch_size, device, n_tasks, init_inc)

    # Create model or load model
    if(load_model == True):
        # Check if saved_models folder exists
        if not os.path.exists("saved_models"):
            raise Exception("No saved_models folder found, cannot load model")
        # Load model
        model = get_model(model_name, data_name, encoder_name)
        checkpoint = torch.load("saved_models/"+ data_name+ "_" +model_name+"_CL"+".pth")
        model.load_state_dict(checkpoint['model_state_dict'])

        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        handle_device(model, device)
        print("Loaded PyTorch Model State from saved_models/" + data_name+ "_"+model_name+"_CL"+".pth")
    elif(load_model == False):
        model = get_model(model_name, data_name, encoder_name)
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        handle_device(model, device)
    else:
        raise Exception("load_model must be a boolean, got: "+str(load_model)+" of type: "+str(type(load_model)))

    # Train model
    train_losses, test_losses, train_accs, test_accs, times, optimizer = opt.train_CL(
        model, 
        train_scenario,
        test_scenario, 
        optimizer,
        learning_rate=learning_rate, 
        epochs=epochs
    )
    task_names = list(range(len(train_scenario)))
    # We add the results to the results dataframe
    results.update_results(data_name, model_name, epochs, batch_size, learning_rate, train_losses, test_losses, train_accs, test_accs, times, task_names, init_inc)

    # Save model if save_name is given
    if(save_model==True):
        # Check if saved_models folder exists
        if not os.path.exists("saved_models"):
            os.makedirs("saved_models")
        # Save model
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, 
            "saved_models/"+data_name+ "_" + model_name+ "_CL" + ".pth"
            )
        print("Saved PyTorch Model State to saved_models/" +data_name+ "_" +model_name + "_CL" + ".pth")
    return data_name, model_name

# Main function
def main(args, is_CL):
    if (is_CL == False):
        data_name, model_name = run_exp(
            data_name=args.data_name, 
            model_name=args.model_name,
            batch_size=args.batch_size, 
            learning_rate=args.learning_rate,
            epochs=args.epochs, 
            load_model=args.load_model,
            save_model=args.save_model
            )
        plot.plot_default(data_name, model_name, is_CL=False)
    else:
        data_name, model_name = run_exp_CL(
            data_name=args.data_name, 
            model_name=args.model_name,
            n_tasks=args.n_tasks,
            init_inc=args.init_inc,
            batch_size=args.batch_size, 
            learning_rate=args.learning_rate,
            epochs=args.epochs, 
            load_model=args.load_model,
            save_model=args.save_model
            )
        plot.plot_default(data_name, model_name, is_CL=True)


if __name__ == "__main__":
    # Create argument parser
    parser = ArgumentParser()

    # Arguments related to experiment
    parser.add_argument("--data_name", type=str, help="Name of dataset to use", default="MNIST")
    parser.add_argument("--model_name", type=str, help="Name of model to use", default="FC_FF_NN")
    parser.add_argument("--batch_size", type=int, help="Batch size", default=64)
    parser.add_argument("--learning_rate", type=float, help="Learning rate", default=1e-3)
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=5)

    # Argument for loading model
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--no-load_model', dest='load_model', action='store_false')
    parser.set_defaults(load_model=True)

    # Argument for saving model
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--no-save_model', dest='save_model', action='store_false')
    parser.set_defaults(save_model=True)

    # Arguments related to CL, if n-tasks given will run CL experiment
    parser.add_argument("--n_tasks", type=int, help="Number of tasks", default=-1)
    parser.add_argument("--init_inc", type=int, help="Number of classes for first task", default=2)

    args = parser.parse_args()
    # TRAIN
    is_CL = (args.n_tasks != -1)
    main(args, is_CL)
