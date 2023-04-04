import os
import torch
import data
import simple
import opt
import results
import plot
from argparse import ArgumentParser

# Given model_name detects what model to create and returns said model
def detect_model(model_name, data_name, encoder_name):
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

# Used to create model, optimizer and prev_epoch, if load_model is True then load model from saved_models folder
def get_model(load_model, model_name, data_name, encoder_name, learning_rate, epochs, device, is_CL=False):
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
        model = detect_model(model_name, data_name, encoder_name)
        checkpoint = torch.load("saved_models/"+ data_name+ "_"+model_name+CL_ext+".pth", map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])

        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if is_CL:
            prev_epoch = False
        else:
            prev_epoch = checkpoint['epoch']
            if(prev_epoch > epochs):
                raise Exception("Cannot load model from epoch: "+str(prev_epoch)+" and train for a total of: "+str(epochs)+" epochs")

        handle_device(model, device)
        print("Loaded PyTorch Model State from saved_models/"+ data_name + "_" + model_name+ CL_ext+".pth")
    elif(load_model == False):
        model = detect_model(model_name, data_name, encoder_name)
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        handle_device(model, device)
        if is_CL:
            prev_epoch = False
        else:
            prev_epoch = 0
    else:
        raise Exception("load_model must be a boolean, got: "+str(load_model)+" of type: "+str(type(load_model)))
    return model, optimizer, prev_epoch

# Used to save models
def save(model, optimizer, data_name, model_name, epochs, is_CL=False):
    if is_CL:
        CL_ext = "_CL"
    else:
        CL_ext = ""
    
    # Check if saved_models folder exists
    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")
    
    # Save model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, 
        "saved_models/"+data_name+ "_"+ model_name+ CL_ext + ".pth"
        )
    print("Saved PyTorch Model State to saved_models/"+data_name+ "_"+model_name+ CL_ext +".pth")

# Run an experiment
def run_exp(data_name="MNIST", model_name="RN50_clip_FF_FC_NN", batch_size=64, learning_rate=1e-3, epochs=5, load_model=False, save_model=False, device=False):
    # Use GPU if available else use CPU, if device is given use that device
    if device == False:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using {} device".format(device))
    else:
        print("Using {} device".format(device))

    # Print experiment info
    print("\n \n \n---------------------------- New Experiment ----------------------------")
    print("Data: "+data_name, "Model: "+model_name, "Batch size: "+str(batch_size), "Learning rate: "+str(learning_rate), "Epochs: "+str(epochs), "Load model: "+str(load_model), "Save model: "+str(save_model), sep="\n")

    # Load data
    encoder_name = is_encoder(model_name)
    if(encoder_name!=False):
        train_dataloader, test_dataloader = data.get_data_loader_encoder(data_name, encoder_name, batch_size, device)
    else:
        train_dataloader, test_dataloader = data.get_data_loader(data_name, batch_size, device)

    # Create model
    model, optimizer, prev_epoch = get_model(load_model, model_name, data_name, encoder_name, learning_rate, epochs, device, is_CL=False)

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
        save(model, optimizer, data_name, model_name, epochs, is_CL=False)
    
    return data_name, model_name

# Run an experiment in CL scenario
def run_exp_CL(data_name="MNIST", model_name="RN50_clip_FF_FC_NN", n_tasks=5, init_inc=2, batch_size=64, learning_rate=1e-3, epochs=5, load_model=False, save_model=False, device=False):
    # Use GPU if available else use CPU, if device is given use that device
    if device == False:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using {} device".format(device))
    else:
        print("Using {} device".format(device))
    
    # Print experiment info
    print("\n \n \n---------------------------- New Experiment ----------------------------")
    print("Data: "+data_name, "Model: "+model_name, "Number of tasks: "+str(n_tasks), "Number of classes in first task: " + str(init_inc), "Batch size: "+str(batch_size), "Learning rate: "+str(learning_rate), "Epochs: "+str(epochs), "Load model: "+str(load_model), "Save model: "+str(save_model), sep="\n")
    
    # Load data
    encoder_name = is_encoder(model_name)
    if(encoder_name!=False):
        train_scenario, test_scenario = data.get_data_loader_encoder_CL(data_name, encoder_name, batch_size, device, n_tasks)
    else:
        train_scenario, test_scenario = data.get_data_loader_CL(data_name, batch_size, device, n_tasks, init_inc)

    # Create model or load model
    model, optimizer, _ = get_model(load_model, model_name, data_name, encoder_name, learning_rate, epochs, device, is_CL=True)

    # Train model
    metrics, optimizer = opt.train_CL(
        model, 
        train_scenario,
        test_scenario, 
        optimizer,
        data_name,
        model_name,
        learning_rate=learning_rate, 
        epochs=epochs
    )

    # Create task names (Currently is just [0, ..., n_tasks-1])
    task_names = list(range(len(train_scenario)))

    # We add the results to the results dataframe
    results.update_results_CL(data_name, model_name, epochs, batch_size, learning_rate, task_names, init_inc, metrics)

    # Save model if save_name is given
    if(save_model==True):
        save(model, optimizer, data_name, model_name, epochs, is_CL=True)
    
    return data_name, model_name

# Main function
def main(args, is_CL):
    # Runs a regular experiment or CL experiment depending on value of is_CL
    if (is_CL == False):
        data_name, model_name = run_exp(
            data_name=args.data_name, 
            model_name=args.model_name,
            batch_size=args.batch_size, 
            learning_rate=args.learning_rate,
            epochs=args.epochs, 
            load_model=args.load_model,
            save_model=args.save_model,
            device=args.device
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
            save_model=args.save_model,
            device=args.device
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

    # Argument for overriding which device to use
    parser.add_argument("--device", type=str, help="Which device to use", default=False)

    # Parse arguments and check if CL experiment
    args = parser.parse_args()
    is_CL = (args.n_tasks != -1)

    # Run experiment
    main(args, is_CL)
