import torch
from torch import nn
from torch.utils.data import DataLoader
import time
import os

def print_flush(*args, **kwargs):
    print(*args, **kwargs, flush=True)

# Used to save models at checkpoints
def save_checkpoint(model, optimizer, data_name, model_name, checkpoint_id, epochs=False , is_CL=True):
    if is_CL:
        if epochs != False:
            raise ValueError("epochs should be False when is_CL is True")
        CL_ext = "_CL"
    else:
        if epochs == False:
            raise ValueError("epochs should not be False when is_CL is False")
        CL_ext = ""

    path = data_name+ "_"+model_name+ CL_ext
    
    # Check if saved_models folder exists
    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")
    # Check if folder for model exists
    if not os.path.exists("saved_models/" + path):
        os.makedirs("saved_models/" + path)
    
    # Save model
    if is_CL:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, 
            "saved_models/" + path + "/" + path + "_" + checkpoint_id + ".pth"
            )
    else:
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, 
            "saved_models/" + path + "/" + path + "_" + checkpoint_id + ".pth"
            )
    print_flush("Saved PyTorch Model State to saved_models/" + path + "/" + path + "_" + checkpoint_id +".pth")

# Performs one epoch of training
def train_loop(dataloader, model, loss_fn, optimizer):
    # Check if model is in training mode and set to training mode if not
    if(model.training==False):
        model.train()
    
    # Get size of dataset, keep track of loss and accuracy and start training loop
    size = len(dataloader.dataset)
    train_loss, correct = 0, 0
    for batch, (X, y, *ignore) in enumerate(dataloader):
        # Move data to device that model is on
        X  = X.to(model.device)
        y = y.to(model.device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Keep track of loss and accuracy
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        # print_flush loss every 100 batches
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print_flush(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # Find number of batches and calculate average loss and accuracy
    num_batches = len(dataloader)
    train_loss /= num_batches
    correct /= size
    train_acc = 100*correct
    return train_loss, train_acc

# Performs testing
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for (X, y, *ignore) in dataloader:
            # Move data to device that model is on
            X  = X.to(model.device)
            y = y.to(model.device)
            # Compute prediction and loss
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # Keep track of number of correct predictions
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    # Calculate average loss and accuracy
    test_loss /= num_batches
    correct /= size
    test_acc = 100*correct
    print_flush(f"Test Error: \n Accuracy: {(test_acc):>0.1f}%, Avg loss: {test_loss:>8f} \n")  
    return test_loss, test_acc

# Calculates accuracy and loss for past data
def past_data(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for (X, y, t) in dataloader:
            # Move data to device that model is on
            X  = X.to(model.device)
            y = y.to(model.device)
            # Compute prediction and loss
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # Keep track of number of correct predictions
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    # Calculate average loss and accuracy
    test_loss /= num_batches
    correct /= size
    test_acc = 100*correct
    return test_loss, test_acc
            

# Function that trains neural network over multiple epochs
def train(model, train_dataloader, test_dataloader, optimizer,
        learning_rate = 1e-3, epochs = (0,5), 
        loss_fn=nn.CrossEntropyLoss()
        ):
    
    # Initialize arrays to keep track of losses and accuracy's 
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    times = []

    # Perform multiple epochs of training
    starting_epoch, end_epoch = epochs
    training_start_time = time.time()
    for t in range(starting_epoch, end_epoch):
        # print_flush epoch number
        print_flush(f"Epoch {t+1}\n-------------------------------")

        # Perform training and save loss and accuracy
        train_loss, train_acc = train_loop(train_dataloader, model, loss_fn, optimizer)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Save time
        times.append(time.time() - training_start_time)

        # Perform testing and save loss and accuracy
        test_loss, test_acc = test_loop(test_dataloader, model, loss_fn)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
    print_flush("Done!")
    return train_losses, test_losses, train_accs, test_accs, times, optimizer

# Function that trains neural network over multiple epochs in CL scenario, note that if SGD with momentum is used then the momentum is reset after each task
def train_CL(model, train_scenario, test_scenario, optimizer, data_name, model_name, 
        learning_rate = 1e-3, epochs = 5, 
        loss_fn=nn.CrossEntropyLoss(),
        reset_fc = False 
        ):
    # Collect past data-loaders for calculating metrics
    past_train_dataloader = {}
    past_test_dataloader = {}

    # Initialize arrays/dictionaries to keep track of various metrics
    # Keeps track of time
    times = []
    # Keeps track of losses and accuracies for task currently being trained on
    train_CTLs = []
    test_CTLs = []
    train_CTAs = []
    test_CTAs = []
    # Keeps track of losses and accuracies for each individual task, starts keeping track of a task once we've trained on it once
    train_TLs = {}
    train_TAs = {}
    test_TLs = {}
    test_TAs = {}
    # Keeps track of losses and accuracies for all tasks trained on so far, ie. average loss and accuracy over all tasks trained on so far
    train_PTLs = []
    train_PTAs = []
    test_PTLs = []
    test_PTAs = []
    # Keeps track of losses and accuracies on each specific task over training  
    ids = [i for i in range(len(train_scenario))]
    train_TLs = {i: [] for i in ids}
    train_TAs = {i: [] for i in ids}
    test_TLs = {i: [] for i in ids}
    test_TAs = {i: [] for i in ids}

    # Record starting time
    training_start_time = time.time()

    # Extract all data loaders for each task
    all_train_dataloader = {id : DataLoader(i) for id, i in enumerate(train_scenario)}
    all_test_dataloader = {id : DataLoader(i) for id, i in enumerate(test_scenario)}

    # Perform multiple epochs of training
    first = True
    for task_id, (train_dataset, test_dataset) in enumerate(zip(train_scenario, test_scenario)):
        # print_flush task number
        print_flush(f"Task {task_id}\n-------------------------------")
        # Create data-loaders for current task
        train_dataloader = DataLoader(train_dataset)
        test_dataloader = DataLoader(test_dataset)
        # Reset optimizer if SGD with momentum is used
        if (type (optimizer).__name__ == 'SGD') and (optimizer.defaults['momentum'] != 0) and (not first):
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        if (type (optimizer).__name__ == 'Adam') and (not first):
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # TODO: remove below once finished investigating
        if reset_fc == True:
            print_flush("----------Resetting parameters----------")
            for layer in model.fc:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        # Perform multiple epochs of training and testing
        for _ in range(epochs):
            # Perform training and save loss and accuracy on current task
            train_CTL, train_CTA = train_loop(train_dataloader, model, loss_fn, optimizer)
            train_CTLs.append(train_CTL)
            train_CTAs.append(train_CTA)
            train_TLs[task_id].append(train_CTL)
            train_TAs[task_id].append(train_CTA)

            # Calculate loss and accuracy for past tasks, both task specific (TL/TA) and average across all tasks (PTL/PTA) for training data
            train_PTL = train_CTL
            train_PTA = train_CTA
            if first == False:
                # Calculate loss and accuracy for past tasks, both task specific (TL/TA) and average across all tasks (PTL/PTA) for testing data
                for t, dl in past_train_dataloader.items():
                    train_TL, train_TA = past_data(dl, model, loss_fn)
                    train_TLs[t].append(train_TL)
                    train_TAs[t].append(train_TA)
                    train_PTL += train_TL
                    train_PTA += train_TA
                train_PTLs.append(train_PTL/len(past_train_dataloader))
                train_PTAs.append(train_PTA/len(past_train_dataloader))
            else:
                train_PTLs.append(train_PTL)
                train_PTAs.append(train_PTA)
            # Also calculate average loss and accuracy across tasks not yet seen
            for t, dl in all_train_dataloader.items():
                if (t not in past_train_dataloader) and (t != task_id):
                    train_TL, train_TA = past_data(dl, model, loss_fn)
                    train_TLs[t].append(train_TL)
                    train_TAs[t].append(train_TA)

            # Save time
            times.append(time.time() - training_start_time)

            # Perform testing and save loss and accuracy on current task
            test_CTL, test_CTA = test_loop(test_dataloader, model, loss_fn)
            test_CTLs.append(test_CTL)
            test_CTAs.append(test_CTA)
            test_TLs[task_id].append(test_CTL)
            test_TAs[task_id].append(test_CTA)

            # Calculate loss and accuracy for past tasks, both task specific (TL/TA) and average across all tasks (PTL/PTA) for testing data
            test_PTL = test_CTL
            test_PTA = test_CTA
            if first == False:
                for t, dl in past_test_dataloader.items():
                    test_TL, test_TA = past_data(dl, model, loss_fn)
                    test_TLs[t].append(test_TL)
                    test_TAs[t].append(test_TA)
                    test_PTL += test_TL
                    test_PTA += test_TA
                test_PTLs.append(test_PTL/len(past_test_dataloader))
                test_PTAs.append(test_PTA/len(past_test_dataloader))
            else:
                test_PTLs.append(test_PTL)
                test_PTAs.append(test_PTA)
            # Also calculate average loss and accuracy across tasks not yet seen
            for t, dl in all_test_dataloader.items():
                if (t not in past_test_dataloader) and (t != task_id):
                    test_TL, test_TA = past_data(dl, model, loss_fn)
                    test_TLs[t].append(test_TL)
                    test_TAs[t].append(test_TA)
        
        # Save checkpoint of model
        save_checkpoint(model, optimizer, data_name, model_name, "task_"+str(task_id), is_CL=True)
        # Save data-loaders for current task
        past_train_dataloader[task_id] = train_dataloader
        past_test_dataloader[task_id] = test_dataloader
        first = False

    print_flush("Done!")
    # Store metrics in dictionary
    metrics = {
        "Train CTL": train_CTLs,
        "Test CTL": test_CTLs,
        "Train CTA": train_CTAs,
        "Test CTA": test_CTAs,
        "Train PTL": train_PTLs,
        "Train PTA": train_PTAs,
        "Test PTL": test_PTLs,
        "Test PTA": test_PTAs,
        "Times": times
    }
    # Let's now add the TL and TA metrics we have collected for each task to the metrics dictionary
    for t in train_TLs.keys():
        # we format the TL and TA metrics we have stored so that they are the same length as the other metrics
        n_pad = len(train_CTLs)-len(train_TLs[t])
        metrics["Train TL task "+str(t)] = [None]*n_pad + train_TLs[t]
        metrics["Train TA task "+str(t)] = [None]*n_pad + train_TAs[t]
        metrics["Test TL task "+str(t)] = [None]*n_pad + test_TLs[t]
        metrics["Test TA task "+str(t)] = [None]*n_pad + test_TAs[t]
    return metrics, optimizer
