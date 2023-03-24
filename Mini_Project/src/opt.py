import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

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

        # Print loss every 100 batches
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
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
        for X, y, *ignore in dataloader:
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
    print(f"Test Error: \n Accuracy: {(test_acc):>0.1f}%, Avg loss: {test_loss:>8f} \n")  
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

    # Perfrom multiple epochs of training
    starting_epoch, end_epoch = epochs
    for t in range(starting_epoch, end_epoch):
        # Print epoch number
        print(f"Epoch {t+1}\n-------------------------------")

        # Perform training and save loss and accuracy
        train_loss, train_acc = train_loop(train_dataloader, model, loss_fn, optimizer)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Perform testing and save loss and accuracy
        test_loss, test_acc = test_loop(test_dataloader, model, loss_fn)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
    print("Done!")
    return train_losses, test_losses, train_accs, test_accs, optimizer
