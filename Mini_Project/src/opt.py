import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Performs one epoch of training
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Performs testing
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")  

# Function that trains neural network over multiple epochs
def train(model, train_dataloader, test_dataloader, 
            learning_rate = 1e-3, epochs = 5, batch_size = 64, 
            loss_fn=nn.CrossEntropyLoss(), optimizer=None, verbose=True):
    # If no optimizer is given, use SGD 
    if(optimizer is None):
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Perfrom multiple epochs of training
    for t in range(epochs):
        if(verbose==True):
            print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")

#TODO: keep track of loss and accuracy over time
#TODO: add plotting of loss and accuracy over time