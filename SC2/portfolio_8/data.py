from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# Function to get training and testing data
# downloads and stores data in folder named "data" if not already there
def get_data(name, device):
    if(name == "MNIST"):
        training_data = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor()
        )

        test_data = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor()
        )
    elif(name == "CIFAR100"):
        training_data = datasets.CIFAR100(
            root="data",
            train=True,
            download=True,
            transform=ToTensor()
        )
        test_data = datasets.CIFAR100(
            root="data",
            train=False,
            download=True,
            transform=ToTensor()
        )
    elif(name == "CIFAR10"):
        training_data = datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=ToTensor()
        )
        test_data = datasets.CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=ToTensor()
        )   
    else:
        raise Exception("Not given valid dataset name must be: MNIST, CIFAR10 or CIFAR100")
    
    return training_data, test_data


# Function to retrieve data and create data loaders
def get_data_loader(name, batch_size, device, num_workers=0):
    # Get training and testing data
    training_data, test_data = get_data(name, device)

    # Create data loaders
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_dataloader, test_dataloader