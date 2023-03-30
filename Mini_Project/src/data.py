import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from latent_CL.Utils.utils import set_seed  
from latent_CL.dataset_encoder import prepare_scenarios
from latent_CL.args import ArgsGenerator
from latent_CL.Models.model import ModelContainer
from continuum import ClassIncremental
from continuum.datasets import MNIST, CIFAR10, CIFAR100

# Function to get training and testing data
# downloads and stores data in folder named "data" if not already there
def get_data(name, device):
    if(name == "MNIST"):
        training_data = MNIST(
            "data", 
            download=True, 
            train=True
        ),

        test_data = MNIST(
            "data", 
            download=True, 
            train=False
        )
    elif(name == "CIFAR100"):
        training_data = CIFAR100(
            "data", 
            download=True, 
            train=True
        ),

        test_data = CIFAR100(
            "data", 
            download=True, 
            train=False
        )
    elif(name == "CIFAR10"):
        training_data = CIFAR10(
            "data", 
            download=True, 
            train=True
        ),

        test_data = CIFAR10(
            "data", 
            download=True, 
            train=False
        )
    else:
        raise Exception("Not given valid dataset name must be: MNIST, CIFAR10 or CIFAR100")
    
    return training_data[0], test_data

# Function to retrieve data and create data loaders
def get_data_loader(name, batch_size, device):
    # Get training and testing data
    training_data, test_data = get_data(name, device)

    # turn data into batch scenario and transform to tensor
    training_taskset = training_data.to_taskset() 
    test_taskset = test_data.to_taskset()

    # Create data loaders
    train_dataloader = DataLoader(training_taskset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_taskset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader

# Function to retrieve data and create data loaders in continuous learning setting which are called scenarios
def get_data_loader_CL(name, batch_size, device, n_tasks, init_n_tasks=2):
    # Get training and testing data
    training_data, test_data = get_data(name, device)

    # Create CL scenarios
    train_scenario = ClassIncremental(
        training_data,
        increment=n_tasks,
        initial_increment=init_n_tasks
    )

    test_scenario = ClassIncremental(
        test_data,
        increment=n_tasks,
        initial_increment=init_n_tasks
    )

    # We return our newly created CL scenarios
    return train_scenario, test_scenario

def get_data_loader_encoder(data_name, encoder_name, batch_size, device):
    # We generate arguments for retrieving and loading the encoded data (and runs data through encoder if not already encoded)
    args_generator = ArgsGenerator(dataset_name=data_name, 
                                    dataset_encoder_name=encoder_name,
                                    permute_task_order = False ,
                                    n_classes = None, 
                                    n_tasks = 1, 
                                    epochs = 10, 
                                    batch_size = batch_size, 
                                    encoding_batch_size = batch_size, 
                                    encode_with_continuum = True, 
                                    device = device,

                                    estimate_compute_regime = False, 
                                    estimate_time = False, 
                                    estimate_compute_regime_encoding = False, 

                                    regime ='latent_ER',

                                    data_path = "./data", 
                                    weights_path = None, 
                                )
    args_model = ModelContainer.Options()

    # We now use prepare_scenarios (from latent_CL) to retrieve the encoded data and then create data loaders
    scenario, scenario_test = prepare_scenarios(args_generator, args_model) 
    train_taskset = scenario[0]
    train_dataloader = DataLoader(train_taskset, batch_size=batch_size, shuffle=True)
    test_taskset = scenario_test[0]
    test_dataloader = DataLoader(test_taskset, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader

def get_data_loader_encoder_CL(data_name, encoder_name, batch_size, device, n_tasks):
    # We generate arguments for retrieving and loading the encoded data (and runs data through encoder if not already encoded)
    args_generator = ArgsGenerator(dataset_name=data_name, 
                                    dataset_encoder_name=encoder_name,
                                    permute_task_order = False ,
                                    n_classes = None, 
                                    n_tasks = 1, 
                                    k_shot = None, #number of shots per class in each new task;
                                    epochs = 10, 
                                    batch_size = batch_size, 
                                    encoding_batch_size = batch_size, 
                                    encode_with_continuum = True, 
                                    device = device,

                                    estimate_compute_regime = False, 
                                    estimate_time = False, 
                                    estimate_compute_regime_encoding = False, 

                                    regime ='latent_ER',

                                    data_path = "./data", 
                                    weights_path = None, 
                                )
    args_model = ModelContainer.Options()

    # We now use prepare_scenarios (from latent_CL) to retrieve the encoded data and create a training and testing CL scenario
    train_scenario, test_scenario = prepare_scenarios(args_generator, args_model) 
    # We return our newly created CL scenarios
    return train_scenario, test_scenario
