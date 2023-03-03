import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from latent_CL.Utils.utils import set_seed  
from latent_CL.dataset_encoder import prepare_scenarios
from latent_CL.args import ArgsGenerator
from latent_CL.Models.model import ModelContainer
from continuum.generators import ClassOrderGenerator

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
    else:
        raise Exception("Not given valid dataset name must be: MNIST")
    
    return training_data, test_data

# Function to retrieve data and create data loaders
def get_data_loader(name, batch_size, device):
    # Get training and testing data
    training_data, test_data = get_data(name, device)

    # Create data loaders
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader


def get_data_loader_encoder(name, batch_size, device):
    # We generate arguments for retrieving and loading the encoded data (and runs data through encoder if not already encoded)
    args_generator = ArgsGenerator(dataset_name=name, 
                                    dataset_encoder_name='RN50_clip',
                                    permute_task_order = False ,
                                    n_classes = None, 
                                    n_tasks = 2, 
                                    epochs = 100, 
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

    # This returns a continual learning scenario for the training and testing data 
    scenario, scenario_test = prepare_scenarios(args_generator, args_model) 
    # print(f"scenario: {scenario}, scenario_test: {scenario_test}")

    # We create a generator that will 
    scenario_generator = ClassOrderGenerator(scenario)
    for task_id, (train_taskset, test_taskset) in enumerate(zip(scenario, scenario_test)):
        train_dataloader = DataLoader(train_taskset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_taskset, batch_size=batch_size, shuffle=True)
        # x,y,t = train_taskset[0]
        # print(f"task {task_id}, x in {x.shape}, y {y}")  
    return train_dataloader, test_dataloader

#TODO: create two versions of encoded data loader: one for continual scenario and one for full data