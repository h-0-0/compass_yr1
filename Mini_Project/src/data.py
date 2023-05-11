from torch.utils.data import DataLoader, Dataset
from torch import from_numpy
import numpy as np
from latent_CL.dataset_encoder import prepare_scenarios
from latent_CL.args import ArgsGenerator
from latent_CL.Models.model import ModelContainer
from continuum import ClassIncremental
from continuum.datasets import MNIST, CIFAR10, CIFAR100, H5Dataset
import h5py
from pathlib import Path

class H5Dataset_Batch(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path

        # Open the HDF5 file
        self.h5file = h5py.File(self.file_path, "r")

        # Get the number of samples in the dataset
        self.num_samples = len(self.h5file["data"])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # get data
        x = self.h5file["data"][index]
        x = from_numpy(x)

        # get label
        y =  self.h5file["targets"][index][0].astype(np.int_)
        return (x, y)

    def close(self):
        self.h5file.close()

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
def get_data_loader_CL(name, batch_size, device, n_tasks, init_inc=2):
    # Get training and testing data
    training_data, test_data = get_data(name, device)

    # Calculate the number of classes per task
    _, y, _ = training_data.get_data()
    n_classes = len(np.unique(y))
    increment = (n_classes - init_inc)/(n_tasks-1)
    if(increment != int(increment)):
        raise Exception("Number of classes per task not evenly divisible for given number of tasks and initial increment")
    else:
        increment = int(increment)
    
    # Create CL scenarios
    train_scenario = ClassIncremental(
        training_data,
        increment=increment,
        initial_increment=init_inc
    )

    test_scenario = ClassIncremental(
        test_data,
        increment=increment,
        initial_increment=init_inc
    )

    # We return our newly created CL scenarios
    return train_scenario, test_scenario

def get_data_loader_encoder(data_name, encoder_name, batch_size, device):
    if encoder_name in ["fVGG", "fAE"]:
        if data_name == "CIFAR10":
            encoder_training_data = "CIFAR100"
        else:
            encoder_training_data = data_name
        # Create dataloaders from encoded data from hdf5 file
        train_dataloader = DataLoader(H5Dataset_Batch("./data/EncodedDatasets/"+data_name+ "_by_"+ encoder_training_data+"_"+encoder_name + "_train.hdf5"), batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(H5Dataset_Batch("./data/EncodedDatasets/"+data_name+ "_by_"+ encoder_training_data+"_"+encoder_name + "_test.hdf5"), batch_size=batch_size, shuffle=True)
        return train_dataloader, test_dataloader
    else:
        if(encoder_name == "fResnet18"):
            encoder_name = "resnet18"
        if(encoder_name == "fResnet50"):
            encoder_name = "resnet50"
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
        train_data = scenario[0]
        test_data= scenario_test[0]
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        return train_dataloader, test_dataloader

def get_data_loader_encoder_CL(data_name, encoder_name, batch_size, device, n_tasks, init_inc=2):
    if encoder_name in ["fVGG", "fAE"]:
        if data_name == "CIFAR10":
            encoder_training_data = "CIFAR100"
        else:
            encoder_training_data = data_name
        # Load in encoded data from hdf5 file
        train_set = H5Dataset(x=None, y=None, t=None, data_path="./data/EncodedDatasets/"+data_name+ "_by_"+ encoder_training_data+"_"+encoder_name + "_train_CL.hdf5")
        test_set = H5Dataset(x=None, y=None, t=None, data_path="./data/EncodedDatasets/"+data_name+ "_by_"+ encoder_training_data+"_"+encoder_name + "_test_CL.hdf5")
        # Get info for creating CL scenarios
        _, y, _ = train_set.get_data()
        n_classes = len(np.unique(y))
        increment = (n_classes - init_inc)/(n_tasks-1)
        if(increment != int(increment)):
            raise Exception("Number of classes per task not evenly divisible for given number of tasks and initial increment")
        else:
            increment = int(increment)
        # Create CL scenarios
        train_scenario = ClassIncremental(
            train_set,
            increment=increment,
            initial_increment=init_inc
        )
        test_scenario = ClassIncremental(
            test_set,
            increment=increment,
            initial_increment=init_inc
        )
        return train_scenario, test_scenario
    # TODO: above if staement is not working
    else:
        if(encoder_name == "fResnet18"):
            encoder_name = "resnet18"
        if(encoder_name == "fResnet50"):
            encoder_name = "resnet50"

        # We generate arguments for retrieving and loading the encoded data (and runs data through encoder if not already encoded)
        args_generator = ArgsGenerator(dataset_name=data_name, 
                                        dataset_encoder_name=encoder_name,
                                        permute_task_order = False ,
                                        n_classes = None, 
                                        n_tasks = n_tasks, 
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