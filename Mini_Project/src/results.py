import os
import pandas as pd
from numpy import repeat as np_repeat

# Function that creates data frame to store results
def create_results_df(is_CL: bool=False):
    if(is_CL==False):
        # Create results dataframe
        results_df = pd.DataFrame(
            columns=[
                "Model Name",
                "Epoch",
                "Batch Size",
                "Learning Rate",
                "Train Loss",
                "Test Loss",
                "Train Accuracy",
                "Test Accuracy",
            ]
        )
    else:
        # Create results dataframe
        results_df = pd.DataFrame(
            columns=[
                "Model Name",
                "Epoch",
                "Batch Size",
                "Learning Rate",
                "Train Loss",
                "Test Loss",
                "Train Accuracy",
                "Test Accuracy",
                "Task Name",
                "Initial Number of Tasks"
            ]
        )
    return results_df

# Function to save results to csv file
def save_results(results_df, data_name, CL_ext=""):
    # Check if results folder exists
    if not os.path.exists("results"):
        os.makedirs("results")
    # Save results
    results_df.to_csv("results/"+data_name+ CL_ext +".csv", index=False, mode='w+')
    print("Saved results to results/"+data_name+ CL_ext +".csv")

# Function to load results from csv file
def load_results(data_name, CL_ext=""):
    # Check if results folder exists
    if not os.path.exists("results"):
        raise Exception("No results folder found, cannot load results")
    # Load results
    results_df = pd.read_csv("results/"+data_name+ CL_ext +".csv")
    print("Loaded results from results/"+data_name+ CL_ext +".csv")
    return results_df

# Function to add to results dataframe
def add_to_results(results_df, model_name, epochs, batch_size, learning_rate, train_losses, test_losses, train_accs, test_accs, task_names=False, init_n_tasks=False):
    if(task_names==False):
        # Add results to dictionary
        new_results = {
            "Model Name": [model_name]*(epochs[1]- epochs[0]),
            "Epoch": [i for i in range(epochs[0], epochs[1])],
            "Batch Size": [batch_size]*(epochs[1]- epochs[0]),
            "Learning Rate": [learning_rate]*(epochs[1]- epochs[0]),
            "Train Loss": train_losses,
            "Test Loss": test_losses,
            "Train Accuracy": train_accs,
            "Test Accuracy": test_accs,
        }
    else:
        # Add results to dictionary
        new_results = {
            "Model Name": [model_name]*len(task_names)*epochs,
            "Epoch": [i for i in range(epochs)]*len(task_names),
            "Batch Size": [batch_size]*len(task_names)*epochs,
            "Learning Rate": [learning_rate]*len(task_names)*epochs,
            "Train Loss": train_losses,
            "Test Loss": test_losses,
            "Train Accuracy": train_accs,
            "Test Accuracy": test_accs,
            "Task Name": np_repeat(task_names, epochs),
            "Initial Number of Tasks": [init_n_tasks]*len(task_names)*epochs
        }
    new_results = pd.DataFrame(new_results)
    # Add results to results dataframe
    results_df = pd.concat( [results_df, new_results], ignore_index=True)
    return results_df

# Function that checks if model already exists in results dataframe
def model_exists(results_df, model_name):
    if model_name in results_df["Model Name"].values:
        return True
    else:
        return False

# Function that loads results, adds to results dataframe and saves results
def update_results(data_name, model_name, start_end_epochs, batch_size, learning_rate, train_losses, test_losses, train_accs, test_accs, task_names=False, init_n_tasks=False):
    # If task_names not False we assume we are in CL scenario, accordingly we set is_CL to True and adjust behaviour accordingly
    if task_names != False:
        is_CL = True
        CL_ext = "_CL"
    else:
        is_CL = False
        CL_ext = ""
    
    # Check if results folder exists and create if not
    if not os.path.exists("results"):
        os.makedirs("results")
        print("Created results folder")
    
    # Check if results file exists and create if not
    if not os.path.exists("results/"+data_name+CL_ext + ".csv"):
        results_df = create_results_df(is_CL)
        save_results(results_df, data_name, CL_ext=CL_ext)
        print("Created results data frame and saved to file")
    
    # Load results
    results_df = load_results(data_name, CL_ext=CL_ext)

    # Check if model already exists in results dataframe, if not in CL scenario will first check if we are training from where we left off, in which case we dont want to remove results from the data frame. If we are in the CL scenario we will remove results as we don't have any checkpointing bulit in yet
    if model_exists(results_df, model_name) and (is_CL==False):
        if (start_end_epochs[0] == 0):
            print("Model already exists in results and was trained from scratch, will remove old results")
            results_df = remove_results(results_df, model_name)
    elif model_exists(results_df, model_name) and (is_CL==True):
        print("Model already exists in results, will remove old results")
        results_df = remove_results(results_df, model_name)
    
    # Add to results
    results_df = add_to_results(results_df, model_name, start_end_epochs, batch_size, learning_rate, train_losses, test_losses, train_accs, test_accs, task_names=task_names, init_n_tasks=init_n_tasks)
    
    # Save results
    save_results(results_df, data_name, CL_ext=CL_ext)

# Function that removes results of a given model from results dataframe
def remove_results(results_df, model_name):
    # Remove results
    results_df = results_df[results_df["Model Name"] != model_name]
    return results_df