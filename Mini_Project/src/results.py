import os
import pandas as pd

# Function that creates data frame to store results
def create_results_df():
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
    return results_df

# Function to save results to csv file
def save_results(results_df, data_name):
    # Check if results folder exists
    if not os.path.exists("results"):
        os.makedirs("results")
    # Save results
    results_df.to_csv("results/"+data_name+".csv", index=False, mode='w+')
    print("Saved results to results/"+data_name+".csv")

# Function to load results from csv file
def load_results(data_name):
    # Check if results folder exists
    if not os.path.exists("results"):
        raise Exception("No results folder found, cannot load results")
    # Load results
    results_df = pd.read_csv("results/"+data_name+".csv")
    print("Loaded results from results/"+data_name+".csv")
    return results_df

# Function to add to results dataframe
def add_to_results(results_df, model_name, start_end_epochs, batch_size, learning_rate, train_losses, test_losses, train_accs, test_accs):
    # Add results to dictionary
    new_results = {
        "Model Name": [model_name]*(start_end_epochs[1]- start_end_epochs[0]),
        "Epoch": [i for i in range(start_end_epochs[0], start_end_epochs[1])],
        "Batch Size": [batch_size]*(start_end_epochs[1]- start_end_epochs[0]),
        "Learning Rate": [learning_rate]*(start_end_epochs[1]- start_end_epochs[0]),
        "Train Loss": train_losses,
        "Test Loss": test_losses,
        "Train Accuracy": train_accs,
        "Test Accuracy": test_accs,
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
def update_results(data_name, model_name, start_end_epochs, batch_size, learning_rate, train_losses, test_losses, train_accs, test_accs):
    # Check if results folder exists and create if not
    if not os.path.exists("results"):
        os.makedirs("results")
        print("Created results folder")
    # Check if results file exists and create if not
    if not os.path.exists("results/"+data_name+".csv"):
        results_df = create_results_df()
        save_results(results_df, data_name)
        print("Created results data frame and saved to file")
    
    # Load results
    results_df = load_results(data_name)
    # Check if model already exists in results dataframe
    if model_exists(results_df, model_name) and start_end_epochs[0] == 0:
        print("Model already exists in results and was trained from scratch, will remove old results")
        results_df = remove_results(results_df, model_name)
    # Add to results
    results_df = add_to_results(results_df, model_name, start_end_epochs, batch_size, learning_rate, train_losses, test_losses, train_accs, test_accs)
    # Save results
    save_results(results_df, data_name)

# Function that removes results of a given model from results dataframe
def remove_results(results_df, model_name):
    # Remove results
    results_df = results_df[results_df["Model Name"] != model_name]
    return results_df
