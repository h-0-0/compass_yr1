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
def save_results(results_df, file_name = "results"):
    # Check if results folder exists
    if not os.path.exists("results"):
        os.makedirs("results")
    # Save results
    results_df.to_csv("results/"+file_name+".csv", index=False)
    print("Saved results to results/"+file_name+".csv")

# Function to load results from csv file
def load_results(file_name="results"):
    # Check if results folder exists
    if not os.path.exists("results"):
        raise Exception("No results folder found, cannot load results")
    # Load results
    results_df = pd.read_csv("results/"+file_name+".csv")
    print("Loaded results from results/"+file_name+".csv")
    return results_df

# Function to add to results dataframe
def add_to_results(results_df, model_name, epochs, batch_size, learning_rate, train_losses, test_losses, train_accs, test_accs):
    # Add results to dictionary
    new_results = {
        "Model Name": [model_name]*epochs,
        "Epoch": [i for i in range(epochs)],
        "Batch Size": [batch_size]*epochs,
        "Learning Rate": [learning_rate]*epochs,
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
def model_exists(model_name, results_df):
    if model_name in results_df["Model Name"].values:
        return True
    else:
        return False

# Function that loads results, adds to results dataframe and saves results
def update_results(model_name, epochs, batch_size, learning_rate, train_losses, test_losses, train_accs, test_accs, file_name="results"):
    # Check if results folder exists and create if not
    if not os.path.exists("results"):
        os.makedirs("results")
        print("Created results folder")
    # Check if results file exists and create if not
    if not os.path.exists("results/"+file_name+".csv"):
        results_df = create_results_df()
        save_results(results_df, file_name)
        print("Created results data frame and saved to file")
    
    # Load results
    results_df = load_results("results")
    # Check if model already exists in results dataframe
    if model_exists(model_name, results_df):
        print("Model already exists in results, will remove old results")
        remove_results(model_name, file_name)
    # Add to results
    results_df = add_to_results(results_df, model_name, epochs, batch_size, learning_rate, train_losses, test_losses, train_accs, test_accs)
    # Save results
    save_results(results_df, "results")

# Function that removes results of a given model from results dataframe
def remove_results(model_name, file_name="results"):
    # Load results
    results_df = load_results(file_name)
    # Remove results
    results_df = results_df[results_df['Model Name'] != model_name]
    # Save results
    save_results(results_df, file_name)

