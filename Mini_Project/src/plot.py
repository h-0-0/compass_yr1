import matplotlib.pyplot as plt
import pandas as pd
import results

# Plot training and testing loss over time for a given model
def plot_loss(results_df, model_name, path=False):
    # Get training and testing loss for model
    train_losses = results_df.loc[results_df['Model Name'] == model_name, 'Train Loss'].values
    test_losses = results_df.loc[results_df['Model Name'] == model_name, 'Test Loss'].values

    # Plot training and testing loss over time
    plt.plot(train_losses, label="Training loss")
    plt.plot(test_losses, label="Testing loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over training for " + model_name)
    plt.legend()

    # Save or show plot
    if path == False:
        plt.show()
    else:
        plt.savefig(path + model_name + "_loss.png")
        plt.clf()

# Plot training and testing accuracy over time for a given model
def plot_accuracy(results_df, model_name, path=False):
    # Get training and testing accuracy for model
    train_accs = results_df.loc[results_df['Model Name'] == model_name, 'Train Accuracy'].values
    test_accs = results_df.loc[results_df['Model Name'] == model_name, 'Test Accuracy'].values

    # Plot training and testing accuracy over time
    plt.plot(train_accs, label="Training accuracy")
    plt.plot(test_accs, label="Testing accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over training for " + model_name)
    plt.legend()

    # Save or show plot
    if path == False:
        plt.show()
    else:
        plt.savefig(path + model_name + "_acc.png")
        plt.clf()

# Given a list of data sets used and model names ran in a session create and save all default plots
def plot_default(data_names, model_names):
    zipped = zip(data_names, model_names)
    zipped = sorted(zipped, key=lambda x: x[1])
    curr = ""
    for (data_name, model_name) in zipped:
        if(data_name != curr):
            # Load new results
            results_df = results.load_results(data_name)
            path = "plots/"+ data_name +"/" 
            curr = data_name
        # Plot loss and accuracy for each model
        plot_loss(results_df, model_name, path=path)
        plot_accuracy(results_df, model_name, path=path)