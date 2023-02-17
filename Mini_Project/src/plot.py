import matplotlib.pyplot as plt
import pandas as pd
import results

# Plot training and testing loss over time for a given model
def plot_loss(results_df, model_name, title="Loss over time"):
    # Get training and testing loss for model
    train_losses = results_df.loc[results_df['Model Name'] == model_name, 'Train Loss'].values
    test_losses = results_df.loc[results_df['Model Name'] == model_name, 'Test Loss'].values

    # Plot training and testing loss over time
    plt.plot(train_losses, label="Training loss")
    plt.plot(test_losses, label="Testing loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.show()

# Plot training and testing accuracy over time for a given model
def plot_accuracy(results_df, model_name, title="Accuracy over time"):
    # Get training and testing accuracy for model
    train_accs = results_df.loc[results_df['Model Name'] == model_name, 'Train Accuracy'].values
    test_accs = results_df.loc[results_df['Model Name'] == model_name, 'Test Accuracy'].values

    # Plot training and testing accuracy over time
    plt.plot(train_accs, label="Training accuracy")
    plt.plot(test_accs, label="Testing accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.show()

#TODO: deal with saving functionality