import matplotlib.pyplot as plt
import pandas as pd
import results
import os
from numpy import unique

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
        # Check if path exists
        if not os.path.exists(path):
            os.makedirs(path)
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
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + model_name + "_acc.png")
        plt.clf()

# Plot training and testing loss for current task being trained on over training for a given model
def plot_CTL(results_df, model_name, path=False, xs = False):
    # Get training and testing loss for model
    train_CTL = results_df.loc[results_df['Model Name'] == model_name, 'Train CTL'].values
    test_CTL = results_df.loc[results_df['Model Name'] == model_name, 'Test CTL'].values

    # Plot training and testing loss over time
    if xs == False:
        plt.plot(train_CTL, label="Training CTL")
        plt.plot(test_CTL, label="Testing CTL")
        plt.xlabel("Epoch")
    else:
        plt.plot(train_CTL, label="Training CTL")
        plt.plot(test_CTL, label="Testing CTL")
        plt.xlabel("Task.Epoch")
        plt.xticks([i for i in range(len(xs))], [str(x) for x in xs])
    plt.ylabel("CTL")
    plt.title("Current Task Loss (CTL) over training for " + model_name)
    plt.legend()

    # Save or show plot
    if path == False:
        plt.show()
    else:
        # Check if path exists
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + model_name + "_CTL.png")
        plt.clf()

#  Plot training and testing accuracy for current task being trained on over training for a given model
def plot_CTA(results_df, model_name, path=False, xs = False):
    # Get training and testing loss for model
    train_CTA = results_df.loc[results_df['Model Name'] == model_name, 'Train CTA'].values
    test_CTA = results_df.loc[results_df['Model Name'] == model_name, 'Test CTA'].values

    # Plot training and testing loss over time
    if xs == False:
        plt.plot(train_CTA, label="Training CTA")
        plt.plot(test_CTA, label="Testing CTA")
        plt.xlabel("Epoch")
    else:
        plt.plot(train_CTA, label="Training CTA")
        plt.plot(test_CTA, label="Testing CTA")
        plt.xlabel("Task.Epoch")
        plt.xticks([i for i in range(len(xs))], [str(x) for x in xs])
    plt.ylabel("CTA")
    plt.title("Current Task Accuracy (CTA) over training for " + model_name)
    plt.legend()

    # Save or show plot
    if path == False:
        plt.show()
    else:
        # Check if path exists
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + model_name + "_CTA.png")
        plt.clf()

# Plot training and testing loss for all task trained on up to current task being trained on over training for a given model
def plot_PTL(results_df, model_name, path=False, xs = False):
    # Get training and testing loss for model
    train_PTL = results_df.loc[results_df['Model Name'] == model_name, 'Train PTL'].values
    test_PTL = results_df.loc[results_df['Model Name'] == model_name, 'Test PTL'].values

    # Plot training and testing loss over time
    if xs == False:
        plt.plot(train_PTL, label="Training PTL")
        plt.plot(test_PTL, label="Testing PTL")
        plt.xlabel("Epoch")
    else:
        plt.plot(train_PTL, label="Training PTL")
        plt.plot(test_PTL, label="Testing PTL")
        plt.xlabel("Task.Epoch")
        plt.xticks([i for i in range(len(xs))], [str(x) for x in xs])
    plt.ylabel("PTL")
    plt.title("Past Task Loss (PTL) over training for " + model_name)
    plt.legend()

    # Save or show plot
    if path == False:
        plt.show()
    else:
        # Check if path exists
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + model_name + "_PTL.png")
        plt.clf()
    
# Plot training and testing accuracy for all task trained on up to current task being trained on over training for a given model
def plot_PTA(results_df, model_name, path=False, xs = False):
    # Get training and testing loss for model
    train_PTA = results_df.loc[results_df['Model Name'] == model_name, 'Train PTA'].values
    test_PTA = results_df.loc[results_df['Model Name'] == model_name, 'Test PTA'].values

    # Plot training and testing loss over time
    if xs == False:
        plt.plot(train_PTA, label="Training PTA")
        plt.plot(test_PTA, label="Testing PTA")
        plt.xlabel("Epoch")
    else:
        plt.plot(train_PTA, label="Training PTA")
        plt.plot(test_PTA, label="Testing PTA")
        plt.xlabel("Task.Epoch")
        plt.xticks([i for i in range(len(xs))], [str(x) for x in xs])
    plt.ylabel("PTA")
    plt.title("Past Task Accuracy (PTA) over training for " + model_name)
    plt.legend()

    # Save or show plot
    if path == False:
        plt.show()
    else:
        # Check if path exists
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + model_name + "_PTA.png")
        plt.clf()

# Plot training and testing loss seperately for all task trained on up to current task being trained on over training for a given model
def plot_TLs(results_df, model_name, path=False, xs = False):
    # Get list of task ids
    task_ids = results_df.loc[results_df['Model Name'] == model_name, 'Task Name'].values
    # Only keep unique task ids
    task_ids = unique(task_ids)
    for task_id in task_ids:
        task_id = str(task_id)
        # Get training and testing loss for model
        train_TL = results_df.loc[results_df['Model Name'] == model_name, 'Train TL task '+task_id].values
        test_TL = results_df.loc[results_df['Model Name'] == model_name, 'Test TL task '+task_id].values

        # Plot training and testing loss over time
        if xs == False:
            plt.plot(train_TL, label="Training TL for task "+ task_id)
            plt.plot(test_TL, label="Testing TL for task "+ task_id)
            plt.xlabel("Epoch")
        else:
            plt.plot(train_TL, label="Training TL for task "+ task_id)
            plt.plot(test_TL, label="Testing TL for task "+ task_id)
            plt.xlabel("Task.Epoch")
            plt.xticks([i for i in range(len(xs))], [str(x) for x in xs])
        plt.ylabel("TL for task "+ task_id)
        plt.title("Task Loss (TL) on task "+ task_id +" over training for " + model_name)
        plt.legend()

        # Save or show plot
        if path == False:
            plt.show()
        else:
            # Check if path exists
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(path + model_name + "_TL_task_"+ task_id +".png")
            plt.clf()

# Function that adds padding at begining so that all lists are same length
def pad_list(lists):
    # Get max length
    max_len = max([len(l) for l in lists])
    # Pad all lists
    for i in range(len(lists)):
        lists[i] = [0]*(max_len-len(lists[i])) + lists[i]
    return lists

# Plot training and testing loss for all task trained on up to current task being trained on over training for a given model
def plot_TL(results_df, model_name, path=False, xs = False):
    # Get list of task ids
    task_ids = results_df.loc[results_df['Model Name'] == model_name, 'Task Name'].values
    # Get total number of epochs
    epochs = [i for i in range(len(task_ids))]
    # Only keep unique task ids
    task_ids = unique(task_ids)

    # Create figure for plotting
    plt.figure()
    for task_id in task_ids:
        task_id = str(task_id)
        # Get training and testing loss for model
        train_TL = results_df.loc[results_df['Model Name'] == model_name, 'Train TL task '+task_id].values
        test_TL = results_df.loc[results_df['Model Name'] == model_name, 'Test TL task '+task_id].values
        # Pad lists
        train_TL, test_TL = pad_list([train_TL, test_TL])

        # Plot training and testing loss over time
        if xs == False:
            plt.plot(epochs, train_TL, label="Training TL for task "+ task_id)
            plt.plot(epochs, test_TL, label="Testing TL for task "+ task_id, linestyle="dashed")
            plt.xlabel("Epoch")
        else:
            plt.plot(epochs, train_TL, label="Training TL for task "+ task_id)
            plt.plot(epochs, test_TL, label="Testing TL for task "+ task_id, linestyle="dashed")
            plt.xlabel("Task.Epoch")
            plt.xticks([i for i in range(len(xs))], [str(x) for x in xs])
        plt.ylabel("Task Loss (TL)")
        plt.title("Task Loss (TL) on tasks over training for " + model_name)
        plt.legend()

    # Save or show plot
    if path == False:
        plt.show()
    else:
        # Check if path exists
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + model_name + "_TL_task.png")
        plt.clf()

# Plot training and testing loss seperately for all task trained on up to current task being trained on over training for a given model
def plot_TAs(results_df, model_name, path=False, xs = False):
    # Get list of task ids
    task_ids = results_df.loc[results_df['Model Name'] == model_name, 'Task Name'].values
    # Only keep unique task ids
    task_ids = unique(task_ids)
    for task_id in task_ids:
        task_id = str(task_id)
        # Get training and testing loss for model
        train_TA = results_df.loc[results_df['Model Name'] == model_name, 'Train TA task '+task_id].values
        test_TA = results_df.loc[results_df['Model Name'] == model_name, 'Test TA task '+task_id].values

        # Plot training and testing loss over time
        if xs == False:
            plt.plot(train_TA, label="Training TA for task "+ task_id)
            plt.plot(test_TA, label="Testing TA for task "+ task_id)
            plt.xlabel("Epoch")
        else:
            plt.plot(train_TA, label="Training TA for task "+ task_id)
            plt.plot(test_TA, label="Testing TA for task "+ task_id)
            plt.xlabel("Task.Epoch")
            plt.xticks([i for i in range(len(xs))], [str(x) for x in xs])
        plt.ylabel("TA for task "+ task_id)
        plt.title("Task Accuracy (TA) on task "+ task_id +" over training for " + model_name)
        plt.legend()

        # Save or show plot
        if path == False:
            plt.show()
        else:
            # Check if path exists
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(path + model_name + "_TA_task_"+ task_id +".png")
            plt.clf()

# Plot training and testing accuracy for all task trained on up to current task being trained on over training for a given model
def plot_TA(results_df, model_name, path=False, xs = False):
    # Get list of task ids
    task_ids = results_df.loc[results_df['Model Name'] == model_name, 'Task Name'].values
    # Get total number of epochs
    epochs = [i for i in range(len(task_ids))]
    # Only keep unique task ids
    task_ids = unique(task_ids)

    # Create figure for plotting
    plt.figure()
    for task_id in task_ids:
        task_id = str(task_id)
        # Get training and testing loss for model
        train_TA = results_df.loc[results_df['Model Name'] == model_name, 'Train TA task '+task_id].values
        test_TA = results_df.loc[results_df['Model Name'] == model_name, 'Test TA task '+task_id].values
        # Pad lists
        train_TA, test_TA = pad_list([train_TA, test_TA])

        # Plot training and testing loss over time
        if xs == False:
            plt.plot(epochs, train_TA, label="Training TA for task "+ task_id)
            plt.plot(epochs, test_TA, label="Testing TA for task "+ task_id, linestyle="dashed")
            plt.xlabel("Epoch")
        else:
            plt.plot(epochs, train_TA, label="Training TA for task "+ task_id)
            plt.plot(epochs, test_TA, label="Testing TA for task "+ task_id, linestyle="dashed")
            plt.xlabel("Task.Epoch")
            plt.xticks([i for i in range(len(xs))], [str(x) for x in xs])
        plt.ylabel("Task Accuracy (TA)")
        plt.title("Task Accuracy (TA) on tasks over training for " + model_name)
        plt.legend()

    # Save or show plot
    if path == False:
        plt.show()
    else:
        # Check if path exists
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + model_name + "_TA_task.png")
        plt.clf()

# Given dataframe and model name creates x values for plotting from epochs and task columns
def create_xs(results_df, model_name):
    # Get epochs and tasks for model
    epochs = results_df.loc[results_df['Model Name'] == model_name, 'Epoch'].values
    tasks = results_df.loc[results_df['Model Name'] == model_name, 'Task Name'].values
    # Create x values
    xs = []
    for i in range(len(epochs)):
        xs.append(float(str(tasks[i]) + "." + str(epochs[i])))
    return xs

# Given a list of data sets used and model names ran in a session create and save all default plots
def plot_default(data_names, model_names, is_CL: bool=False):
    # Use different path if we are plotting CL results
    if(is_CL):
        CL_ext="_CL"
    else:
        CL_ext=""

    if(type(data_names) == list):
        zipped = zip(data_names, model_names)
        zipped = sorted(zipped, key=lambda x: x[1])
        curr = ""
        for (data_name, model_name) in zipped:
            if(data_name != curr):
                # Load new results
                results_df = results.load_results(data_name, CL_ext=CL_ext)
                path = "plots/"+ data_name + CL_ext +"/" 
                curr = data_name
            if(is_CL == False):
                # Plot loss and accuracy for each model
                plot_loss(results_df, model_name, path=path)
                plot_accuracy(results_df, model_name, path=path)
            else:
                # Plot all key metrics for each model
                xs = create_xs(results_df, model_name)
                plot_CTL(results_df, model_name, path=path, xs=xs)
                plot_CTA(results_df, model_name, path=path, xs=xs)
                plot_PTL(results_df, model_name, path=path, xs=xs)
                plot_PTA(results_df, model_name, path=path, xs=xs)
                plot_TL(results_df, model_name, path=path, xs=xs)
                plot_TA(results_df, model_name, path=path, xs=xs)
    else:
        # Load new results
        results_df = results.load_results(data_names, CL_ext=CL_ext)
        path = "plots/"+ data_names + CL_ext +"/" 
        if(is_CL == False):
            # Plot loss and accuracy for each model
            plot_loss(results_df, model_names, path=path)
            plot_accuracy(results_df, model_names, path=path)
        else:
            # Plot all key metrics for each model
            xs = create_xs(results_df, model_names)
            plot_CTL(results_df, model_names, path=path, xs=xs)
            plot_CTA(results_df, model_names, path=path, xs=xs)
            plot_PTL(results_df, model_names, path=path, xs=xs)
            plot_PTA(results_df, model_names, path=path, xs=xs)
            plot_TL(results_df, model_names, path=path, xs=xs)
            plot_TA(results_df, model_names, path=path, xs=xs)
