import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def create_folder(folder):
        
    subfolders = folder.split('/')
    current_path = ''

    for sf in subfolders:
        current_path += sf + '/'

        if not os.path.exists(current_path):
            os.mkdir(current_path)

# @MPR
# added some plot funktions to visualize data
def plot_loss(log_dir, title):
    df = pd.read_csv(os.path.join(log_dir, "results.csv"))
    epochs = df["Epoch"]
    train_losses = df["Train Loss"]
    validation_losses = df["Val Loss"]
    dice_score = df["Val Dice Score"]
    
    fig, axes = plt.subplots(1, 2, figsize = (25, 7))
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)
    axes[0].set_xticks(epochs)
    axes[0].plot(train_losses, label = "Train Loss")
    axes[0].plot(validation_losses, label = "Validation Loss")
    axes[0].legend()
    
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Score")
    axes[1].grid(True)
    axes[1].set_xticks(epochs)
    axes[1].plot(dice_score, label = "Overall Val Dice Score", color = "red")
    axes[1].set_ylim((0, 1))
    axes[1].legend()
    fig.suptitle(title)
    fig.set_facecolor("white")
    fig.savefig(os.path.join(log_dir, "results_loss.png"))

def plot_dices(log_dir, title):
    df = pd.read_csv(os.path.join(log_dir, "results.csv"))
    epochs = df["Epoch"]
    dice_by_class_start_index =  df.columns.get_loc('Val Recall') + 1
    dice_scores = df["Val Dice Score"]

    fig, ax = plt.subplots(1, figsize = (20, 7))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice Score")
    ax.grid(True)
    ax.set_xticks(epochs)
    ax.plot(dice_scores, label = "Overall Val Dice Score", ls = "--", linewidth=3)
    for cls in df.columns[dice_by_class_start_index:]:
        ax.plot(df[cls], label = cls.split("_")[1] + "_DS")
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    fig.suptitle(title)
    fig.set_facecolor("white")
    fig.subplots_adjust(right=0.8)
    fig.savefig(os.path.join(log_dir, "results_ind_dice.png"))

def plot_accuracy(log_dir, title):
    df = pd.read_csv(os.path.join(log_dir, "results.csv"))
    epochs = df["Epoch"]
    train_accuracies = df["Train Accuracy"]
    train_foreground_accuracies = df["Train Foreground Accuracy"]
    val_accuracies = df["Val Accuracy"]
    val_foreground_accuracies = df["Val Foreground Accuracy"]
    
    fig, axes = plt.subplots(1, 2, figsize = (20, 7))
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True)
    axes[0].set_xticks(epochs)
    axes[0].plot(train_accuracies, label = "Train Accuracy")
    axes[0].plot(val_accuracies, label = "Validation Accuracy")
    axes[0].legend()
    axes[0].set_ylim((0, 1))
    
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Foreground Accuracy")
    axes[1].grid(True)
    axes[1].set_xticks(epochs)
    axes[1].plot(train_foreground_accuracies, label = "Train Accuracy")
    axes[1].plot(val_foreground_accuracies, label = "Validation Accuracy")
    axes[1].legend()
    axes[1].set_ylim((0, 1))
    fig.suptitle(title)
    fig.set_facecolor("white")
    fig.savefig(os.path.join(log_dir, "results_accuracy.png"))

def plot_examples(logdir, images, annotations, preds, identifier, n_classes):
    inference_folder = os.path.join(logdir, "inferences")
    create_folder(inference_folder)
    for i in range(images.shape[0]):
        plt.imsave(os.path.join(inference_folder, f"{identifier}{i}_image.png"), 
                   np.transpose(images[i], [1, 2, 0]))
        # TODO: Update the following lines as if annotations have multiple channels for the future
        plt.imsave(os.path.join(inference_folder, f"{identifier}{i}_annotation.png"), 
                   annotations[i], vmin = 0, vmax = n_classes)
        plt.imsave(os.path.join(inference_folder, f"{identifier}{i}_pred.png"), 
                   preds[i], vmin = 0, vmax = n_classes)
