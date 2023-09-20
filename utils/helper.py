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
def plot_loss(log_dir, title):
    df = pd.read_csv(os.path.join(log_dir, "results.csv"))
    epochs = df["Epoch"]
    train_losses = df["Train Loss"]
    validation_losses = df["Validation Loss"]
    dice_scores = df["Validation Dice Score"]
    
    fig, axes = plt.subplots(1, 2, figsize = (20, 7))
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
    axes[1].plot(dice_scores, label = "Validation Dice Score", color = "red")
    axes[1].legend()
    axes[1].set_ylim((0, 1))
    fig.suptitle(title)
    fig.set_facecolor("white")
    fig.savefig(os.path.join(log_dir, "results.png"))

def plot_examples(logdir, images, annotations, preds, identifier):
    inference_folder = os.path.join(logdir, "inferences")
    create_folder(inference_folder)
    for i in range(images.shape[0]):
        plt.imsave(os.path.join(inference_folder, f"{identifier}{i}_image.png"), 
                   np.transpose(images[i], [1, 2, 0]))
        # TODO: Must update the following lines as if annotations have multiple channels for the future
        plt.imsave(os.path.join(inference_folder, f"{identifier}{i}_annotation.png"), 
                   np.squeeze(annotations[i]), vmin = 0, vmax = 1)
        plt.imsave(os.path.join(inference_folder, f"{identifier}{i}_pred.png"), 
                   np.squeeze(preds[i]), vmin = 0, vmax = 1)