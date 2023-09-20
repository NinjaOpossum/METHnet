# @MPR

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from albumentations.pytorch import ToTensorV2

from progress.bar import IncrementalBar

from models import unet
from datastructure import dataset_torch
from utils.train_utils import dice_coef, CompositeLoss
from utils.helper import plot_loss, plot_examples

import os
import matplotlib.pyplot as plt

def train(train_set, validation_set, k, log_dir, setting):

    network_settings = setting.get_network_setting()
    class_settings = setting.get_class_setting()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    n_classes = class_settings.get_n_classes()
    model = unet.UNet(network_settings.get_in_channels(),
                    n_classes,
                    network_settings.get_depth(),
                    network_settings.get_wf(),
                    True,
                    network_settings.get_batch_norm(),
                    network_settings.get_up_mode())

    model.to(device)
    model.load_state_dict(torch.load(r"C:\Users\PC\Desktop\GBM_Pathology\workdir\Runs\Pathology-23-09-17-23-34-47\fold_0\model_last"))

    criterion = torch.nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters())

    train_dataset = dataset_torch.PathologyDataset(train_set)
    validation_dataset = dataset_torch.PathologyDataset(validation_set)

    batch_size = network_settings.get_batch_size()

    train_dataloader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
        pin_memory = True,
    )
    
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size = batch_size,
        shuffle = True,
        pin_memory = True,
    )

    epochs = network_settings.get_epochs()
    n_batches_train = min(network_settings.get_patches_per_epoch() // network_settings.get_batch_size(), len(train_dataloader))

    writer = SummaryWriter(os.path.join(log_dir, "tensorboard_logs"))
    results_csv = pd.DataFrame(columns = ['Epoch', 
                                          'Train Loss', 
                                          'Validation Loss', 
                                          'Validation Dice Score', 
                                          'Train Patches', 
                                          'Validation Patches'])
    

    min_val_loss = np.inf

    for epoch in range(epochs):
        print(f'Starting to the epoch {epoch}\n')

        bar = IncrementalBar('Training The Model', max = n_batches_train)
        total_loss = 0

        for idx, (images, annotations) in enumerate(train_dataloader):
            images = images.to(device)
            annotations = annotations.float().to(device)
            optim.zero_grad()
            preds = model(images)
            loss = criterion(preds, annotations)
            loss.backward()
            optim.step()
            writer.add_scalar("Loss/Train_Step", loss, idx + epoch * n_batches_train)
            bar.next()
            total_loss += loss.item()
            if idx == n_batches_train - 1:
                images = images.detach().cpu().numpy()
                annotations = annotations.detach().cpu().numpy()
                pred_binary = (preds.detach().cpu().numpy() > 0).astype(int)
                plot_examples(log_dir, images, annotations, pred_binary, f'epoch{epoch}_train')
                break

        n_batches_validation = min(network_settings.get_validation_patch_limit() // network_settings.get_batch_size(), len(validation_dataloader))

        print()
        bar = IncrementalBar('Validating The Model', max = n_batches_validation)
        mean_train_loss =  total_loss / n_batches_train
        writer.add_scalar("Loss/Train_Epoch",mean_train_loss, epoch)
        torch.save(model.state_dict(), os.path.join(log_dir, "model_last"))

        with torch.no_grad():
            mean_dice = 0
            mean_validation_loss = 0
            for idx, (images, annotations) in enumerate(validation_dataloader):
                images = images.to(device)
                annotations = annotations.float().to(device)
                preds = model(images)
                loss = criterion(preds, annotations)
                mean_validation_loss += loss.item() / n_batches_validation
                annotations = annotations.cpu().numpy()
                pred_binary = (preds.cpu().numpy() > 0).astype(int)
                mean_dice += dice_coef(annotations, pred_binary, n_classes) / n_batches_validation
                bar.next()
                if idx == n_batches_validation - 1:
                    images = images.cpu().numpy()
                    plot_examples(log_dir, images, annotations, pred_binary, f'epoch{epoch}_validation')
                    break
            writer.add_scalar('Dice_Score/Validation_Epoch', mean_dice, epoch)
            writer.add_scalar('Loss/Validation_Epoch', mean_validation_loss, epoch)
            results_csv.loc[len(results_csv)] = [epoch, 
                                                 mean_train_loss, 
                                                 mean_validation_loss, 
                                                 mean_dice, 
                                                 n_batches_train * batch_size, 
                                                 n_batches_validation * batch_size]
            results_csv.to_csv(os.path.join(log_dir, "results.csv"))
            plot_loss(log_dir, f'Fold {k}')
            print(f"\nFold {k} - Epoch {epoch}: Mean Loss {mean_validation_loss:.3f} Mean Dice {mean_dice:.3f}\n")
            if mean_validation_loss < min_val_loss:
                torch.save(model.state_dict(), os.path.join(log_dir, "model_best"))
                print(f"Model is saved at epoch {epoch} with the validation loss {mean_validation_loss}!")
                min_val_loss = mean_validation_loss
        
