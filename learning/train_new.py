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
from utils.train_utils import dice_coef_per_label, get_precision_recall
from utils.helper import plot_loss, plot_examples, plot_accuracy, plot_dices

import os

def train(train_set, validation_set, k, log_dir, setting):
    network_settings = setting.get_network_setting()

    # train_dataset = dataset_torch.PathologyDataset(train_set)
    train_dataset = dataset_torch.PathologyDataset(validation_set)
    validation_dataset = dataset_torch.PathologyDataset(validation_set)
    classes = train_dataset.classes
    n_classes = len(classes) + 1 #Background
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    torch.cuda.empty_cache()

    model = unet.UNet(network_settings.get_in_channels(),
                    n_classes,
                    network_settings.get_depth(),
                    network_settings.get_wf(),
                    True,
                    network_settings.get_batch_norm(),
                    network_settings.get_up_mode())

    model.to(device)
    #model.load_state_dict(torch.load(r"C:\Users\PC\Desktop\Fiverr\Max_Pathology\workdir\Runs\Pathology-23-09-17-23-34-47\fold_0\model_last"))

    criterion = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters())

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
                                          'Val Loss', 
                                          'Val Dice Score', 
                                          'Train Patches', 
                                          'Val Patches',
                                          'Train Accuracy',
                                          'Train Foreground Accuracy',
                                          'Val Accuracy',
                                          'Val Foreground Accuracy',
                                          'Val Precision',
                                          'Val Recall'] + [f"Val_{i}_Dice Score" for i in classes])
    

    min_val_loss = np.inf

    for epoch in range(epochs):
        print(f'Starting to the epoch {epoch}\n')

        bar = IncrementalBar('Training The Model', max = n_batches_train)
        total_loss = 0
        train_pred_accuracy = 0
        train_pred_foreground_accuracy = 0
        for idx, (images, annotations) in enumerate(train_dataloader):
            images = images.to(device)
            annotations = annotations.long().to(device)
            optim.zero_grad()
            preds = model(images)
            loss = criterion(preds, annotations)
            loss.backward()
            optim.step()
            writer.add_scalar("Loss/Train_Step", loss, idx + epoch * n_batches_train)
            bar.next()
            total_loss += loss.item()
            annotations = annotations.detach().cpu().numpy()
            pred_index = (preds.detach().cpu().numpy().argmax(axis = 1)).astype(int)
            train_pred_accuracy += (pred_index == annotations).mean() / n_batches_train
            train_pred_foreground_accuracy += ((pred_index == annotations) * (annotations > 0)).sum() / (annotations > 0).sum() / n_batches_train
            if idx == n_batches_train - 1:
                images = images.detach().cpu().numpy()
                plot_examples(log_dir, images, annotations, pred_index, f'epoch{epoch}_train', len(classes))
                break

        n_batches_validation = min(network_settings.get_validation_patch_limit() // network_settings.get_batch_size(), len(validation_dataloader))

        print()
        bar = IncrementalBar('Validating The Model', max = n_batches_validation)
        mean_train_loss =  total_loss / n_batches_train
        writer.add_scalar("Loss/Train_Epoch",mean_train_loss, epoch)
        writer.add_scalar("Accuracy/Train_Accuracy",train_pred_accuracy, epoch)
        writer.add_scalar("Accuracy/Train_Foreground_Accuracy",train_pred_foreground_accuracy, epoch)
        torch.save(model.state_dict(), os.path.join(log_dir, "model_last"))

        with torch.no_grad():
            mean_dice = 0
            mean_validation_loss = 0
            val_pred_accuracy = 0
            val_pred_foreground_accuracy = 0
            precision = 0
            recall = 0
            dices_arr = np.zeros(len(classes))
            for idx, (images, annotations) in enumerate(validation_dataloader):
                images = images.to(device)
                annotations = annotations.long().to(device)
                preds = model(images)
                loss = criterion(preds, annotations)
                mean_validation_loss += loss.item() / n_batches_validation
                annotations = annotations.cpu().numpy()
                pred_index = (preds.cpu().numpy().argmax(axis = 1)).astype(int)
                val_pred_accuracy += (pred_index == annotations).mean() / n_batches_validation
                val_pred_foreground_accuracy += ((pred_index == annotations) * (annotations > 0)).sum() / (annotations > 0).sum() / n_batches_validation
                dices = np.array(list(dice_coef_per_label(annotations, pred_index, len(classes) + 1).values()))
                dices_arr += dices / n_batches_validation
                mean_dice += dices.mean() / n_batches_validation
                bar.next()
                cur_precision, cur_recall = get_precision_recall(pred_index, annotations, n_classes)
                precision += cur_precision / n_batches_validation
                recall += cur_recall / n_batches_validation
                if idx == n_batches_validation - 1:
                    images = images.cpu().numpy()
                    plot_examples(log_dir, images, annotations, pred_index, f'epoch{epoch}_validation', len(classes))
                    break
            for cls, dice in zip(classes, dices_arr):
                writer.add_scalar(f'Dice_Score/Val_{cls}', dice, epoch)
            writer.add_scalar('Dice_Score/Val_Mean', mean_dice, epoch)
            writer.add_scalar('Loss/Val_Epoch', mean_validation_loss, epoch)
            writer.add_scalar("Accuracy/Val_Accuracy", val_pred_accuracy, epoch)
            writer.add_scalar("Accuracy/Val_Foreground_Accuracy", val_pred_foreground_accuracy, epoch)
            writer.add_scalar("Precision/Val_Precision", precision, epoch)
            writer.add_scalar("Recall/Val_Recall", recall, epoch)
            results_csv.loc[len(results_csv)] = [epoch, 
                                                 mean_train_loss, 
                                                 mean_validation_loss, 
                                                 mean_dice, 
                                                 n_batches_train * batch_size, 
                                                 n_batches_validation * batch_size,
                                                 train_pred_accuracy,
                                                 train_pred_foreground_accuracy,
                                                 val_pred_accuracy,
                                                 val_pred_foreground_accuracy, 
                                                 precision,
                                                 recall] + list(dices_arr)
            results_csv.to_csv(os.path.join(log_dir, "results.csv"))
            plot_loss(log_dir, f'Fold {k}')
            plot_accuracy(log_dir, f'Fold {k}')
            plot_dices(log_dir, f'Fold {k}')

            print(f"\nFold {k} - Epoch {epoch}: Mean Loss {mean_validation_loss:.3f} Mean Dice {mean_dice:.3f}\n")
            if mean_validation_loss < min_val_loss:
                torch.save(model.state_dict(), os.path.join(log_dir, "model_best"))
                print(f"Model is saved at epoch {epoch} with the validation loss {mean_validation_loss}!")
                min_val_loss = mean_validation_loss
        
