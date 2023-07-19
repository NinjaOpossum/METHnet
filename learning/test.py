#code borrowed from https://github.com/choosehappy/PytorchDigitalPathology

#v1
#26/10/2018


dataname="GBM" #should match the value used to train the network, will be used to load the appropirate model
gpuid=0


patch_size=256 #should match the value used to train the network
batch_size=1 #nicer to have a single batch so that we can iterately view the output, while not consuming too much 
edge_weight=1

# https://github.com/jvanvugt/pytorch-unet
#torch.multiprocessing.set_start_method("fork")
import random, sys
import cv2
import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.ndimage
import skimage
import time

import skimage.io as skio
from sklearn.metrics import confusion_matrix

# from imgDataset import GBMIMGDataset
import pickle
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from models.unet import UNet
from utils import dice_coef_multilabel, dice_coef_per_label 

import PIL

#load the model, note that the paramters are coming from the checkpoint, since the architecture of the model needs to exactly match the weights saved
torch.cuda.set_device(gpuid)
device = torch.device(f'cuda:{gpuid}' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(f"../models/{dataname}_unet_best_model.pth")
model = UNet(n_classes=checkpoint["n_classes"], in_channels=checkpoint["in_channels"], padding=checkpoint["padding"],depth=checkpoint["depth"],
             wf=checkpoint["wf"], up_mode=checkpoint["up_mode"], batch_norm=checkpoint["batch_norm"]).to(device)
print(f"total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")
print(device)
model.load_state_dict(checkpoint["model_dict"])

dataset = {}
dataLoader = {}

# load test dataset from train_unet
with open('../test_dataset.pkl', 'rb') as f:
    testDataset = pickle.load(f)


dataLoader["test"] = DataLoader(
    testDataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)

# Nehmen Sie ein einzelnes Batch von Daten
dataiter = iter(dataLoader["test"])
images, labels = dataiter.__next__()

# Jetzt sind 'images' und 'labels' Tensoren, die ein Batch von Bildern bzw. Labels enthalten.
# Sie können sie wie jeden anderen Tensor in PyTorch verwenden.
# Zum Beispiel, um die Größe des Batches zu sehen:
print(images.shape)
print(labels.shape)

# Um ein einzelnes Bild aus dem Batch anzuzeigen, können Sie matplotlib verwenden:
import matplotlib.pyplot as plt

# Wählen Sie das erste Bild im Batch
image = images[0]
label = labels[0]

# Wenn das Bild mehrere Kanäle hat (z.B. RGB), müssen wir die Kanal-Dimension an das Ende verschieben
if image.shape[0] == 3:  # RGB-Bilder
    image = image.permute(1, 2, 0)
    print(image.shape)


# Erstelle eine Figur mit zwei Subplots
fig, ax = plt.subplots(1, 2)

# Zeige das Bild und das Label
ax[0].imshow(image)
ax[0].set_title('Image')

# Wir gehen davon aus, dass das Label nur einen Kanal hat und als Graustufenbild dargestellt werden kann.
ax[1].imshow(label.squeeze(), cmap='gray')
ax[1].set_title('Label')

plt.show()

# show some predictions and get average precision (per label) for unseen inputs
nmbr_classes = 2
avg_prec = 0
avg_prec_label = {}
under_80 = 0  # Zählt die Anzahl der Dice-Koeffizienten unter 0.8
dice_scores = []

'''
Hier wird eine Schleife über den Testdaten-Loader gestartet. In jeder Iteration der Schleife wird ein Datenbatch 
(bestehend aus Eingabedaten X und Labels y) aus dem Daten-Loader geladen.
Die Funktion enumerate wird verwendet, um zusätzlich den Index (oder die Zählung) der aktuellen Iteration zu erhalten.
'''
for i, (X, y) in enumerate(dataLoader["test"]):
    '''
    Hier wird das Tensor y (die Labels) von Gradienten-Updates abgekoppelt (mittels detach()), unnötige Dimensionen entfernt
    (mittels squeeze()) und in ein NumPy-Array umgewandelt (mittels numpy()). detach() ist notwendig, wenn man das Tensor
    außerhalb eines Gradienten-Updating-Kontexts verwenden möchte, squeeze() entfernt alle Dimensionen mit Größe 1, und numpy()
    konvertiert das Tensor in ein NumPy-Array, da viele Operationen in PyTorch ähnlich wie in NumPy funktionieren.
    '''
    y = y.detach().squeeze().numpy()
    '''
    Hier wird das Eingabetensor X in ein NumPy-Array umgewandelt und unnötige Dimensionen werden entfernt, 
    ähnlich wie bei y oben.
    '''
    img = np.squeeze(X).numpy()
    img = ((img - img.min()) * (1/(img.max() - img.min()) * 255)).astype('uint8')
    '''
    Hier wird die Farbreihenfolge des Bildes von BGR (Blue, Green, Red), wie sie in OpenCV verwendet wird, in RGB (Red, Green, Blue)
    geändert, wie sie in vielen anderen Bibliotheken (einschließlich Matplotlib) verwendet wird.
    '''
    X = X.type('torch.FloatTensor')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    X = X.to(device, torch.float32)
    X = torch.swapaxes(X, 1, 3).swapaxes(2,3).to(device)
    prediction = model(X)
    prediction = prediction.cpu().detach().squeeze().numpy()
    prediction = prediction.argmax(axis = 0)

    dice_coef = dice_coef_multilabel(y, prediction, 2)
    dice_scores.append(dice_coef)


    if img.shape[2] != 3:
        print(f"Unerwartete Dimensionen für img: {img.shape}")
    # print(np.max(prediction))
    prediction_mask = np.ma.array(prediction, mask=prediction>0)
    prediction_mask = np.logical_and(prediction_mask,prediction_mask)
    # print(np.sum(prediction_mask))
    img_masked = img.copy()
    img_masked[prediction_mask] = (0,255,0)
    # if i%100 == 0: # Abhängig von Anzahlt der Datensätze
    fig, axs = plt.subplots(1,4, figsize=(26,26))
    axs[0].imshow(img)
    axs[0].set_title('1a: Originalbild')
    axs[1].set_title('1b: Original mit Prediction')
    axs[1].imshow(img_masked)
    #axs[1].imshow(prediction * 255 / nmbr_classes, vmin = 0, vmax= 255, alpha = 0.5)
    #axs[3].imshow([list(range(nmbr_classes))])
    # axs[3].set_title('Farblegende')
    # axs[4].imshow(y * 255 / nmbr_classes, vmin = 0, vmax= 255)
    axs[2].imshow(y)
    axs[2].set_title('1c: Originallabel')
    axs[3].imshow(prediction * 255 / nmbr_classes, vmin = 0, vmax= 255)
    axs[3].set_title('1d: Prediction')
    plt.show()
        
    ''' 
    Hier wird der Dice-Koeffizient(ein Maß für die Übereinstimmung zwischen zwei binären Bildern)
    für jede Klasse (Label) berechnet. Das Ergebnis ist ein Wörterbuch, das die Koeffizienten für jede Klasse enthält.
    '''
    # TODO:
    #  Die Tatsache, dass die Labels vor der Verarbeitung nur die Werte 149 und 255 haben,
    #  ist interessant. Es scheint, als ob es nur zwei Klassen in deinen Labels gibt.
    #  Wenn das so ist, und du ein binäres Klassifikationsproblem hast (was der Fall zu sein scheint,
    #  da du die Labels zu 0 und 1 änderst), dann könnte es sein, dass du nicht die
    #  dice_coef_per_label Funktion verwenden solltest, die für mehrere Klassen gedacht ist.   
    dice_dict = dice_coef_per_label(y, prediction, nmbr_classes)
    
    '''
    Diese Schleife berechnet die kumulative Summe der Dice-Koeffizienten für jede Klasse über alle Iterationen.
    Dies wird verwendet, um später den Durchschnitt zu berechnen. Beachten Sie, dass die Summierung bei class_nmbr = 1 beginnt,
    da class_nmbr = 0 die Hintergrundklasse repräsentiert, die in der Regel nicht von Interesse ist.
    ''' 
    if i == 0:
        avg_prec_label = dice_dict
    else:
        for class_nmbr in range(1, nmbr_classes):
            avg_prec_label[class_nmbr] = float(avg_prec_label[class_nmbr]) + float(dice_dict[class_nmbr])

    '''
    Hier wird der Dice-Koeffizient für das gesamte Bild (alle Klassen zusammen) berechnet und zur Gesamtsumme hinzugefügt.
    '''        
    print('Dice coefficient for iteration {}: {}'.format(i, dice_coef))

    if dice_coef < 0.8:  # Überprüfen, ob der Dice-Koeffizient unter 0.8 liegt
        under_80 += 1  # Wenn ja, erhöhen Sie den Zähler

    avg_prec += dice_coef
    print()

'''
Dieser Code berechnet und druckt den durchschnittlichen Dice-Koeffizienten (AP) über alle Iterationen und Klassen.
'''
#print('AP: {}'.format(avg_prec / (i+1)))
avg_prec_iter = avg_prec / (i+1)
print('Average Dice coefficient up to iteration {}: {}'.format(i, avg_prec_iter))
print('Number of times Dice coefficient under 0.8 up to iteration {}: {}'.format(i, under_80))

plt.figure(figsize=(10, 6))
plt.boxplot(dice_scores, vert=False)
plt.title('Boxplot der Dice-Koeffizienten')
plt.xlabel('Dice-Koeffizient')
plt.axvline(np.mean(dice_scores), color='r', linestyle='dashed', linewidth=2)
plt.show()

for i, (X, y) in enumerate(dataLoader["test"]):
    img = np.squeeze(X).numpy()
    img = ((img - img.min()) * (1/(img.max() - img.min()) * 255)).astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print('original')
    plt.imshow(img)
    plt.show()
    y = y.detach().squeeze().numpy()
    for i in range(1):
        if i in y:
            print(f"original {i}")
            plt.imshow(np.where(y == i, 1,0))
            plt.show()
    X = X.type('torch.FloatTensor')
    X = torch.swapaxes(X, 1, 3).swapaxes(2,3).to(device)
    prediction = model(X)
    prediction = prediction.cpu().detach().squeeze().numpy()
    prediction = prediction.argmax(axis = 0)
    for i in range(2):
        if i in prediction:
            print(f"prediction {i}")
            pic = np.where(prediction == i, 1,0)
            plt.imshow(pic)
            plt.show()
    plt.imshow(prediction, cmap= "gray")
    plt.show()
    dif = prediction - y
    dif = np.where(dif == 0, 0, 1)
    dif_f_count = dif.sum()
    print(f"False pixels {dif_f_count/(1024*1024)}")
    break