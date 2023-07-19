# imports
import glob
import sys
import os
from sklearn.metrics import confusion_matrix
import random
import math
import time
import scipy.ndimage
import numpy as np
from albumentations.pytorch import ToTensorV2
from albumentations import *
import cv2
import matplotlib.pyplot as plt
import PIL
# from utils import dice_coef_multilabel
# from imgDataset import GBMIMGDataset
from models.unet import UNet  # code borrowed from https://github.com/jvanvugt/pytorch-unet
from torch.utils.data.sampler import RandomSampler
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch
import pickle
import torch.backends.cudnn as cudnn

# from tensorboardX import SummaryWriter

# params
dataname = "GBM"
ignore_index = (
    -100
)  # Unet has the possibility of masking out pixels in the output image, we can specify the index value here (though not used)
gpuid = 0

# --- unet params
# these parameters get fed directly into the UNET class, and more description of them can be discovered there
n_classes = 2  # number of classes in the data mask that we'll aim to predict
in_channels = 3  # input channel of the data, RGB = 3
padding = True  # should levels be padded
depth = 5  # depth of the network
wf = 2  # wf (int): number of filters in the first layer is 2**wf, was 6
# should we simply upsample the mask, or should we try and learn an interpolation
up_mode = "upsample"
batch_norm = True  # should we use batch normalization between the layers

# --- training params
batch_size = 2
patch_size = 256
num_epochs = 100
edge_weight = 1.1  # edges tend to be the most poorly segmented given how little area they occupy in the training set, this paramter boosts their values along the lines of the original UNET paper
phases = ["train", "val"]
validation_phases = [
    "val"
]  # when should we do valiation? note that validation is time consuming, so as opposed to doing for both training and validation, we do it only for vlaidation at the end of the epoch

# data config
train_perc = 0.7
val_perc = 0.2

# helper function for pretty printing of current time and remaining time


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent + 0.00001)
    rs = es - s
    return "%s (- %s)" % (asMinutes(s), asMinutes(rs))


# specify if we should use a GPU (cuda) or only the CPU
if torch.cuda.is_available():
    print(torch.cuda.get_device_properties(gpuid))
    torch.cuda.set_device(gpuid)
    device = torch.device(f"cuda:{gpuid}")
    print("training runs on gpu")
else:
    print("training runs on cpu")
    device = torch.device(f"cpu")

torch.cuda.empty_cache()

# build the model according to the paramters specified above and copy it to the GPU. finally print out the number of trainable parameters
model = UNet(
    n_classes=n_classes,
    in_channels=in_channels,
    padding=padding,
    depth=depth,
    wf=wf,
    up_mode=up_mode,
    batch_norm=batch_norm,
).to(device)
print(
    f"total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")

# transforms for the data
transforms = Compose([
    VerticalFlip(p=.5),
    HorizontalFlip(p=.5),
    ToTensorV2()
])

# Dataset and Dataloader
# create random list and split data
imgPath = ""
file_names = [
    filename for filename in os.listdir(imgPath)
    if os.path.isfile(os.path.join(imgPath, filename))
]

label_list = [
    imgPath + "/labels/" + ".".join(filename.split(".")[:-1]) + "-labelled.png"
    for filename in file_names
]

img_list = [os.path.join(imgPath, file) for file in file_names]

pairs = list(zip(img_list, label_list))
random.shuffle(pairs)

split_index_train = int(len(pairs) * train_perc)
split_index_val = int(split_index_train + len(pairs) * val_perc)

# create Dataset and Dataloader
dataset = {}
dataLoader = {}
dataset['train'] = GBMIMGDataset(
    img_list=[pair[0] for pair in pairs[:split_index_train]],
    label_list=[pair[1] for pair in pairs[:split_index_train]],
    transforms=transforms
)
dataset['val'] = GBMIMGDataset(
    img_list=[pair[0] for pair in pairs[split_index_train: split_index_val]],
    label_list=[pair[1] for pair in pairs[split_index_train: split_index_val]],
    transforms=transforms
)
dataset['test'] = GBMIMGDataset(
    img_list=[pair[0] for pair in pairs[split_index_val:]],
    label_list=[pair[1] for pair in pairs[split_index_val:]]
)

testDataset = dataset['test']
# Pfad zum aktuellen Arbeitsverzeichnis
current_dir = os.getcwd()

# Pfad zur Zieldatei relativ zum aktuellen Arbeitsverzeichnis
relative_path = 'src/pklDatasets/'

# Vollständiger Pfad zur Zieldatei
file_path = os.path.join(current_dir, relative_path)

# Ordner erstellen, falls er nicht existiert
os.makedirs(os.path.dirname(file_path), exist_ok=True)

# save dataset Test in pickle file
with open('test_dataset.pkl', 'wb') as f:
    pickle.dump(testDataset, f)


for phase in phases:
    dataLoader[phase] = DataLoader(
        dataset[phase],
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    print(f"Länge des {phase} Datasets: ", len(dataset[phase]))


# adam is going to be the most robust, though perhaps not the best performing, typically a good place to start
optim = torch.optim.Adam(model.parameters())
# optim = torch.optim.SGD(model.parameters(),
#                           lr=.1,
#                           momentum=0.9,
#                           weight_decay=0.0005)

# +
# we have the ability to weight individual classes, in this case we'll do so based on their presense in the trainingset
# to avoid biasing any particular class

# nclasses = dataset["train"].numpixels.shape[1]
# class_weight=dataset["train"].numpixels[1,0:2] #don't take ignored class into account here
# class_weight = torch.from_numpy(1-class_weight/class_weight.sum()).type('torch.FloatTensor').to(device)
class_weight = None

# show final used weights, make sure that they're reasonable before continouing
# print("class_weight:", class_weight)
# reduce = False makes sure we get a 2D output instead of a 1D "summary" value
criterion = nn.CrossEntropyLoss(
    weight=class_weight, ignore_index=ignore_index, reduce=None)


# model training
writer = SummaryWriter()  # open the tensorboard visualiser
best_loss_on_test = np.Infinity
edge_weight = torch.as_tensor(edge_weight, dtype=torch.float32).to(device)
cmatrix = {key: np.zeros((2, 2)) for key in phases}
start_time = time.time()

if not os.path.exists("models"):
    os.makedirs("models")


def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.contiguous().view(-1)
    y_pred_f = y_pred.contiguous().view(-1)
    intersection = (y_true_f * y_pred_f).sum()
    return (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)


for epoch in range(num_epochs):
    # zero out epoch based performance variables
    all_loss = {key: torch.zeros(0).to(device) for key in phases}
    all_acc = {key: 0.0 for key in phases}
    # To count the number of batches in each phase
    batch_count = {key: 0 for key in phases}

    for phase in phases:  # iterate through both training and validation states
        if phase == 'train':
            model.train()  # Set model to training mode
        else:  # when in eval mode, we don't want parameters to be updated
            model.eval()   # Set model to evaluate mode

        all_loss[phase] = torch.zeros(0).to(device)

        # for each of the batches
        for ii, (X, y) in enumerate(dataLoader[phase]):
            X = X.to(device)  # [Nbatch, 3, H, W]
            # [Nbatch, H, W] with class indices (0, 1)
            y = y.type('torch.LongTensor').to(device)

            assert y.min() >= 0 and y.max() < 2

            # dynamically set gradient computation, in case of validation, this isn't needed
            with torch.set_grad_enabled(phase == 'train'):
                # disabling is good practice and improves inference time
                X = X.float()
                prediction = model(X)  # [N, Nclass, H, W]

                try:
                    loss_matrix = criterion(prediction, y)
                    # can skip if edge weight==1
                    loss = (loss_matrix * (edge_weight)).mean()
                except:
                    continue

                if phase == "train":
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    train_loss = loss

                all_loss[phase] = torch.cat(
                    (all_loss[phase], loss.detach().view(1, -1)))

                if phase in validation_phases:  # if this phase is part of validation, compute confusion matrix
                    # Get the predictions as class indices (with the highest probability)
                    _, preds = torch.max(prediction, 1)

                    dice = dice_coef(y, preds)
                    all_acc[phase] += dice.item()
                    batch_count[phase] += 1  # Increment the count

        all_loss[phase] = all_loss[phase].cpu().numpy().mean()
        # Calculate average Dice Accuracy for this phase
        if batch_count[phase] == 0:
            # Set accuracy to 0 if no batches in the phase
            all_acc[phase] = 0.0
        else:
            # Calculate average accuracy if batches exist
            all_acc[phase] /= batch_count[phase]

        # save metrics to tensorboard
        writer.add_scalar(f'{phase}/loss', all_loss[phase], epoch)
        if phase in validation_phases:
            writer.add_scalar(f'{phase}/acc', all_acc[phase], epoch)
            # writer.add_scalar(f'{phase}/TN', cmatrix[phase][0, 0], epoch)
            # writer.add_scalar(f'{phase}/TP', cmatrix[phase][1, 1], epoch)
            # writer.add_scalar(f'{phase}/FP', cmatrix[phase][0, 1], epoch)
            # writer.add_scalar(f'{phase}/FN', cmatrix[phase][1, 0], epoch)
            # writer.add_scalar(
            #     f'{phase}/TNR', cmatrix[phase][0, 0]/(cmatrix[phase][0, 0]+cmatrix[phase][0, 1]), epoch)
            # writer.add_scalar(
            #     f'{phase}/TPR', cmatrix[phase][1, 1]/(cmatrix[phase][1, 1]+cmatrix[phase][1, 0]), epoch)

    print('%s ([%d/%d] %d%%), [train loss: %.4f] [val loss: %.4f] [acc val: %.4f]' % (timeSince(start_time, (epoch+1) / num_epochs),
                                                                                      epoch+1, num_epochs, (epoch+1) / num_epochs * 100, all_loss["train"], all_loss["val"], all_acc["val"]), end="")

    # if current loss is the best we've seen, save model state with all variables
    # Bestimme den relativen Pfad zum gewünschten Verzeichnis
    save_path = os.path.join(os.getcwd(), 'models')

    # necessary for recreation
    if all_loss["val"] < best_loss_on_test:
        best_loss_on_test = all_loss["val"]
        print("  **")
        state = {'epoch': epoch + 1,
                 'model_dict': model.state_dict(),
                 'optim_dict': optim.state_dict(),
                 'best_loss_on_test': all_loss,
                 'n_classes': n_classes,
                 'in_channels': in_channels,
                 'padding': padding,
                 'depth': depth,
                 'wf': wf,
                 'up_mode': up_mode, 'batch_norm': batch_norm}

        torch.save(state, os.path.join(
            save_path, f"{dataname}_unet_best_model.pth"))
    else:
        print("")