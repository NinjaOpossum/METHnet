# utils.py

# @MPR

import numpy as np
import torch


def dice_coef(y_true, y_pred, smooth=1e-6):
    '''
    returns the dice coefficient for y_true and y_pred
    y_true and y_pred must be one label
    '''
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    y_sum = np.sum(y_true_f) + np.sum(y_pred_f)
    return (2 * intersection + smooth) / (y_sum + smooth)

    # y_true_f = y_true.contiguous().view(-1)
    # y_pred_f = y_pred.contiguous().view(-1)
    # intersection = (y_true_f * y_pred_f).sum()
    # return (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)


def dice_coef_multilabel(y_true, y_pred, numLabels):
    '''
    returns the dice coefficient for y_true and y_pred for multilabels
    '''
    dice = 0
    for i in range(1, numLabels):
        y_true_norm = np.where(y_true == i, 1, 0).astype(np.uint8)
        y_pred_norm = np.where(y_pred == i, 1, 0).astype(np.uint8)
        dice += dice_coef(y_true_norm, y_pred_norm)
        
    return dice / numLabels


def dice_coef_per_label(y_true, y_pred, numLabels):
    '''
    returns the dice coefficient for y_true and y_pred for each label
    '''
    label_dice_dict = {}
    for i in range(1, numLabels):
        y_true_norm = np.where(y_true == i, 1, 0).astype(np.uint8)
        y_pred_norm = np.where(y_pred == i, 1, 0).astype(np.uint8)
        label_dice_dict[i] = dice_coef(y_true_norm, y_pred_norm)
    return label_dice_dict

def get_precision_recall(pred_index, annotations, n_classes):
     """
    Compute the average precision and recall for a multi-class classification.
    
    Parameters:
    - pred_index (array-like): Model predictions for each sample.
    - annotations (array-like): True class labels for each sample.
    - n_classes (int): Total number of classes.
    
    Returns:
    - mean_precision (float): Average precision over all classes (except class 0).
    - mean_recall (float): Average recall over all classes (except class 0).
    
    Note:
    - This function assumes that class 0 is a background or ignore class and does not
      compute precision or recall for it.
    """
    precision = 0
    recall = 0
    for i in range(1, n_classes):
        pred_index_class_bin = (pred_index == i).astype(int) 
        annotations_class_bin = (annotations == i).astype(int)
        tp = ((pred_index_class_bin == 1) * pred_index_class_bin == annotations_class_bin).sum()
        fp = ((pred_index_class_bin == 1) * pred_index_class_bin != annotations_class_bin).sum()
        fn = ((pred_index_class_bin == 0) * pred_index_class_bin != annotations_class_bin).sum()
        precision += tp / (tp + fp)
        recall += tp / (tp + fn)
    mean_precision = precision / (n_classes - 1)
    mean_recall = recall / (n_classes - 1)
    return mean_precision, mean_recall

class CompositeLoss(torch.nn.Module):
    def __init__(self):
        super(CompositeLoss, self).__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()

    def multi_class_dice_loss(self, inputs, targets, smooth=1e-3):
        # inputs are logits, targets are binary labels
        probs = torch.sigmoid(inputs)
        inse = torch.sum(probs * targets, (0, 2, 3))
        l = torch.sum(probs, (0, 2, 3))
        r = torch.sum(targets, (0, 2, 3))
        loss = 1.0 - (2.0 * inse + smooth) / (l + r + smooth)
        loss = torch.mean(loss)
        return loss

    def forward(self, inputs, targets):
        loss = self.bce(inputs, targets) + self.multi_class_dice_loss(inputs, targets)
        return loss

