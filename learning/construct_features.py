import torch

from models.restnet_custom import resnet50_baseline
from progress.bar import IncrementalBar
import os
import numpy as np
from torch.utils.data import SequentialSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

class Patient_Dataset(Dataset):
    """
    A class which inherits Dataset to use DataLoader

    Attributes
    ----------
    X : WholeSlideImage Object
        The Whole Slide Image Object for this patient
    i : int
        The Tile Property index for this WSI to use

    Methods
    -------
    __len__()
        Number of items to process - number of Tiles
    __getitem__(idx)
        Return current image
    """
    def __init__(self, X, i):
        """
        X : WholeSlideImage Object
            The Whole Slide Image Object for this patient
        i : int
            The Tile Property index for this WSI to use
        """
        # WSI
        self.X = X
        # Tile property index
        self.i = i

    def __len__(self):
        """ Return number of items to process

        Returns
        -------
        int 
            Number of items
        """
        # Number of available tiles
        return len(self.X.get_tiles_list()[self.i])

    def __getitem__(self, idx):
        """ Return Tile Image at idx

        Parameters
        ----------
        idx : int
            Index of Tile
        
        Returns
        -------
        TorchTensor
            The Tiles image casted to TorchTensor
        """
        # Get augmented image at index
        a = self.X.get_tiles_list()[self.i][idx].get_image(self.X.get_image(), True)
        # Transpose for channel first
        a = np.transpose(a, (2, 0, 1))
        # Cast to float
        a = a.astype(np.float32)
        # Check if has three dimensions
        if len(np.shape(a)) == 3:
            a = np.expand_dims(a, axis=0)
        # To torch tensor
        all_p = torch.from_numpy(a)
        
        return all_p

def collate_features(batch):
    """ Concatenate batch as TorchTensor

    Parameters
    ----------
    batch : list [TorchTensor]
        Batch of images 

    Returns
    -------
    TorchTensor
        Concatenated batch
    """
    # concatenate batch
    img = torch.cat([item for item in batch], dim=0)

    return img

def construct_features(patients, setting):
    """ Create features for patients and save them
    Parameters
    ----------
    patients : list [[Patient]]
        list of list per class of Patient to create features for
    setting : Setting
        Setting as defined by class
    """

    # Get feature setting
    feature_setting = setting.get_feature_setting()
    # Get model - default pretrained ResNet50
    model = resnet50_baseline(pretrained=True)
    # To GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    # Evaluation mode
    model.eval()
    # Number of patients to process
    n_patients = sum([len(p) for p in patients])
    #Track progress
    bar = IncrementalBar('Create features for patients ', max=n_patients)
    torch.backends.cudnn.enabled = False
    # Iterate patient classes
    for patient in patients:
        # Iterate patients in class
        for p in patient:
            # Iterate image properties
            for wp in p.get_wsis():
                # Iterate WSIs with image property
                for wsi in wp:
                    # Iterate Tile properties
                    for i in range(len(wsi.get_tile_properties())):
                        # If skip exisiting and already existing continue
                        if feature_setting.get_skip_existing() and os.path.exists(wsi.get_feature_file_names()[i]):
                            continue
                        # If no tiles then nothing to be done
                        if len(wsi.get_tiles_list()) != 0:
                            # Open WSI
                            wsi.load_wsi()
                            # Create Dataset
                            data = Patient_Dataset(wsi, i)
                            # Create Dataloader
                            loader = DataLoader(dataset=data, batch_size=feature_setting.get_batch_size(), collate_fn=collate_features, sampler=SequentialSampler(data))
                            # patient features for tile property
                            patient_features = np.zeros((0, feature_setting.get_feature_dimension()))
                            # Pass batches
                            for batch in loader:
                                with torch.no_grad():
                                    batch = batch.to(device, non_blocking=True)
                                    # Compute features
                                    features = model(batch)
                                    features = features.cpu().numpy()
                                    # Concatenate features
                                    patient_features = np.concatenate((patient_features, features), axis=0)

                            # Close WSI
                            wsi.close_wsi()
                        

                        # Create directory to save features and keys
                        feature_directory = {}
                        # Fill directory
                        for j, tile in enumerate(wsi.get_tiles_list()[i]):
                            # Key Tile position, value feature of Tile
                            feature_directory[tile.get_position()] = patient_features[j]
                        # Save features
                        wsi.set_features(feature_directory, i)
            bar.next()
    bar.finish()