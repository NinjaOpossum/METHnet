# @MPR

from utils import helper

class NetworkSettingNew(object):

    def __init__(self, working_directory):
        """
        Parameters
        ----------
        working_directory : string
            Working directory for data
        """

        # Model Parameters
        self.in_channels = 3
        # n_classes will be retrieved from the class_setting
        self.depth = 5
        self.wf = 6
        self.batch_norm = True
        self.up_mode='upconv'

        # Training Parameters
        self.runs = 1
        self.epochs = 5
        self.patches_per_epoch = 30
        self.validation_patch_limit = 30
        self.batch_size = 6

        # Logging Options
        self.run_identifier = "Pathology"

    def get_in_channels(self):
        return self.in_channels

    def get_depth(self):
        return self.depth
    
    def get_wf(self):
        return self.wf
    
    def get_batch_norm(self):
        return self.batch_norm
    
    def get_up_mode(self):
        return self.up_mode
    
    def get_F(self):
        return 1024

        #Model hyper_parameters

        # helper.create_folder(self.model_folder)

    def get_runs(self):
        return self.runs
    
    def get_epochs(self):
        return self.epochs
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_patches_per_epoch(self):
        return self.patches_per_epoch
    
    def get_validation_patch_limit(self):
        return self.validation_patch_limit
    
    def get_run_identifier(self):
        return self.run_identifier
