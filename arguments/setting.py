from arguments.data_setting import DataSetting as DataSetting
from arguments.class_setting import ClassSetting as ClassSetting
# from arguments.network_setting import NetworkSetting as NetworkSetting
from arguments.network_setting_new import NetworkSettingNew as NetworkSetting
from arguments.feature_setting import FeatureSetting as FeatureSetting

class Setting(object):
    """ Class holding the different sub setting files holding all Parameters 

    Attributes
    ----------
    data_setting : DataSetting
        Parameters for data
    class_setting : ClassSetting
        Parameters for classification problem
    network_setting : NetworkSetting
        Parameters for model and training
    feature_setting : FeatureSetting
        Parameters for encoded features

    Methods
    -------
    get_data_setting()
        Return data_setting
    get_class_setting()
        Return class_setting
    get_network_setting()
        Return network_setting
    get_feature_setting()
        Return feature_setting
    """
    # @MPR
    def __init__(self, data_directories, csv_file, working_directory, qupath_project_file = None):
        """
        Parameters
        ----------
        data_directories : list [string]
            Directories where WSIs are stored
        csv_file : string
            CSV file where patient information is stored
        working_directory : string
            Working folder where files should be stored
        """
        # @MPR
        self.data_setting = DataSetting(data_directories, csv_file, working_directory, qupath_project_file)
        self.class_setting = ClassSetting()
        self.network_setting = NetworkSetting(working_directory)
        self.feature_setting = FeatureSetting()

    def get_data_setting(self):
        """ Return data_setting
        Returns
        -------
        DataSetting
            The DataSetting
        """
        return self.data_setting

    def get_class_setting(self):
        """ Return class_setting
        Returns
        -------
        ClassSetting
            The ClassSetting
        """
        return self.class_setting

    def get_network_setting(self):
        """ Return network_setting
        Returns
        -------
        NetworkSetting
            The NetworkSetting
        """
        return self.network_setting

    def get_feature_setting(self):
        """ Return feature_setting
        Returns
        -------
        FeatureSetting
            The FeatureSetting
        """
        return self.feature_setting