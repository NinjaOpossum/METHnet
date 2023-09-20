import preprocessing.reader as reader

import numpy as np

class Tile(object):
    """
    A class to represent one patch/tile of a WSI

    Attributes
    ----------
    position : tuple (int, int)
        The position of the Tile in the WSI in pixel (x, y)
    size : tuple (int, int)
        The size of the tile in pixel (x, y)
    input_size : tuple (int, int)
        The expected input size of the encoder in pixel (x, y)
    augmentation : Albumentations Object
        The albumentations augmentation sequence to apply
    level : int
        The level of the image pyramid at which the Tile is extracted
    attention_values : list [float]
        A list of floats with the attention values obtained in multiple runs
    attention_values_relative : list [float]
        A list of floats with the relative attention values obtained in multiple runs
    Methods
    -------
    get_image(wsi, augment=True)
        Returns the image data belonging to the Tile
    # @MPR
    get_image_qupath(wsi)
        Returns the image from the qupath project
    # @MPR
    get_annotation(self, wsi):
        Returns the annotations as binary masks as C X W X H
    get_position()
        Getter position attribute
    get_size()
        Getter size attribute
    add_attention_value(value)
        Append a attention value to attention_values
    add_attention_value_relative(value)
        Append a relative attention value to attention_values_relative
    get_attention_values()
        Getter attention_values attribute
    get_attention_values_relative()
        Getter attention_values_relative attribute
    """
    def __init__(self, x, y, size, input_size, augmentation, level):
        """
        Parameters
        ----------
        x : int
            x position in pixel of Tile in WSI
        y : int
            y position in pixel of Tile in WSI
        size : tuple (int, int)
            size of Tile in pixel
        input_size : tuple (int, int)
            expected input_size of encoder network
        augmentation : Albumentations Object
            Augmentation sequence to apply to Tile
        level : int
            Level at which Tile is extracted from WSI
        """
        # Position of tile in WSI px
        self.position = (x,y)
        # Size of tile in px
        self.size = size
        # Input size of tile for Encoder
        self.input_size = input_size
        # Augmentation sequence
        self.augmentation = augmentation
        # Level at which tile was extracted
        self.level = level

        # Attention values for this tile
        self.attention_values = []
        # Relative attention values for this tile
        self.attention_values_relative = []

    def get_image(self, wsi, augment=True):
        """ Return Tiles image content
        Parameters
        ----------
        wsi : OpenSlide Object
            The WSI from which the image data should be extracted
        augment : bool
            Wheter to apply augmentation sequence or not
        Returns
        -------
        numpy array
            The image content belonging to the Tile
        """
        # Read image region in RGB
        img = reader.read_region(wsi, corner=self.position, size=self.size, level_downsamples=self.level).convert('RGB')
        img = np.array(img)
        
        # Check if size fits and catch to small regions
        if np.shape(img)[0] < self.input_size[0]:
            img = np.concatenate((img,np.zeros((self.size[0]-np.shape(img)[0],np.shape(img)[1],3))),axis=0)

        if np.shape(img)[1] < self.input_size[1]:
            img = np.concatenate((img,np.zeros((self.size[0],self.size[1]-np.shape(img)[1],3))),axis=1)

        # If augmentation should be performed apply augmentation sequence
        if augment:
            img = self.augmentation(image=img)['image']
        return img

    # @MPR
    def get_image_qupath(self, wsi):
        """ returns the image from the qupath project
        Args:
            wsi datastructure.wsi.WholeSlideImage: Wsi object from the patient

        Returns:
            img numpy array: tile img
        """
        qupath_project = wsi.get_qupath_project()

        if not qupath_project:
            return None

        img = None
        file_name = wsi.get_file_name()
        for image_id_current, image in enumerate(qupath_project.images):
            if image.image_name == file_name:
                image_id = image_id_current
                try:
                    img = qupath_project.get_tile(img_id = image_id, 
                        location = self.position, 
                        size = self.size,
                        downsample_level  = self.level)
                except:
                    print(file_name + " doesn't have an image in the QuPath project!")
                    return None
                break
        if img is None:
            print(file_name + " doesn't exists in the QuPath project")
        return img

    # @MPR
    def get_annotation(self, wsi):
        """ returns the annotations as binary masks as C X W X H

        Args:
            wsi datastructure.wsi.WholeSlideImage: Wsi object from the patient

        Returns:
            annotation numpy array: binary masks in different channes, shape: C X W X H
        """
        qupath_project = wsi.get_qupath_project()

        if not qupath_project:
            return None

        annotation = None
        file_name = wsi.get_file_name()
        for image_id_current, image in enumerate(qupath_project.images):
            if image.image_name == file_name:
                image_id = image_id_current
                try:
                    annotation = qupath_project.get_tile_annot_mask(img_id = image_id, 
                        location = self.position, 
                        size = self.size, 
                        downsample_level  = self.level,
                        # @MPR
                        multichannel = True,
                        class_filter = ["Tumor"])
                except:
                    print(file_name + " doesn't have an annotation in the QuPath project!")
                    return None
                break
        if annotation is None:
            print(file_name + " doesn't exists in the QuPath project")
        return annotation

    def get_position(self):
        """ Getter position attribute
        Returns
        -------
        tuple (int, int)
            The position in pixel (x, y) of the Tile
        """
        return self.position

    def get_size(self):
        """ Getter size attribute
        Returns
        -------
        tuple (int, int)
            The size in pixel (x, y) of the Tile
        """
        return self.size

    def add_attention_value(self, value):
        """ Append a value to attention value attribute
        Parameters
        ----------
        value : float
            Additional attention value
        """
        self.attention_values.append(value)

    def add_attention_value_relative(self, value):
        """ Append a value to attention value relative attribute
        Prameters
        ---------
        value : float
            Addtional relative attention value
        """
        self.attention_values_relative.append(value)

    def get_attention_values(self):
        """ Getter attention_values attribute
        Returns
        -------
        list [float]
            list of stored attention values for this Tile
        """
        return self.attention_values

    def get_attention_values_relative(self):
        """ Getter attention_values_relative attribute
        Returns
        -------
        list [float]
            list of stored relative attention values for this Tile
        """
        return self.attention_values_relative
