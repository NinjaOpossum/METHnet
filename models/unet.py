# unet.py
# Adapted from https://discuss.pytorch.org/t/unet-implementation/426
# Code borrowd from: https://github.com/jvanvugt/pytorch-unet/blob/master/unet.py

"""
- In der Klasse `UNet` wird das eigentliche UNet-Modell implementiert.
- Die Methode `__init__` initialisiert das Modell mit den übergebenen Parametern. Hierbei wird die Tiefe des Modells, die Anzahl der Filter im ersten Layer,
    die Anzahl der Eingangskanäle und die Anzahl der Ausgangskanäle festgelegt. Es werden auch verschiedene Optionen festgelegt, wie z. B. ob eine Polsterung angewendet werden soll,
    Batch-Normalisierung verwendet werden soll und welcher Up-Sampling-Modus verwendet werden soll (upconv oder upsample).
- In der `forward` Methode wird eine Vorwärtsberechnung durch das Modell durchgeführt. Zunächst wird das Eingangsbild durch den Abwärtsweg des Modells geschickt,
    bei dem es schrittweise durch verschiedene Konvolutionsblöcke und Pooling-Schichten verkleinert wird. Die Ausgänge jeder Konvolutionsoperation werden aufbewahrt,
    um später im Aufwärtsweg verwendet zu werden.
- Im Aufwärtsweg wird das Eingangsbild durch verschiedene Up-Sampling-Blöcke und Konvolutionsblöcke geleitet, um eine Segmentierungskarte zu erzeugen.
- In der Klasse `UNetConvBlock` wird ein Konvolutionsblock definiert, der in der `down_path` und `up_path` des UNet-Modells verwendet wird.
- Die `forward` Methode der Klasse `UNetConvBlock` führt eine Vorwärtsberechnung durch den Konvolutionsblock durch.
- In der Klasse `UNetUpBlock` wird ein Up-Sampling-Block definiert, der in der `up_path` des UNet-Modells verwendet wird.
- Die `forward` Methode der Klasse `UNetUpBlock` führt eine Vorwärtsberechnung durch den Up-Sampling-Block durch. Dabei wird das Eingangsbild durch eine Up-Sampling-Operation auf
    eine größere Größe gebracht und dann mit dem entsprechenden Block aus dem Abwärtsweg konkateniert. Anschließend wird ein Konvolutionsblock auf die konkatenierten Ausgaben angewendet,
    um eine endgültige Ausgabe zu erzeugen.
"""

import torch
from torch import nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, depth=5, wf=6, padding=False,
                 batch_norm=False, up_mode='upconv'):
        """
        Initializes the UNet model with given parameters.

        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()

        # Check that up_mode is valid
        assert up_mode in ('upconv', 'upsample')

        self.padding = padding
        self.depth = depth

        # Number of channels in the first layer is 2**wf
        prev_channels = in_channels

        # Create the down-sampling path of the network
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2**(wf+i),
                                                padding, batch_norm))
            prev_channels = 2**(wf+i)

        # Create the up-sampling path of the network
        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2**(wf+i), up_mode,
                                            padding, batch_norm))
            prev_channels = 2**(wf+i)

        # Final convolution layer
        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        """
        Computes a forward pass through the UNet model.

        Args:
            x (tensor): input tensor

        Returns:
            output tensor
        """
        blocks = []

        # Pass the input through the down-sampling path of the network
        for i, down in enumerate(self.down_path):
            x = down(x)

            # Save the output of each down-sampling block for the up-sampling path
            if i != len(self.down_path)-1:
                blocks.append(x)

                # Down-sample the output of the block to pass it to the next block
                x = F.avg_pool2d(x, 2)

        # Pass the output of the last down-sampling block through the up-sampling path
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])

        # Pass the output of the last up-sampling block through the final convolution layer
        return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        """
        Initializes a convolutional block in the UNet model.

        Args:
            in_size (int): size of input tensor
            out_size (int): size of output tensor
            padding (bool): whether to apply padding
            batch_norm (bool): whether to apply batch normalization
        """
        super(UNetConvBlock, self).__init__()

        # Create a list of layers for the convolutional block
        block = []

        # Add the first convolutional layer to the block
        block.append(nn.Conv2d(in_size, out_size, kernel_size=3,
                               padding=int(padding)))

        # Add a ReLU activation function after the first convolutional layer
        block.append(nn.ReLU())

        # Add batch normalization after the first activation function (optional)
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        # Add the second convolutional layer to the block
        block.append(nn.Conv2d(out_size, out_size, kernel_size=3,
                               padding=int(padding)))

        # Add a ReLU activation function after the second convolutional layer
        block.append(nn.ReLU())

        # Add batch normalization after the second activation function (optional)
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        # Create a sequential module with the layers in the block list
        self.block = nn.Sequential(*block)

    def forward(self, x):
        """
        Computes a forward pass through the convolutional block.

        Args:
            x (tensor): input tensor

        Returns:
            output tensor
        """
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        """
        Initializes an upsample block in the UNet model.

        Args:
            in_size (int): size of input tensor
            out_size (int): size of output tensor
            up_mode (str): type of upsampling to use
            padding (bool): whether to apply padding
            batch_norm (bool): whether to apply batch normalization
        """
        super(UNetUpBlock, self).__init__()

        # Set up the upsampling operation based on the up_mode parameter
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,
                                         stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(in_size, out_size, kernel_size=1))

        # Create a convolutional block to be applied after the upsampling
        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        """
        Crops the center of a layer to a specified target size.

        Args:
            layer (tensor): input tensor to crop
            target_size (tuple): target size of crop

        Returns:
            cropped tensor
        """

        # Get the height and width of the input layer
        _, _, layer_height, layer_width = layer.size()

        # Calculate the difference between the input layer's height/width and the target height/width
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2

        # Perform the crop
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        """
        Computes a forward pass through the upsample block.

        Args:
            x (tensor): input tensor
            bridge (tensor): output tensor from corresponding downsample block

        Returns:
            output tensor
        """

        # Perform the upsampling operation
        up = self.up(x)

        # Crop the output of the corresponding downsample block
        crop1 = self.center_crop(bridge, up.shape[2:])

        # Concatenate the upsampled tensor and the cropped tensor
        out = torch.cat([up, crop1], 1)

        # Apply the convolutional block
        out = self.conv_block(out)

        # Apply softmax activation
        # out = F.softmax(out, dim=1)

        return out
