# # TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
import math


## Define the block of layers as a function
def conv_block(in_channels, out_channels, kernel_size, dropout_prob=0.5):
    
    return nn.Sequential(
        
        nn.Conv2d(in_channels, out_channels, kernel_size),
        nn.BatchNorm2d(out_channels),
        nn.ELU(),
        nn.Dropout(dropout_prob),
        nn.MaxPool2d(2, dilation=1)
    )

class Flatten(nn.Module):
    def forward(self, x):
        flattened = x.view(x.size(0), -1)
#         print("Shape after Flatten:", flattened.shape)  # Add this line to print the shape
        return flattened

def fc_block_1(input_size, output_size, dropout_prob=0.5):
    block = nn.Sequential(
        Flatten(),               # Flatten Layer
        nn.Linear(input_size, output_size),  # Fully connected (Dense) Layer
        nn.ELU(),                   # ELU Activation
        nn.Dropout(p=dropout_prob)  # Dropout
    )
    return block

def fc_block_2(input_size, output_size, dropout_prob=0.5):
    block = nn.Sequential(
#         Flatten(),               # Flatten Layer
        nn.Linear(input_size, output_size),  # Fully connected (Dense) Layer
#         nn.ELU(),                   # ELU Activation
        nn.Dropout(p=dropout_prob)  # Dropout
    )
    return block

def fc_block_3(input_size, output_size):
    block = nn.Sequential(
#         Flatten(),               # Flatten Layer
        nn.Linear(input_size, output_size),  # Fully connected (Dense) Layer
#         nn.ELU(),                   # ELU Activation
#         nn.Dropout(p=dropout_prob)  # Dropout
    )
    return block
#         pool = nn.MaxPool2d(2, 2)

def compute_output_width(input_width = 96, conv_kernel_size = 3, conv_stride = 1, pool_kernel_size = 2, pool_stride = 2, padding=0, dilation=1):
    
    # Output Size after Conv2D layer :
    ## https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d    
    output_width = math.floor( ((input_width - dilation*(conv_kernel_size - 1) + 2*padding -1) / conv_stride) + 1 )
    # the output Size after one pool layer :
    ## https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d
    output_width = math.floor( ((output_width - dilation*(pool_kernel_size - 1) + 2*padding -1) / pool_stride) + 1 )
    
    
    return output_width

# +
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
#         self.conv1 = nn.Conv2d(1, 32, 5

        #         input_width = 96
        input_width = 224
        
        conv2D_depth_in = 1
        conv2D_depth_out = 32                        
        self.conv1 = conv_block(conv2D_depth_in, conv2D_depth_out, 3, 0.1)
        output_width = compute_output_width(input_width)
#         print("conv2D_depth_out after 1st conv block = "+str(conv2D_depth_out))
#         print("output_width after 1st conv block = "+str(output_width))
        
        conv2D_depth_in = conv2D_depth_out
        conv2D_depth_out = 2*conv2D_depth_out        
        self.conv2 = conv_block(conv2D_depth_in, conv2D_depth_out, 3, 0.2)
        output_width = compute_output_width(output_width)
#         print("conv2D_depth_out after 2nd conv block = "+str(conv2D_depth_out))
#         print("output_width after 2nd conv block = "+str(output_width))
        
        conv2D_depth_in = conv2D_depth_out
        conv2D_depth_out = 2*conv2D_depth_out        
        self.conv3 = conv_block(conv2D_depth_in, conv2D_depth_out, 3, 0.3)
        output_width = compute_output_width(output_width)
#         print("conv2D_depth_out after 3rd conv block = "+str(conv2D_depth_out))
#         print("output_width after 3rd conv block = "+str(output_width))
        
        conv2D_depth_in = conv2D_depth_out
        conv2D_depth_out = 2*conv2D_depth_out        
        self.conv4 = conv_block(conv2D_depth_in, conv2D_depth_out, 3, 0.4)
        output_width = compute_output_width(output_width)
#         print("conv2D_depth_out after 4th conv block = "+str(conv2D_depth_out))
#         print("output_width after 4th conv block = "+str(output_width))


#         input_size = conv2D_depth*output_width*output_width
#         print("input_size = "+str(input_size))
        
        self.fc1 = fc_block_1(conv2D_depth_out*output_width*output_width , 1000, 0.5)
        
        self.fc2 = fc_block_2(1000 , 1000, 0.6)
        
        self.fc3 = fc_block_3(1000 , 2*68)

        self.last_conv_block_depth = conv2D_depth_out
        
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
    
# -



#         input_size = conv2D_depth*output_width*output_width
#         print("input_size = "+str(input_size))
#         self.fc1 = fc_block_1(conv2D_depth_out*output_width*output_width , 1000, 0.5)

        
    
        
        
        
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.conv1(x)
#         print("self.conv1(x).size() = "+str(x.size()))
        x = self.conv2(x)
#         print("self.conv2(x).size() = "+str(x.size()))
        x = self.conv3(x)
#         print("self.conv3(x).size() = "+str(x.size()))
        x = self.conv4(x)
#         print("self.conv4(x).size() = "+str(x.size()))

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

#         print("self.conv4.shape = "+str(self.conv4.shape))
#         x = fc_block_1(self.last_conv_block_depth*self.conv4.size()[2]*self.conv4.size()[3] , 1000, 0.5)(x)
#         x = fc_block_2(1000 , 1000, 0.6)(x)
#         x = fc_block_3(1000 , 2)(x)


        # a modified x, having gone through all the layers of your model, should be returned
        return x
