import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision




class SumPool(torch.nn.Module):
    def __init__(self, kernel_size, stride=1):
        super(SumPool, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.sumpool = torch.nn.AvgPool2d(kernel_size=kernel_size, stride=stride, divisor_override=1)

    def forward(self, x):
        return self.sumpool(x)


class PadSumPool(torch.nn.Module):
    def __init__(self, kernel_size, stride=1):
        super(PadSumPool, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        self.sumpool = torch.nn.AvgPool2d(kernel_size=kernel_size, stride=stride, divisor_override=1)

    def forward(self, x):
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)
        return self.sumpool(x)


class CustomPadSumPool(torch.nn.Module):
    def __init__(self, kernel_size, stride=1):
        # Custom sum pool layer that maintains image size,
        # via custom padding (cyclic in y-axis, zeros in x-axis)
        # Summation of all pixels in 4x4 square 
        # concatenated with input + used as pt estimate output
        super(CustomPadSumPool, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2

        self.sumpool = torch.nn.AvgPool2d(kernel_size=kernel_size, stride=stride, divisor_override=1)

    def forward(self, x):
        # cyclic padding on the y-axis
        x = F.pad(x, (0, 0, self.padding, self.padding), mode='circular')

        # zero padding on the x-axis 
        x = F.pad(x, (self.padding, self.padding, 0, 0), mode='constant', value=0)
        # print("Padded Input!\n",x)
        return self.sumpool(x)



class MaskSumPool(nn.Module):
    def __init__(self, kernel_size, in_channels, stride=1, pool_mask=None):
        super(MaskSumPool, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2

        if pool_mask is not None:
            self.pool_mask = pool_mask
        else:
            # self.pool_mask = torch.ones((kernel_size, kernel_size)) # default
            self.pool_mask = torch.ones((kernel_size, kernel_size), dtype=torch.float32)
        print(self.pool_mask.shape)
        self.pool_mask = self.pool_mask.unsqueeze(0).expand(self.in_channels, self.in_channels, self.in_channels)
        print(self.pool_mask.shape)
        self.pool_mask = self.pool_mask.view(1, self.in_channels, kernel_size, kernel_size)
        print(self.pool_mask.shape)

        # Create a custom convolutional layer (using groups to handle multiple channels)
        self.conv = nn.Conv2d(
            in_channels=self.in_channels, 
            out_channels=self.in_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,  # padding handled later
            groups=1,  # Don't perform multi-channel convolution
            bias=False 
        )

        # Set the convolution weights to the custom pooling mask
        with torch.no_grad():
            self.conv.weight = nn.Parameter(self.pool_mask.view(1, self.in_channels, kernel_size, kernel_size))

        # Freeze the weights so that they are not updated during backpropagation
        self.conv.weight.requires_grad = False

    def forward(self, x):
        x = x.float()  # convert to float 
        # x = F.pad(x, (0, 0, self.padding, self.padding), mode='circular')
        x = F.pad(x, (0, 0, self.padding, self.padding), mode='constant', value=0)
        x = F.pad(x, (self.padding, self.padding, 0, 0), mode='constant', value=0)
        with torch.no_grad():
            return self.conv(x)



# class ConvSumPool(nn.Module):
#     def __init__(self, kernel_size, stride=1, padding=0):
#         super(ConvSumPool, self).__init__()
        
#         # Define a convolution layer with kernel_size filled with 1's (sum pooling)
#         self.conv = nn.Conv2d(
#             in_channels=1,  # assume grayscale input
#             out_channels=1,  # single output channel
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#             bias=False  # No bias is needed for sum pooling
#         )
        
#         # Initialize the weights to be all 1's (sum pooling kernel)
#         with torch.no_grad():
#             self.conv.weight.fill_(1)

#     def forward(self, x):
#         x = x.float()  # Convert to float if not already
#         # Apply the sum pooling operation using convolution
#         return self.conv(x)





if __name__ =="__main__":

    # input = torch.arange(81).reshape(1,9,9)
    input = torch.arange(25).reshape(1,5,5)
    input = input.unsqueeze(0)
    # input = torch.randint(10,size=(1,10,10))
    # input = F.pad(input, (0, 0, 2, 2), mode='circular')
    print("Input:")
    print(input.shape)
    print(input)
    print()
    print("Output:")
    s_layer = PadSumPool(kernel_size=3)
    output = s_layer(input)
    print(output.shape)
    print(output)

    s_layer = CustomPadSumPool(kernel_size=3)
    output = s_layer(input)
    print(output.shape)
    print(output)
    print()
    print()
    print()
    print("Masked sum pool")
    pool_mask = torch.tensor([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]])
    # ms_layer = MaskSumPool(kernel_size=3, stride=1, pool_mask=None)
    ms_layer = MaskSumPool(kernel_size=3, in_channels=1, stride=1, pool_mask=None)
    output = ms_layer(input)
    print(output.shape)
    print(output)
    print()
    print()
    print()

    input1 = torch.arange(36).reshape(1,6,6)
    input2 = torch.arange(start=36,end=0,step=-1).reshape(1,6,6)
    input = torch.cat((input1,input2)).unsqueeze(0)
    print(input.shape)
    print(input)
    print()
    print()
    ms_layer = MaskSumPool(kernel_size=3, in_channels=2, stride=1, pool_mask=None)
    output = ms_layer(input)
    print(output.shape)



