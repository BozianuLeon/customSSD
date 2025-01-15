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
        print(x.shape)
        # cyclic padding on the y-axis
        x = F.pad(x, (0, 0, self.padding, self.padding), mode='circular')

        # zero padding on the x-axis 
        x = F.pad(x, (self.padding, self.padding, 0, 0), mode='constant', value=0)

        return self.sumpool(x)


class MaskSumPool(nn.Module):
    def __init__(self, kernel_size, in_channels, stride=1, pool_mask=None):
        # Custom sum pool layer that maintains image size,
        # via custom padding (cyclic in y-axis, zeros in x-axis)
        # Masked kernel to control which pixels contribute to sum
        # concatenated with input + used as pt estimate output
        super(MaskSumPool, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2

        if pool_mask is not None:
            assert (kernel_size==pool_mask.shape[0]) and (kernel_size==pool_mask.shape[1]) 
            self.pool_mask = pool_mask
        else:
            self.pool_mask = torch.ones((kernel_size, kernel_size), dtype=torch.float32) # default

        self.conv = nn.Conv2d(
            in_channels=self.in_channels, 
            out_channels=self.in_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,  # padding handled later
            groups=self.in_channels,  # depthwise convolution, no mixing channels
            bias=False 
        )

        with torch.no_grad():
            self.pool_mask = self.pool_mask.expand(self.in_channels,-1,-1).unsqueeze(0)
            self.pool_mask = self.pool_mask.permute(1,0,2,3)
            self.conv.weight = nn.Parameter(self.pool_mask)

        # Freeze the weights so that they are not updated during backpropagation
        self.conv.weight.requires_grad = False

    def forward(self, x):
        x = x.float()  # convert to float 
        
        x = F.pad(x, (0, 0, self.padding, self.padding), mode='circular')
        x = F.pad(x, (self.padding, self.padding, 0, 0), mode='constant', value=0)
        
        with torch.no_grad():
            return self.conv(x)





if __name__ =="__main__":

    input = torch.arange(25).reshape(1,5,5)
    input = input.unsqueeze(0)
    print("Input:")
    print(input.shape)
    print(input)
    print("==========================")
    print()

    s_layer = PadSumPool(kernel_size=3)
    output = s_layer(input)
    print("Pad Sum Pool Output:")
    print(output.shape)
    print(output)
    print("==========================")
    print()


    custom_pool_mask = torch.tensor([[0, 1, 0],
                                     [1, 1, 1],
                                     [0, 1, 0]],dtype=torch.float32)
    # ms_layer = MaskSumPool(kernel_size=3, in_channels=1, stride=1, pool_mask=None)
    ms_layer = MaskSumPool(kernel_size=3, in_channels=1, stride=1, pool_mask=custom_pool_mask)
    output = ms_layer(input)
    print("Masked sum pool:")
    print("Kernel mask:", custom_pool_mask)
    print(output.shape)
    print(output)
    print("==========================")
    print()



    input1 = torch.arange(36).reshape(1,6,6)
    input2 = torch.arange(start=36,end=0,step=-1).reshape(1,6,6)
    input3 = torch.ones((1,6,6))
    input = torch.cat((input1,input2,input3)).unsqueeze(0)
    print("Multiple Channel Input:")
    print(input.shape)
    print(input)
    print("==========================")
    print()

    # ms_layer = MaskSumPool(kernel_size=3, in_channels=3, stride=1, pool_mask=None)
    ms_layer = MaskSumPool(kernel_size=3, in_channels=3, stride=1, pool_mask=custom_pool_mask)
    output = ms_layer(input)
    print("Masked sum pool:")
    print(output.shape)
    print(output)
    print("==========================")
    print()

    input1b = torch.ones((1,6,6))*2
    input2b = torch.ones((1,6,6))*3
    input3b = torch.ones((1,6,6))*4
    inputb = torch.cat((input1b,input2b,input3b))
    input = torch.cat((input1,input2,input3))
    batch = torch.stack((input,inputb),dim=0)
    print("Batched Input:")
    print(batch.shape)
    print(batch)
    print("==========================")
    print()

    # ms_layer = MaskSumPool(kernel_size=3, in_channels=3, stride=1, pool_mask=None)
    ms_layer = MaskSumPool(kernel_size=3, in_channels=3, stride=1, pool_mask=custom_pool_mask)
    output = ms_layer(batch)
    print("Masked sum pool:")
    print(output.shape)
    print(output)
    print("==========================")
    print()
