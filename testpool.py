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
        
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)
        
        with torch.no_grad():
            return self.conv(x)





example = torch.tensor([[0,2,3,2,2,1,2,1,0,1],
[2,1,7,4,4,0,1,2,1,0],
[3,4,5,3,2,1,0,1,1,3],
[0,2,3,1,0,1,0,1,1,1],
[1,1,0,0,2,0,1,0,2,0],
[1,0,0,1,1,2,2,1,0,0],
[0,1,0,1,3,7,4,3,2,1],
[0,2,1,2,3,5,9,6,2,1],
[0,0,0,2,1,2,3,2,0,0],
[2,1,0,1,0,1,2,0,1,0]])



print(example)



sp = MaskSumPool(kernel_size=3,in_channels=1)
print(sp(example.unsqueeze(0)))




custom_pool_mask = torch.tensor([[0, 0, 0, 1, 1, 1, 0, 0, 0],
                                    [0, 0, 1, 1, 1, 1, 1, 0, 0],
                                    [0, 1, 1, 1, 1, 1, 1, 1, 0],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [0, 1, 1, 1, 1, 1, 1, 1, 0],
                                    [0, 0, 1, 1, 1, 1, 1, 0, 0],
                                    [0, 0, 0, 1, 1, 1, 0, 0, 0],],dtype=torch.float32)

custom_pool_mask = torch.tensor([[0,1,0],
                                 [1,1,1],
                                 [0,1,0]],dtype=torch.float32)
                                    
sp = MaskSumPool(kernel_size=3,in_channels=1,pool_mask=custom_pool_mask)
print(sp(example.unsqueeze(0)))
