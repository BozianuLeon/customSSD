import torch
import torch.nn as nn
import torchvision


class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        # Custom, simple implementation of 2d layernorm, 
        # based on https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html but permutes 
        # channels, reducing number of learnable params
        # see also https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html
        super(LayerNorm2d, self).__init__()
        self.dim = dim
        self.eps = eps
        self.ln = nn.LayerNorm(dim, eps=self.eps) 

    def forward(self, x):
        # First, permute to (batch, height, width, channels)
        x = x.permute(0, 2, 3, 1)
        # Apply LayerNorm over the last dimension (channels)
        x = self.ln(x)
        # Permute back to (batch, channels, height, width)
        x = x.permute(0, 3, 1, 2)
        return x
    


class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layer_scale=1e-6):
        # Simple implementation of convnext block. See https://arxiv.org/abs/2201.03545 
        # Utilising custom layernorm (taken from https://pytorch.org/vision/main/_modules/torchvision/models/convnext.html)
        # See also https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py 
        # for alternative implementation details
        super(ConvNeXtBlock, self).__init__()
        self.gelu = nn.GELU()

        self.conv_d7x7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=3, groups=in_channels) #depthwise conv
        # self.ln = nn.LayerNorm([out_channels, feat_map_size, feat_map_size]) # increases numb params
        self.ln = LayerNorm2d(out_channels)
        self.conv_1x1_1 = nn.Conv2d(out_channels, out_channels*4, kernel_size=1, stride=1, padding=0, bias=True)
        self.gelu = nn.GELU()
        self.conv_1x1_2 = nn.Conv2d(out_channels*4, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.layer_scale = nn.Parameter(torch.ones(in_channels, 1, 1) * layer_scale)

    def forward(self, input):

        out = self.conv_d7x7(input)
        out = self.ln(out)
        
        out = self.conv_1x1_1(out)
        out = self.gelu(out)

        out = self.layer_scale * self.conv_1x1_2(out)

        ret = out + input
        return ret




class ConvNeXt_T(nn.Module):
    def __init__(self, num_channels=3, num_classes=1000):
        super(ConvNeXt_T, self).__init__()
        
        self.gelu = nn.GELU()
        # STEM CELL BLOCK (112,112)->(56,56)
        self.conv1 = nn.Conv2d(num_channels, 96, kernel_size=4, stride=4, padding=0, bias=True)
        # self.ln1 = nn.LayerNorm([96, 56, 56])
        self.ln1 = LayerNorm2d(96)

        # BLOCK-2 (56,56)
        self.conv2_1 = ConvNeXtBlock(in_channels=96, out_channels=96)
        self.conv2_2 = ConvNeXtBlock(in_channels=96, out_channels=96)
        self.conv2_3 = ConvNeXtBlock(in_channels=96, out_channels=96)

        # in-between res2->res3 down sampling (56,56)->(28,28), channels 96->192
        # self.down_ln2 = nn.LayerNorm([96, 56, 56])
        self.down_ln2 = LayerNorm2d(96)
        self.down_res2 = nn.Conv2d(96, 192, kernel_size=2, stride=2)

        # BLOCK-3 (28,28)
        self.conv3_1 = ConvNeXtBlock(in_channels=192, out_channels=192)
        self.conv3_2 = ConvNeXtBlock(in_channels=192, out_channels=192)
        self.conv3_3 = ConvNeXtBlock(in_channels=192, out_channels=192)

        # in-between res3->res4 down sampling (28,28)->(14,14)
        # self.down_ln3 = nn.LayerNorm([192, 28, 28])
        self.down_ln3 = LayerNorm2d(192)
        self.down_res3 = nn.Conv2d(192, 384, kernel_size=2, stride=2) 

        # BLOCK-4 (14,14)
        self.conv4_1 = ConvNeXtBlock(in_channels=384, out_channels=384)
        self.conv4_2 = ConvNeXtBlock(in_channels=384, out_channels=384)
        self.conv4_3 = ConvNeXtBlock(in_channels=384, out_channels=384)
        self.conv4_4 = ConvNeXtBlock(in_channels=384, out_channels=384)
        self.conv4_5 = ConvNeXtBlock(in_channels=384, out_channels=384)
        self.conv4_6 = ConvNeXtBlock(in_channels=384, out_channels=384)
        self.conv4_7 = ConvNeXtBlock(in_channels=384, out_channels=384)
        self.conv4_8 = ConvNeXtBlock(in_channels=384, out_channels=384)
        self.conv4_9 = ConvNeXtBlock(in_channels=384, out_channels=384)

        # in-between res4->res5 down sampling (14,14)->(7,7)
        # self.down_ln4 = nn.LayerNorm([384, 14, 14])
        self.down_ln4 = LayerNorm2d(384)
        self.down_res4 = nn.Conv2d(384, 768, kernel_size=2, stride=2) 

        # BLOCK-5 (7,7)
        self.conv5_1 = ConvNeXtBlock(in_channels=768, out_channels=768)
        self.conv5_2 = ConvNeXtBlock(in_channels=768, out_channels=768)
        self.conv5_3 = ConvNeXtBlock(in_channels=768, out_channels=768)

        #global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        # self.ln6 = nn.LayerNorm([768, 1, 1])
        self.ln6 = LayerNorm2d(768)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(768,num_classes)

    def forward(self,x):
        #stem
        out = self.gelu(self.ln1(self.conv1(x)))

        #conv2
        out = self.conv2_1(out)
        out = self.conv2_2(out)
        out = self.conv2_3(out)

        out = self.down_res2(self.down_ln2(out))

        #conv3
        out = self.conv3_1(out)        
        out = self.conv3_2(out)        
        out = self.conv3_3(out)        

        out = self.down_res3(self.down_ln3(out))

        #conv4
        out = self.conv4_1(out)        
        out = self.conv4_2(out)        
        out = self.conv4_3(out)        
        out = self.conv4_4(out)        
        out = self.conv4_5(out)        
        out = self.conv4_6(out)        
        out = self.conv4_7(out)        
        out = self.conv4_8(out)        
        out = self.conv4_9(out)        

        out = self.down_res4(self.down_ln4(out))

        #conv5
        out = self.conv5_1(out)       
        out = self.conv5_2(out)       
        out = self.conv5_3(out)       

        #output
        out = self.avg_pool(out)
        out = self.ln6(out)
        out = self.fc(self.flat(out))
        
        return out



class ConvNeXt50(nn.Module):
    def __init__(self, num_channels=3, num_classes=1000):
        # Custom convnext50, with stride=1 updated to fit
        # the shape of detector images. Necessary due to the rectangular shape
        # of the "images" (125x96)
        super(ConvNeXt50, self).__init__()
        
        self.gelu = nn.GELU()
        # STEM CELL BLOCK (112,112)->(56,56)
        self.conv1 = nn.Conv2d(num_channels, 96, kernel_size=4, stride=4, padding=0, bias=True)
        # self.ln1 = nn.LayerNorm([96, 56, 56])
        self.ln1 = LayerNorm2d(96)

        # BLOCK-2 (56,56)
        self.conv2_1 = ConvNeXtBlock(in_channels=96, out_channels=96)
        self.conv2_2 = ConvNeXtBlock(in_channels=96, out_channels=96)
        self.conv2_3 = ConvNeXtBlock(in_channels=96, out_channels=96)

        # in-between res2->res3 down sampling (56,56)->(28,28), channels 96->192
        # self.down_ln2 = nn.LayerNorm([96, 56, 56])
        self.down_ln2 = LayerNorm2d(96)
        self.down_res2 = nn.Conv2d(96, 192, kernel_size=1, stride=2) #should be kernel size 2

        # BLOCK-3 (28,28)
        self.conv3_1 = ConvNeXtBlock(in_channels=192, out_channels=192)
        self.conv3_2 = ConvNeXtBlock(in_channels=192, out_channels=192)
        self.conv3_3 = ConvNeXtBlock(in_channels=192, out_channels=192)

        # in-between res3->res4 down sampling (28,28)->(14,14)
        # self.down_ln3 = nn.LayerNorm([192, 28, 28])
        self.down_ln3 = LayerNorm2d(192)
        self.down_res3 = nn.Conv2d(192, 384, kernel_size=1, stride=2) #should be kernel size 2

        # BLOCK-4 (14,14)
        self.conv4_1 = ConvNeXtBlock(in_channels=384, out_channels=384)
        self.conv4_2 = ConvNeXtBlock(in_channels=384, out_channels=384)
        self.conv4_3 = ConvNeXtBlock(in_channels=384, out_channels=384)
        self.conv4_4 = ConvNeXtBlock(in_channels=384, out_channels=384)
        self.conv4_5 = ConvNeXtBlock(in_channels=384, out_channels=384)
        self.conv4_6 = ConvNeXtBlock(in_channels=384, out_channels=384)
        self.conv4_7 = ConvNeXtBlock(in_channels=384, out_channels=384)
        self.conv4_8 = ConvNeXtBlock(in_channels=384, out_channels=384)
        self.conv4_9 = ConvNeXtBlock(in_channels=384, out_channels=384)

        # in-between res4->res5 down sampling (14,14)->(7,7)
        # self.down_ln4 = nn.LayerNorm([384, 14, 14])
        self.down_ln4 = LayerNorm2d(384)
        self.down_res4 = nn.Conv2d(384, 768, kernel_size=1, stride=2) #should be kernel size 2

        # BLOCK-5 (7,7)
        self.conv5_1 = ConvNeXtBlock(in_channels=768, out_channels=768)
        self.conv5_2 = ConvNeXtBlock(in_channels=768, out_channels=768)
        self.conv5_3 = ConvNeXtBlock(in_channels=768, out_channels=768)

        #global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        # self.ln6 = nn.LayerNorm([768, 1, 1])
        self.ln6 = LayerNorm2d(768)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(768,num_classes)

    def forward(self,x):
        #stem
        out = self.gelu(self.ln1(self.conv1(x)))

        #conv2
        out = self.conv2_1(out)
        out = self.conv2_2(out)
        out = self.conv2_3(out)

        out = self.down_res2(self.down_ln2(out))

        #conv3
        out = self.conv3_1(out)        
        out = self.conv3_2(out)        
        out = self.conv3_3(out)        

        out = self.down_res3(self.down_ln3(out))

        #conv4
        out = self.conv4_1(out)        
        out = self.conv4_2(out)        
        out = self.conv4_3(out)        
        out = self.conv4_4(out)        
        out = self.conv4_5(out)        
        out = self.conv4_6(out)        
        out = self.conv4_7(out)        
        out = self.conv4_8(out)        
        out = self.conv4_9(out)        

        out = self.down_res4(self.down_ln4(out))

        #conv5
        out = self.conv5_1(out)       
        out = self.conv5_2(out)       
        out = self.conv5_3(out)       

        #output
        out = self.avg_pool(out)
        out = self.ln6(out)
        out = self.fc(self.flat(out))
        
        return out






class ConvNeXt22(nn.Module):
    def __init__(self, num_channels=3, num_classes=1000):
        # Reduced size convnext, keeping ratio of different blocks the same
        # residual blocks (3,3,9,3) -> (1,1,3,1)
        # Same stem and output layers (output cut off when used as backbone)
        super(ConvNeXt22, self).__init__()
        
        self.gelu = nn.GELU()
        # STEM CELL BLOCK (112,112)->(56,56)
        self.conv1 = nn.Conv2d(num_channels, 96, kernel_size=4, stride=4, padding=0, bias=True)
        self.ln1 = LayerNorm2d(96)

        # BLOCK-2 (56,56)
        self.conv2_1 = ConvNeXtBlock(in_channels=96, out_channels=96)

        # in-between res2->res3 down sampling (56,56)->(28,28), channels 96->192
        self.down_ln2 = LayerNorm2d(96)
        self.down_res2 = nn.Conv2d(96, 192, kernel_size=1, stride=2)

        # BLOCK-3 (28,28)
        self.conv3_1 = ConvNeXtBlock(in_channels=192, out_channels=192)

        # in-between res3->res4 down sampling (28,28)->(14,14)
        self.down_ln3 = LayerNorm2d(192)
        self.down_res3 = nn.Conv2d(192, 384, kernel_size=1, stride=2)

        # BLOCK-4 (14,14)
        self.conv4_1 = ConvNeXtBlock(in_channels=384, out_channels=384)
        self.conv4_2 = ConvNeXtBlock(in_channels=384, out_channels=384)
        self.conv4_3 = ConvNeXtBlock(in_channels=384, out_channels=384)

        # in-between res4->res5 down sampling (14,14)->(7,7)
        self.down_ln4 = LayerNorm2d(384)
        self.down_res4 = nn.Conv2d(384, 768, kernel_size=1, stride=2)

        # BLOCK-5 (7,7)
        self.conv5_1 = ConvNeXtBlock(in_channels=768, out_channels=768)

        #global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.ln6 = LayerNorm2d(768)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(768,num_classes)

    def forward(self,x):
        #stem
        out = self.gelu(self.ln1(self.conv1(x)))

        #conv2
        out = self.conv2_1(out)
        out = self.down_res2(self.down_ln2(out))

        #conv3
        out = self.conv3_1(out)              
        out = self.down_res3(self.down_ln3(out))

        #conv4
        out = self.conv4_1(out)        
        out = self.conv4_2(out)        
        out = self.conv4_3(out)               
        out = self.down_res4(self.down_ln4(out))

        #conv5
        out = self.conv5_1(out)             

        #output
        out = self.avg_pool(out)
        out = self.ln6(out)
        out = self.fc(self.flat(out))
        
        return out



class miniConvNeXt(nn.Module):
    def __init__(self, num_channels=3, num_classes=1000):
        # Same number of layers as convnext50 
        # but now with number of channels drastically reduced
        # and still with kernel_size=1 in downsampling layers
        super(miniConvNeXt, self).__init__()
        
        self.gelu = nn.GELU()
        # STEM CELL BLOCK (112,112)->(56,56)
        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=4, stride=4, padding=0, bias=True)
        self.ln1 = LayerNorm2d(16)

        # BLOCK-2 (56,56)
        self.conv2_1 = ConvNeXtBlock(in_channels=16, out_channels=16)
        self.conv2_2 = ConvNeXtBlock(in_channels=16, out_channels=16)
        self.conv2_3 = ConvNeXtBlock(in_channels=16, out_channels=16)

        # in-between res2->res3 down sampling (56,56)->(28,28), channels 96->192
        self.down_ln2 = LayerNorm2d(16)
        self.down_res2 = nn.Conv2d(16, 32, kernel_size=1, stride=2)

        # BLOCK-3 (28,28)
        self.conv3_1 = ConvNeXtBlock(in_channels=32, out_channels=32)
        self.conv3_2 = ConvNeXtBlock(in_channels=32, out_channels=32)
        self.conv3_3 = ConvNeXtBlock(in_channels=32, out_channels=32)

        # in-between res3->res4 down sampling (28,28)->(14,14)
        self.down_ln3 = LayerNorm2d(32)
        self.down_res3 = nn.Conv2d(32, 64, kernel_size=1, stride=2)

        # BLOCK-4 (14,14)
        self.conv4_1 = ConvNeXtBlock(in_channels=64, out_channels=64)
        self.conv4_2 = ConvNeXtBlock(in_channels=64, out_channels=64)
        self.conv4_3 = ConvNeXtBlock(in_channels=64, out_channels=64)
        self.conv4_4 = ConvNeXtBlock(in_channels=64, out_channels=64)
        self.conv4_5 = ConvNeXtBlock(in_channels=64, out_channels=64)
        self.conv4_6 = ConvNeXtBlock(in_channels=64, out_channels=64)
        self.conv4_7 = ConvNeXtBlock(in_channels=64, out_channels=64)
        self.conv4_8 = ConvNeXtBlock(in_channels=64, out_channels=64)
        self.conv4_9 = ConvNeXtBlock(in_channels=64, out_channels=64)

        # in-between res4->res5 down sampling (14,14)->(7,7)
        self.down_ln4 = LayerNorm2d(64)
        self.down_res4 = nn.Conv2d(64, 128, kernel_size=1, stride=2)

        # BLOCK-5 (7,7)
        self.conv5_1 = ConvNeXtBlock(in_channels=128, out_channels=128)
        self.conv5_2 = ConvNeXtBlock(in_channels=128, out_channels=128)
        self.conv5_3 = ConvNeXtBlock(in_channels=128, out_channels=128)

        #global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.ln6 = LayerNorm2d(128)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(128,num_classes)

    def forward(self,x):
        #stem
        out = self.gelu(self.ln1(self.conv1(x)))

        #conv2
        out = self.conv2_1(out)
        out = self.conv2_2(out)
        out = self.conv2_3(out)

        out = self.down_res2(self.down_ln2(out))

        #conv3
        out = self.conv3_1(out)        
        out = self.conv3_2(out)        
        out = self.conv3_3(out)        

        out = self.down_res3(self.down_ln3(out))

        #conv4
        out = self.conv4_1(out)        
        out = self.conv4_2(out)        
        out = self.conv4_3(out)        
        out = self.conv4_4(out)        
        out = self.conv4_5(out)        
        out = self.conv4_6(out)        
        out = self.conv4_7(out)        
        out = self.conv4_8(out)        
        out = self.conv4_9(out)        

        out = self.down_res4(self.down_ln4(out))

        #conv5
        out = self.conv5_1(out)       
        out = self.conv5_2(out)       
        out = self.conv5_3(out)       

        #output
        out = self.avg_pool(out)
        out = self.ln6(out)
        out = self.fc(self.flat(out))
        
        return out



class petiteConvNeXt(nn.Module):
    def __init__(self, num_channels=3, num_classes=1000):
        # Same number of layers as convnext22 
        # but now with number of channels drastically reduced
        # and still with kernel_size=1 in downsampling layers
        super(petiteConvNeXt, self).__init__()
        
        self.gelu = nn.GELU()
        # STEM CELL BLOCK (112,112)->(56,56)
        self.conv1 = nn.Conv2d(num_channels, 8, kernel_size=4, stride=4, padding=0, bias=True)
        self.ln1 = LayerNorm2d(8)

        # BLOCK-2 (56,56)
        self.conv2_1 = ConvNeXtBlock(in_channels=8, out_channels=8)

        # in-between res2->res3 down sampling (56,56)->(28,28), channels 96->192
        self.down_ln2 = LayerNorm2d(8)
        self.down_res2 = nn.Conv2d(8, 16, kernel_size=1, stride=2) #should be kernel size 2

        # BLOCK-3 (28,28)
        self.conv3_1 = ConvNeXtBlock(in_channels=16, out_channels=16)

        # in-between res3->res4 down sampling (28,28)->(14,14)
        self.down_ln3 = LayerNorm2d(16)
        self.down_res3 = nn.Conv2d(16, 32, kernel_size=1, stride=2) #should be kernel size 2

        # BLOCK-4 (14,14)
        self.conv4_1 = ConvNeXtBlock(in_channels=32, out_channels=32)
        self.conv4_2 = ConvNeXtBlock(in_channels=32, out_channels=32)
        self.conv4_3 = ConvNeXtBlock(in_channels=32, out_channels=32)

        # in-between res4->res5 down sampling (14,14)->(7,7)
        self.down_ln4 = LayerNorm2d(32)
        self.down_res4 = nn.Conv2d(32, 64, kernel_size=1, stride=2) #should be kernel size 2

        # BLOCK-5 (7,7)
        self.conv5_1 = ConvNeXtBlock(in_channels=64, out_channels=64)

        #global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.ln6 = LayerNorm2d(64)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(64,num_classes)

    def forward(self,x):
        #stem
        out = self.gelu(self.ln1(self.conv1(x)))

        #conv2
        out = self.conv2_1(out)
        out = self.down_res2(self.down_ln2(out))

        #conv3
        out = self.conv3_1(out)              
        out = self.down_res3(self.down_ln3(out))

        #conv4
        out = self.conv4_1(out)        
        out = self.conv4_2(out)        
        out = self.conv4_3(out)               
        out = self.down_res4(self.down_ln4(out))

        #conv5
        out = self.conv5_1(out)            

        #output
        out = self.avg_pool(out)
        out = self.ln6(out)
        out = self.fc(self.flat(out))
        
        return out




class tinyConvNeXt(nn.Module):
    def __init__(self, num_channels=3, num_classes=1000):
        # Same number of layers as convnext22 
        # but now with number of channels drastically reduced
        # and still with kernel_size=1 in downsampling layers
        super(tinyConvNeXt, self).__init__()
        
        self.gelu = nn.GELU()
        # STEM CELL BLOCK (112,112)->(56,56)
        self.conv1 = nn.Conv2d(num_channels, 8, kernel_size=4, stride=4, padding=0, bias=True)
        self.ln1 = LayerNorm2d(8)

        # BLOCK-2 (56,56)
        self.conv2_1 = ConvNeXtBlock(in_channels=8, out_channels=8)

        # in-between res2->res3 down sampling (56,56)->(28,28), channels 96->192
        self.down_ln2 = LayerNorm2d(8)
        self.down_res2 = nn.Conv2d(8, 16, kernel_size=1, stride=2)

        # BLOCK-3 (28,28)
        self.conv3_1 = ConvNeXtBlock(in_channels=16, out_channels=16)

        # in-between res3->res4 down sampling (28,28)->(14,14)
        self.down_ln3 = LayerNorm2d(16)
        self.down_res3 = nn.Conv2d(16, 24, kernel_size=1, stride=2)

        # BLOCK-4 (14,14)
        self.conv4_1 = ConvNeXtBlock(in_channels=24, out_channels=24)
        self.conv4_2 = ConvNeXtBlock(in_channels=24, out_channels=24)
        self.conv4_3 = ConvNeXtBlock(in_channels=24, out_channels=24) # here's where we cut for backbone reduce channels 32->24

        # in-between res4->res5 down sampling (14,14)->(7,7)
        self.down_ln4 = LayerNorm2d(24)
        self.down_res4 = nn.Conv2d(24, 64, kernel_size=1, stride=2)

        # BLOCK-5 (7,7)
        self.conv5_1 = ConvNeXtBlock(in_channels=64, out_channels=64)

        #global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.ln6 = LayerNorm2d(64)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(64,num_classes)

    def forward(self,x):
        #stem
        out = self.gelu(self.ln1(self.conv1(x)))

        #conv2
        out = self.conv2_1(out)
        out = self.down_res2(self.down_ln2(out))

        #conv3
        out = self.conv3_1(out)              
        out = self.down_res3(self.down_ln3(out))

        #conv4
        out = self.conv4_1(out)        
        out = self.conv4_2(out)        
        out = self.conv4_3(out)               
        out = self.down_res4(self.down_ln4(out))

        #conv5
        out = self.conv5_1(out)            

        #output
        out = self.avg_pool(out)
        out = self.ln6(out)
        out = self.fc(self.flat(out))
        
        return out



class nanoConvNeXt(nn.Module):
    def __init__(self, num_channels=3, num_classes=1000):
        super(nanoConvNeXt, self).__init__()
        
        self.gelu = nn.GELU()
        # STEM CELL BLOCK (112,112)->(56,56)
        self.conv1 = nn.Conv2d(num_channels, 8, kernel_size=4, stride=4, padding=0, bias=True)
        self.ln1 = LayerNorm2d(8)

        # BLOCK-2 (56,56)
        self.conv2_1 = ConvNeXtBlock(in_channels=8, out_channels=8)
        self.channel_res2 = nn.Conv2d(8, 16, kernel_size=1, stride=1)

        # BLOCK-3 
        self.conv3_1 = ConvNeXtBlock(in_channels=16, out_channels=16)
        self.channel_res3 = nn.Conv2d(16, 24, kernel_size=1, stride=1)

        # BLOCK-4 
        self.conv4_1 = ConvNeXtBlock(in_channels=24, out_channels=24)
        self.conv4_2 = ConvNeXtBlock(in_channels=24, out_channels=24) # here's where we cut for SSD. reduce channels 32->24

        # in-between res4->res5 down sampling (14,14)->(7,7)
        self.down_ln4 = LayerNorm2d(24)
        self.down_res4 = nn.Conv2d(24, 64, kernel_size=1, stride=2) #should be kernel size 2

        # BLOCK-5 (7,7)
        self.conv5_1 = ConvNeXtBlock(in_channels=64, out_channels=64)

        #global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.ln6 = LayerNorm2d(64)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(64,num_classes)

    def forward(self,x):
        #stem
        out = self.gelu(self.ln1(self.conv1(x)))

        #conv2
        out = self.conv2_1(out)
        out = self.channel_res2(out)

        #conv3
        out = self.conv3_1(out)              
        out = self.channel_res3(out)

        #conv4
        out = self.conv4_1(out)        
        out = self.conv4_2(out)                   
        out = self.down_res4(self.down_ln4(out))

        #conv5
        out = self.conv5_1(out)            

        #output
        out = self.avg_pool(out)
        out = self.ln6(out)
        out = self.fc(self.flat(out))
        
        return out




class UConvNeXt(nn.Module):
    def __init__(self, num_channels=3,hidden_channels=24):
        # Here we combine a U-net architecture using ConvNeXt block as downsampling
        # TODO: replace convtranspose2d
        # Keep residual connections from the same down/up sampled steps
        super(UConvNeXt, self).__init__()
        
        self.gelu = nn.GELU()

        # Stem Block depthwise-conv
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=hidden_channels, kernel_size=5, stride=1, padding=2, groups=num_channels)
        self.ln1 = LayerNorm2d(hidden_channels)

        # BLOCK-1 (125,96)
        self.conv2_1 = ConvNeXtBlock(in_channels=hidden_channels, out_channels=hidden_channels)

        # Down-res 1 (62, 48)
        self.pool = nn.MaxPool2d(2)
        self.ln2 = LayerNorm2d(hidden_channels)

        # BLOCK-2 (62,48)
        self.conv3_1 = ConvNeXtBlock(in_channels=hidden_channels, out_channels=hidden_channels)

        # Down-res 2 (31, 24)
        self.pool2 = nn.MaxPool2d(2)
        self.ln3 = LayerNorm2d(hidden_channels)

        # BLOCK-3 (31,24)
        self.conv4_1 = ConvNeXtBlock(in_channels=hidden_channels, out_channels=hidden_channels)

        # Up-res 1 (62, 48)
        self.up1 = nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=2, stride=2, padding=0)

        # BLOCK-4 (62, 48)
        self.conv5_1 = ConvNeXtBlock(in_channels=hidden_channels*2, out_channels=hidden_channels*2)

        # Up-res 2 (125, 96)
        self.up2 = nn.ConvTranspose2d(in_channels=hidden_channels*2, out_channels=hidden_channels, kernel_size=2, stride=2, padding=0, output_padding=(1,0))

    def forward(self,x):

        #stem
        out1 = self.gelu(self.ln1(self.conv1(x)))
        # print('End of stem',out1.shape)

        #conv2
        out2 = self.conv2_1(out1)
        # print('End of conv2',out2.shape)

        #pool1
        out3 = self.ln2(self.pool(out2))
        # print('End of pool1',out3.shape)

        #conv3
        out4 = self.conv3_1(out3)
        # print('End of conv3',out4.shape)

        #pool2
        out5 = self.ln3(self.pool2(out4))
        # print('End of pool2',out5.shape)

        #conv4
        out6 = self.conv4_1(out5)
        # print('End of conv4',out6.shape)

        #up sample 1
        out7 = self.up1(out6)
        # print('End of up1',out7.shape)
        out7 = torch.cat([out4,out7],dim=1)

        #conv5
        out8 = self.conv5_1(out7)
        # print('End of conv5',out8.shape)

        #up sample 2
        out9 = self.up2(out8)
        # print('End of up2',out9.shape)
        out9 = torch.cat([out2,out9],dim=1)

        return out9





class smallUConvNeXt(nn.Module):
    def __init__(self, num_channels=3,hidden_channels=24):
        # Similar to UConvNeXt, but with the number of blocks reduced
        # fewer steps down + up
        super(smallUConvNeXt, self).__init__()
        
        self.gelu = nn.GELU()

        # Stem Block depthwise-conv
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=hidden_channels, kernel_size=5, stride=1, padding=2, groups=num_channels)
        self.ln1 = LayerNorm2d(hidden_channels)

        # BLOCK-1 (125,96)
        self.conv2_1 = ConvNeXtBlock(in_channels=hidden_channels, out_channels=hidden_channels)

        # Down-res 1 (62, 48)
        self.pool = nn.MaxPool2d(2)
        self.ln2 = LayerNorm2d(hidden_channels)

        # BLOCK-4 (62, 48)
        self.conv5_1 = ConvNeXtBlock(in_channels=hidden_channels, out_channels=hidden_channels)

        # Up-res 2 (125, 96)
        self.up2 = nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=2, stride=2, padding=0, output_padding=(1,0))

    def forward(self,x):

        #stem
        out1 = self.gelu(self.ln1(self.conv1(x)))
        # print('End of stem',out1.shape)

        #conv2
        out2 = self.conv2_1(out1)
        # print('End of conv2',out2.shape)

        #pool1
        out3 = self.ln2(self.pool(out2))
        # print('End of pool1',out3.shape)

        #conv5
        out8 = self.conv5_1(out3)
        # print('End of conv5',out8.shape)

        #up sample 2
        out9 = self.up2(out8)
        # print('End of up2',out9.shape)
        out9 = torch.cat([out2,out9],dim=1)

        return out9



    




class UConvNeXt_central(nn.Module):
    def __init__(self, num_channels=3,hidden_channels=24):
        # adaption of UConvNeXt to better handle images with different shapes
        # still rectangular. Requires different padding to UConvNeXt case
        super(UConvNeXt_central, self).__init__()
        
        self.gelu = nn.GELU()

        # Stem Block depthwise-conv
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=hidden_channels, kernel_size=5, stride=1, padding=2, groups=num_channels)
        self.ln1 = LayerNorm2d(hidden_channels)

        # BLOCK-1 (125,96)
        self.conv2_1 = ConvNeXtBlock(in_channels=hidden_channels, out_channels=hidden_channels)

        # Down-res 1 (62, 48)
        self.pool = nn.MaxPool2d(2)
        self.ln2 = LayerNorm2d(hidden_channels)

        # BLOCK-2 (62,48)
        self.conv3_1 = ConvNeXtBlock(in_channels=hidden_channels, out_channels=hidden_channels)

        # Down-res 2 (31, 24)
        self.pool2 = nn.MaxPool2d(2)
        self.ln3 = LayerNorm2d(hidden_channels)

        # BLOCK-3 (31,24)
        self.conv4_1 = ConvNeXtBlock(in_channels=hidden_channels, out_channels=hidden_channels)

        # Up-res 1 (62, 48)
        self.up1 = nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=2, stride=2, padding=0)

        # BLOCK-4 (62, 48)
        self.conv5_1 = ConvNeXtBlock(in_channels=hidden_channels*2, out_channels=hidden_channels*2)

        # Up-res 2 (125, 96)
        self.up2 = nn.ConvTranspose2d(in_channels=hidden_channels*2, out_channels=hidden_channels, kernel_size=2, stride=2, padding=0, output_padding=(1,1))

    def forward(self,x):

        #stem
        out1 = self.gelu(self.ln1(self.conv1(x)))
        print('End of stem',out1.shape)

        #conv2
        out2 = self.conv2_1(out1)
        print('End of conv2',out2.shape)

        #pool1
        out3 = self.ln2(self.pool(out2))
        print('End of pool1',out3.shape)

        #conv3
        out4 = self.conv3_1(out3)
        print('End of conv3',out4.shape)

        #pool2
        out5 = self.ln3(self.pool2(out4))
        print('End of pool2',out5.shape)

        #conv4
        out6 = self.conv4_1(out5)
        print('End of conv4',out6.shape)

        #up sample 1
        out7 = self.up1(out6)
        print('End of up1',out7.shape)
        out7 = torch.cat([out4,out7],dim=1)

        #conv5
        out8 = self.conv5_1(out7)
        print('End of conv5',out8.shape)

        #up sample 2
        out9 = self.up2(out8)
        print('End of up2',out9.shape)
        out9 = torch.cat([out2,out9],dim=1)

        return out9














if __name__=="__main__":

    layer = ConvNeXtBlock(in_channels=96, out_channels=96)
    print("Have initiatilised convnext block")
    total_params = sum(p.numel() for p in layer.parameters())
    print(f"{total_params:,} parameters in 1 (initial) convnext block\n")
    pytorch_layer = torchvision.models.convnext.CNBlock(dim=96,layer_scale=1e-6,stochastic_depth_prob=0.0) # CNBlockConfig(input_channels=96,output_channels=192,num_layers=3)
    pytorch_params = sum(p.numel() for p in pytorch_layer.parameters())
    print(f"{pytorch_params:,} parameters in 1 (initial) pytorch convnext block\n")

    input_tensor = torch.randn(1, 3, 224, 224) # from paper 
    # input_tensor = torch.randn(1, 3, 125, 96)  # from det.


    # First model implementation, from paper
    model = ConvNeXt_T(num_classes=1000)
    output_tensor = model(input_tensor)
    print(f"=================== ConvNeXt-T (ConvNeXt-50 from paper) ===================")
    print(f"Input tensor: {input_tensor.shape}, Output tensor: {output_tensor.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters in ConvNeXt-T model")

    pytorch_model = torchvision.models.convnext_tiny()
    pytorch_params = sum(p.numel() for p in pytorch_model.parameters())
    print(f"{pytorch_params:,} total parameters in pytorch ConvNeXt model")
    print(f"(includes stochastic depth (dropout) and linear layers <-> 1x1 convs)")
    print(f"=================== ========== ===================")
    print()


    # Custom implementations, scaling down # of parameters
    print(f"=================== ConvNeXt-50 ===================")
    model = ConvNeXt50(num_classes=1000)
    output_tensor = model(input_tensor)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters in ConvNeXt-50 model")
    print(f"=================== ========== ===================")
    print()

    print(f"=================== ConvNeXt-22 ===================")
    model = ConvNeXt22(num_classes=1000)
    output_tensor = model(input_tensor)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters in custom ConvNeXt-22 model")
    print(f"=================== ========== ===================")
    print()

    print(f"=================== MiniConvNeXt ===================")
    model = miniConvNeXt(num_classes=1000)
    output_tensor = model(input_tensor)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters in custom MiniConvNeXt model")
    print(f"=================== ========== ===================")
    print()

    print(f"=================== PetiteConvNeXt ===================")
    model = petiteConvNeXt(num_classes=1000)
    output_tensor = model(input_tensor)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters in custom PetiteConvNeXt model")
    print(f"=================== ========== ===================")
    print()

    print(f"=================== TinyConvNeXt ===================")
    model = tinyConvNeXt(num_classes=1000)
    output_tensor = model(input_tensor)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters in custom TinyConvNeXt model")
    print(f"=================== ========== ===================")
    print()

    print(f"=================== NanoConvNeXt ===================")
    model = nanoConvNeXt(num_classes=1000)
    output_tensor = model(input_tensor)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters in custom NanoConvNeXt model")
    print(f"=================== ========== ===================")
    print()

    input_tensor = torch.randn(1, 3, 125, 96)  # UConvNext designed for det. size
    print(f"=================== UConvNeXt ===================")
    model = UConvNeXt()
    output_tensor = model(input_tensor)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters in custom UConvNeXt model")
    print(f"=================== ========== ===================")
    print()

    print(f"=================== smallUConvNeXt ===================")
    model = smallUConvNeXt()
    output_tensor = model(input_tensor)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters in custom smallUConvNeXt model")
    print(f"=================== ========== ===================")
    print()

    # [125, 49]
    input_tensor = torch.randn(1, 3, 125, 49)
    print(f"=================== central UConvNeXt ===================")
    model = UConvNeXt_central()
    output_tensor = model(input_tensor)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters in custom central UConvNeXt model")
    print(f"=================== ========== ===================")
    print()

