import torch
import torch.nn as nn
import torchvision



class ResneXtBlock(nn.Module):
    def __init__(self, in_channels, res_channels, out_channels, cardinality=32, downsample=False, stride=1):
        # Simple implementation of resnext block. See https://arxiv.org/abs/1611.05431 
        # where res_channel is the in channel we would *like* to have, 
        # not always possible due to output of previous blocks
        super(ResneXtBlock, self).__init__()
        self.relu = nn.ReLU()

        self.conv1x1_1 = nn.Conv2d(in_channels, res_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1x1_1 = nn.BatchNorm2d(res_channels)

        self.conv3x3 = nn.Conv2d(res_channels, res_channels, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn3x3 = nn.BatchNorm2d(res_channels)
        
        self.conv1x1_2 = nn.Conv2d(res_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1x1_2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        if self.downsample:
            self.skip_con = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, bias=False) # residual connection CONV
            self.skip_bn = nn.BatchNorm2d(out_channels) 
    
    def forward(self,x):
    
        out = self.relu(self.bn1x1_1(self.conv1x1_1(x)))
        
        out = self.relu(self.bn3x3(self.conv3x3(out)))

        out = self.bn1x1_2(self.conv1x1_2(out))

        if self.downsample:
            x = self.skip_bn(self.skip_con(x))
        
        ret = self.relu(out+x)
        return ret




class ResNeXt50(nn.Module):
    def __init__(self, num_classes, num_channels=3):
        super(ResNeXt50, self).__init__()
        self.in_channels = 64
        
        self.relu = nn.ReLU()
        # BLOCK-1 INITIALISE (112,112)
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)

        # BLOCK-2 128->256 channels, (56,56)
        self.conv2_1 = ResneXtBlock(in_channels=64,res_channels=128,out_channels=256, downsample=True)
        self.conv2_2 = ResneXtBlock(in_channels=256,res_channels=128,out_channels=256)
        self.conv2_3 = ResneXtBlock(in_channels=256,res_channels=128,out_channels=256)
    
        # BLOCK-3 256->512 channels, (28,28)
        self.conv3_1 = ResneXtBlock(in_channels=256,res_channels=256,out_channels=512, downsample=True, stride=2)
        self.conv3_2 = ResneXtBlock(in_channels=512,res_channels=256,out_channels=512)
        self.conv3_3 = ResneXtBlock(in_channels=512,res_channels=256,out_channels=512)
        self.conv3_4 = ResneXtBlock(in_channels=512,res_channels=256,out_channels=512)

        # BLOCK-4 512->1024 channels, (14,14)
        self.conv4_1 = ResneXtBlock(in_channels=512,res_channels=512,out_channels=1024, downsample=True, stride=2)
        self.conv4_2 = ResneXtBlock(in_channels=1024,res_channels=512,out_channels=1024)
        self.conv4_3 = ResneXtBlock(in_channels=1024,res_channels=512,out_channels=1024)
        self.conv4_4 = ResneXtBlock(in_channels=1024,res_channels=512,out_channels=1024)
        self.conv4_5 = ResneXtBlock(in_channels=1024,res_channels=512,out_channels=1024)
        self.conv4_6 = ResneXtBlock(in_channels=1024,res_channels=512,out_channels=1024)

        # BLOCK-5 1024->2048 channels, (7,7)
        self.conv5_1 = ResneXtBlock(in_channels=1024,res_channels=1024,out_channels=2048, downsample=True, stride=2)
        self.conv5_2 = ResneXtBlock(in_channels=2048,res_channels=1024,out_channels=2048)
        self.conv5_3 = ResneXtBlock(in_channels=2048,res_channels=1024,out_channels=2048)

        # BLOCK-6 Pooling + Flattening
        # self.avg_pool = nn.AvgPool2d(kernel_size=(7,7), stride=(1,1))
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(2048,1000)

    def forward(self,x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        # print('End of conv2',x.shape)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)
        # print('End of conv3',x.shape)
        
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)
        x = self.conv4_5(x)
        x = self.conv4_6(x)
        # print('End of conv4',x.shape)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        # print('End of conv5',x.shape)

        x = self.avg_pool(x)
        x = torch.flatten(x)
        x = self.fc(x)

        return x   






class ResNeXt20(nn.Module):
    def __init__(self, num_classes, num_channels=3):
        super(ResNeXt20, self).__init__()
        self.in_channels = 64
        
        self.relu = nn.ReLU()
        # BLOCK-1 INITIALISE (112,112)
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)

        # BLOCK-2 128->256 channels, (56,56)
        self.conv2_1 = ResneXtBlock(in_channels=64,res_channels=128,out_channels=256, downsample=True)

    
        # BLOCK-3 256->512 channels, (28,28)
        self.conv3_1 = ResneXtBlock(in_channels=256,res_channels=256,out_channels=512, downsample=True, stride=2)
        self.conv3_2 = ResneXtBlock(in_channels=512,res_channels=256,out_channels=512)


        # BLOCK-4 512->1024 channels, (14,14)
        self.conv4_1 = ResneXtBlock(in_channels=512,res_channels=512,out_channels=1024, downsample=True, stride=2)
        self.conv4_2 = ResneXtBlock(in_channels=1024,res_channels=512,out_channels=1024)
        self.conv4_3 = ResneXtBlock(in_channels=1024,res_channels=512,out_channels=1024)


        # BLOCK-5 1024->2048 channels, (7,7)
        self.conv5_1 = ResneXtBlock(in_channels=1024,res_channels=1024,out_channels=2048, downsample=True, stride=2)


        # BLOCK-6 Pooling + Flattening
        # self.avg_pool = nn.AvgPool2d(kernel_size=(7,7), stride=(1,1))
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(2048,1000)

    def forward(self,x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.conv2_1(x)
        # print('End of conv2',x.shape)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        # print('End of conv3',x.shape)
        
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        # print('End of conv4',x.shape)

        x = self.conv5_1(x)
        # print('End of conv5',x.shape)

        x = self.avg_pool(x)
        x = torch.flatten(x)
        x = self.fc(x)

        return x   




if __name__=="__main__":

    layer = ResneXtBlock(in_channels=1024, res_channels=1024, out_channels=2048, cardinality=32, downsample=False, stride=1)
    total_params = sum(p.numel() for p in layer.parameters())
    print(f"{total_params:,} parameters in 1 (initial) resnext block\n")



    input_tensor = torch.randn(1, 3, 300, 300)  

    # First model implementation, from paper
    model = ResNeXt50(num_classes=1000)
    output_tensor = model(input_tensor)
    print(f"=================== ResNeXt-50 ===================")
    print(f"Input tensor: {input_tensor.shape}, Output tensor: {output_tensor.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters in ResNeXt-50 model")

    pytorch_model = torchvision.models.resnext50_32x4d()
    total_params = sum(p.numel() for p in pytorch_model.parameters())
    print(f"{total_params:,} total parameters in pytorch ResNeXt model")
    print(f"=================== ========== ===================")
    print()

    # Second model implementation, 20 blocks
    model = ResNeXt20(num_classes=1000)
    output_tensor = model(input_tensor)
    print(f"=================== ResNeXt-20 ===================")
    print(f"Input tensor: {input_tensor.shape}, Output tensor: {output_tensor.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters in custom ResNeXt-20 model")
    print(f"=================== ========== ===================")

