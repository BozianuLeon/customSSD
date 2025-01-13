import torch
import torch.nn as nn
import torchvision


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, res_channels, out_channels, stride=1, downsample=False):
        # where res_channel is the in channel we would *like* to have, 
        # not always possible due to output of previous blocks
        super(ResnetBlock, self).__init__()
        self.relu = nn.ReLU()

        self.conv1x1_1 = nn.Conv2d(in_channels, res_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1x1_1 = nn.BatchNorm2d(res_channels)

        self.conv3x3 = nn.Conv2d(res_channels, res_channels, kernel_size=3, stride=stride, padding=1, bias=False)
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
            # print('DOWNSAMPLING!')
            x = self.skip_bn(self.skip_con(x))
        
        ret = self.relu(out+x)
        return ret


class ResNet50(nn.Module):
    def __init__(self, num_classes=10, num_channels=3):
        super(ResNet50, self).__init__()
        self.in_channels = 64
        
        self.relu = nn.ReLU()
        # BLOCK-1 INITIALISE (112,112)
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)

        # BLOCK-2 64->256 channels, (56,56)
        self.conv2_1 = ResnetBlock(in_channels=64,res_channels=64,out_channels=256, downsample=True)
        self.conv2_2 = ResnetBlock(in_channels=256,res_channels=64,out_channels=256)
        self.conv2_3 = ResnetBlock(in_channels=256,res_channels=64,out_channels=256)
    
        # BLOCK-3 128->512 channels, (28,28)
        self.conv3_1 = ResnetBlock(in_channels=256,res_channels=128,out_channels=512, downsample=True, stride=2)
        self.conv3_2 = ResnetBlock(in_channels=512,res_channels=128,out_channels=512)
        self.conv3_3 = ResnetBlock(in_channels=512,res_channels=128,out_channels=512)
        self.conv3_4 = ResnetBlock(in_channels=512,res_channels=128,out_channels=512)

        # BLOCK-4 256->1024 channels, (14,14)
        self.conv4_1 = ResnetBlock(in_channels=512,res_channels=256,out_channels=1024, downsample=True, stride=2)
        self.conv4_2 = ResnetBlock(in_channels=1024,res_channels=256,out_channels=1024)
        self.conv4_3 = ResnetBlock(in_channels=1024,res_channels=256,out_channels=1024)
        self.conv4_4 = ResnetBlock(in_channels=1024,res_channels=256,out_channels=1024)
        self.conv4_5 = ResnetBlock(in_channels=1024,res_channels=256,out_channels=1024)
        self.conv4_6 = ResnetBlock(in_channels=1024,res_channels=256,out_channels=1024)

        # BLOCK-5 512->2048 channels, (7,7)
        self.conv5_1 = ResnetBlock(in_channels=1024,res_channels=512,out_channels=2048, downsample=True, stride=2)
        self.conv5_2 = ResnetBlock(in_channels=2048,res_channels=512,out_channels=2048)
        self.conv5_3 = ResnetBlock(in_channels=2048,res_channels=512,out_channels=2048)

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

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)
        
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)
        x = self.conv4_5(x)
        x = self.conv4_6(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)

        x = self.avg_pool(x)
        x = torch.flatten(x)
        x = self.fc(x)

        return x    




if __name__=="__main__":
    model = ResNet50(num_classes=1000)

    input_tensor = torch.randn(1, 3, 300, 300)  
    output_tensor = model(input_tensor)
    print(f"Input tensor: {input_tensor.shape}, Output tensor: {output_tensor.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
