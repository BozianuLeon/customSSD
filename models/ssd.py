import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.convnext import petiteConvNeXt, tinyConvNeXt, UConvNeXt, UConvNeXt_central
from models.resnext import ResNeXt20
from models.sumpool import MaskSumPool
from models.layernorm import LayerNorm2d

class CustomFeatureExtractor(nn.Module):
    def __init__(self, name, in_channels=3,hidden_channels=24):
        super().__init__()

        if name=="resnext50":
            backbone = ResNeXt50(num_channels=in_channels,num_classes=10)
            self.out_channels = [1024, 512, 512, 256, 256, 256]

            # cut resnet50 after conv4 block (up to conv4_6)
            self.feature_extractor = torch.nn.Sequential(*list(backbone.children())[:17])

            # stop the downsampling (image size) in our new "last" block 
            # set stride to 1,1 in those layers
            conv4_block1 = self.feature_extractor[-6]
            conv4_block1.conv3x3.stride = (1, 1)
            conv4_block1.skip_con.stride = (1, 1)
 
        elif name=="resnext20":
            backbone = ResNeXt20(num_channels=in_channels,num_classes=10)
            self.out_channels = [1024, 512, 512, 256, 256, 256]

            # cut resnet50 after conv4 block (up to conv4_6)
            self.feature_extractor = torch.nn.Sequential(*list(backbone.children())[:10])
            # print(list(backbone.children())[9])
     
            conv4_block1 = self.feature_extractor[-3]
            conv4_block1.conv3x3.stride = (1, 1)
            conv4_block1.skip_con.stride = (1, 1)

        elif name=="petiteconvnext":
            backbone = petiteConvNeXt(num_channels=in_channels,num_classes=10)
            self.out_channels = [32, 64, 64, 32, 32, 32] 

            self.feature_extractor = torch.nn.Sequential(*list(backbone.children())[:12])
            prev_downsample = self.feature_extractor[-4]
            prev_downsample.stride=(1,1)

        elif name=="tinyconvnext":
            backbone = tinyConvNeXt(num_channels=in_channels,num_classes=10)
            self.out_channels = [24, 16, 16, 8, 8, 8] 

            self.feature_extractor = torch.nn.Sequential(*list(backbone.children())[:12])
            prev_downsample = self.feature_extractor[-4]
            prev_downsample.stride=(1,1)

        elif name=="uconvnext":
            self.feature_extractor = UConvNeXt(num_channels=in_channels,hidden_channels=hidden_channels)
            self.out_channels = [hidden_channels*2]

        elif name=="uconvnext_central":
            self.feature_extractor = UConvNeXt_central(num_channels=in_channels,hidden_channels=hidden_channels)
            self.out_channels = [hidden_channels*2]

    def forward(self, x):
        return self.feature_extractor(x)
        

class SSD(torch.nn.Module):
    def __init__(self, backbone_name, in_channels=10):
        super().__init__()

        # grab chosen feature extractor
        self.backbone_name = backbone_name
        self.feature_extractor = CustomFeatureExtractor(in_channels=in_channels*2, hidden_channels=20, name=backbone_name)
        print(f"Backbone model:    {sum(p.numel() for p in self.feature_extractor.parameters()):,} parameters.")

        self.label_num = 1 
        self._build_additional_features(self.feature_extractor.out_channels)
        self.num_defaults = [1]
        
        self.loc = nn.Conv2d(self.feature_extractor.out_channels[0], self.num_defaults[0] * 4, kernel_size=3, padding=1)
        self.conf = nn.Conv2d(self.feature_extractor.out_channels[0], self.num_defaults[0] * self.label_num, kernel_size=3, padding=1)
        print(f"Loc head:          {sum(p.numel() for p in self.loc.parameters()):,} parameters.")
        print(f"Conf head:         {sum(p.numel() for p in self.conf.parameters()):,} parameters.")

        self.sumpool = MaskSumPool(kernel_size=9, in_channels=in_channels, stride=1, pool_mask=None)
        print(f"PT map:            {sum(p.numel() for p in self.sumpool.parameters()):,} (frozen) parameters.")
        
        self._init_weights()

    def _build_additional_features(self, input_size, hidden_channels=24):
        layer1 = nn.Sequential(
                    nn.Conv2d(input_size[0], hidden_channels, kernel_size=3, padding=1, stride=1, bias=False),
                    LayerNorm2d(hidden_channels),
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False),
                    nn.GELU(),)
        
        layer2 = nn.Sequential(
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, stride=1, bias=False),
                    LayerNorm2d(hidden_channels),
                    nn.Conv2d(hidden_channels, self.feature_extractor.out_channels[0], kernel_size=1, bias=False),
                    nn.GELU(),
                    nn.MaxPool2d(2,padding=(1,0)),)

        self.additional_blocks = nn.ModuleList([layer1,layer2])

        total_params = sum(p.numel() for p in self.additional_blocks.parameters())
        print(f"Aux. layers:       {total_params:,} parameters.")

    def _init_weights(self):
        layers = [*self.additional_blocks, self.loc, self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1: nn.init.xavier_uniform_(param)

    # Shape the classifier to the view of bboxes
    def bbox_view(self, src, loc, conf):
        lc_out = loc(src).reshape(src.size(0), 4, -1)
        conf_out = conf(src).reshape(src.size(0),self.label_num,-1)
        return lc_out, conf_out

    def forward(self, x):
        print('input image shape',x.shape)

        ptmap = self.sumpool(x)
        print('ptmap shape',ptmap.shape)
        x = torch.cat((x, ptmap), dim=1)
        print('concat image+sumpool',x.shape)

        x = self.feature_extractor(x)
        print('After feature ext.',x.shape)     

        for l in self.additional_blocks:
            x = l(x)
        print('After additional blocks',x.shape)

        # print('feature maps [125,96] reshape->',125*96, 'but we downsample the image in aux layers to to reduce to [63,48]',63*48,'equal to our step_x, step_y=2' )
        # print('I want: Number of dboxes:  torch.Size([3024, 4])')
        locs, confs = self.bbox_view(x, self.loc, self.conf)
        print('After bbox_view',confs.shape,locs.shape)

        return locs, confs, ptmap[:,0,:,:]




if __name__=="__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SSD(backbone_name="uconvnext_central",in_channels=5)
    model = model.to(device) 
    total_params = sum(p.numel() for p in model.parameters())
    print(model.backbone_name,f'!total \t{total_params:,} parameters.\n')

    random_input = torch.randn(5,125,49)
    random_input = random_input.unsqueeze(0)
    random_input = random_input.to(device)
    print("Input shape", random_input.shape)


    print("One forward pass")
    plocs, plabel, ptmap = model(random_input) #plocs.shape(torch.Size([BS, 4, 3024])) and plabel.shape(torch.Size([BS, 1, 3024]))

