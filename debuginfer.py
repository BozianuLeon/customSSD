import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import v2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
import os

import time
from statistics import mean

import data





# https://pytorch.org/docs/stable/notes/randomness.html
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

# ann_file="/srv/beegfs/scratch/shares/atlas_caloM/mu_200_truthjets/central_2sig_images/anns_central_jets_truth_JZcomb0_train.json"
# ann_file="/srv/beegfs/scratch/shares/atlas_caloM/mu_200_truthjets/central_2sig_images/anns_central_jets_JZcomb0.json"
# ann_file="/home/users/b/bozianu/work/data/mu200/anns_central_jets_20GeV.json"
# ann_file="/srv/beegfs/scratch/shares/atlas_caloM/mu_200_truthjets/central_2sig_images/anns_central_jets_JZ4.2.json"
ann_file="/srv/beegfs/scratch/shares/atlas_caloM/mu_200_truthjets/central_2sig_images/anns_central_jets_truth_JZ0_test.json"
backbone = "smallconvnext_central"
model_dir="/home/users/b/bozianu/work/paperSSD/customSSD/saved_models/"
output_dir="/home/users/b/bozianu/work/paperSSD/customSSD/cache/"
proc="JZ0"

config = {
    "seed"       : 0,
    "device"     : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "NW"         : 0,
    "BS"         : 1,
    "LR"         : 0.01,
    "WD"         : 0.01,
    "wup_epochs" : int(40/3),
    "n_epochs"   : int(40),
    "max_num"    : 150,
}
torch.manual_seed(config["seed"])
MIN_CELLS_PHI,MAX_CELLS_PHI = -3.1334076, 3.134037
# MIN_CELLS_ETA,MAX_CELLS_ETA = -4.823496, 4.823496
MIN_CELLS_ETA,MAX_CELLS_ETA = -2.5, 2.5


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, annotation_file, rnd_flips=False, truth_info=False):
        # Custom dataset that takes in the annotations folder 
        # and will randomly flip the images/bounding boxes during training
        # returns 
        # img: pytorch tensor [5,125,49]
        # validation dict: contains lists/tensors of dict_keys(['boxes', 'labels', 'jet_pt', 'extent', 'h5file', 'h5event', 'event_no'])
        # here the channel order is: 
        # [H_sum_pt,H_max_pt,H_sum_signif,H_max_signif,H_max_noise]
        with open(annotation_file, 'r') as f:
            self.data = json.load(f)
        
        self.truth_info = truth_info
        self.rnd_flips = rnd_flips
        self.transforms = v2.Compose([
                                    v2.RandomHorizontalFlip(p=0.5),
                                    v2.RandomVerticalFlip(p=0.5),
                                    v2.ToPureTensor()
                                ])
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # get event number: index from annotations.json
        anns_i = self.data[str(index)] 

        # load pytorch tensor from annotations path
        img = torch.load(anns_i["image"]["img_path"])
        img[0, :, :] /= 1000 # rescale sum_pt into GeV
        img[1, :, :] /= 1000 # rescale max_pt into GeV
        img[4, :, :] /= 1000 # rescale max_noise into GeV
        img = img.type('torch.FloatTensor') # correct RunTime error DoubleTensor vs FloatTensor

        # Check if there are bounding boxes with width & height > 0
        bboxes = torch.tensor(anns_i["anns"]["bboxes"], dtype=torch.float32)
        height_width_mask = (bboxes[:,2] > 0) & (bboxes[:,3] > 0)
        bboxes = bboxes[height_width_mask]

        # turn boxes from xywh to x1,y1,x2,y2
        bboxes[:,2] = bboxes[:,0] + bboxes[:,2] 
        bboxes[:,3] = bboxes[:,1] + bboxes[:,3] 

        if len(bboxes)==0:
            bboxes = torch.tensor([[-0.4,-0.4,0.4,0.4]], dtype=torch.float32)
            labels = torch.tensor([0], dtype=torch.int64)
        else:
            labels = torch.ones(bboxes.shape[0], dtype=torch.int64)
        
        if self.rnd_flips:
            # Add random vertical/horizontal flip!
            bboxes = torchvision.tv_tensors.BoundingBoxes(bboxes,format="XYXY",canvas_size=img.shape[-2:])
            img, bboxes = self.transforms(img, bboxes)
        
        event_no   = anns_i["image"]["id"]
        h5file     = anns_i["image"]["file"]
        h5event    = anns_i["image"]["event"]
        pT         = anns_i["anns"]["jet_pt"]
        mc_event_w = anns_i["anns"]["mc_event_weight"]
        extent     = anns_i["anns"]["extent"]
        extent_tensor = torch.tensor(extent).float()

        if not self.truth_info:
            return img, {'boxes': bboxes, 'labels': labels, 'jet_pt': pT, 'extent': extent_tensor, 'event_weight': mc_event_w, 'h5file': h5file, 'h5event': h5event, 'event_no': event_no}
        else:
            # Same checks for truth jets
            truth_bboxes = torch.tensor(anns_i["anns"]["truth_jet_boxes"], dtype=torch.float32)
            height_width_mask = (truth_bboxes[:,2] > 0) & (truth_bboxes[:,3] > 0)
            truth_bboxes = truth_bboxes[height_width_mask]

            # turn truth boxes from xywh to x1,y1,x2,y2
            truth_bboxes[:,2] = truth_bboxes[:,0] + truth_bboxes[:,2] 
            truth_bboxes[:,3] = truth_bboxes[:,1] + truth_bboxes[:,3] 
            truth_pt = anns_i["anns"]["truth_jet_pt"]
            return img, {'akt_boxes': bboxes, 'akt_labels': labels, 'akt_jet_pt': pT, 'truth_boxes': truth_bboxes, 'truth_jet_pt': truth_pt, 'extent': extent_tensor, 'event_weight': mc_event_w, 'h5file': h5file, 'h5event': h5event, 'event_no': event_no}
            

    def collate_fn(self,batch):
        images, targets = zip(*batch) 
        images = torch.stack(images, dim=0)
        return images, targets



dataset = CustomDataset(annotation_file=ann_file, rnd_flips=False, truth_info=True)
train_len = int(0.01 * len(dataset))
val_len   = int(0.01 * len(dataset))
test_len  = len(dataset) - train_len - val_len
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])
print('\ttrain / val / test size : ',train_len,'/',val_len,'/',test_len,'\n')

train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=dataset.collate_fn, batch_size=config["BS"], shuffle=True, drop_last=True, num_workers=config["NW"])
val_loader = torch.utils.data.DataLoader(val_dataset, collate_fn=dataset.collate_fn, batch_size=config["BS"], shuffle=False, drop_last=True, num_workers=config["NW"])
test_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=dataset.collate_fn, batch_size=config["BS"], shuffle=False, drop_last=True, num_workers=config["NW"])



#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################


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
            self.conv.weight = nn.Parameter(self.pool_mask.clone())

        # Freeze the weights so that they are not updated during backpropagation
        self.conv.weight.requires_grad = False

    def forward(self, x):
        x = x.float()  # convert to float 
        
        x = F.pad(x, (0, 0, self.padding, self.padding), mode='circular')
        x = F.pad(x, (self.padding, self.padding, 0, 0), mode='constant', value=0)
        
        with torch.no_grad():
            return self.conv(x)


class CustomPad(torch.nn.Module):
    def __init__(self, kernel_size, stride=1):
        # Custom pad layer that maintains image size,
        # via custom padding (cyclic in y-axis, zeros in x-axis)
        super(CustomPad, self).__init__()
        self.padding = (kernel_size - 1) // 2

    def forward(self, x):
        # print(x.shape)
        # cyclic padding on the y-axis
        x = F.pad(x, (0, 0, self.padding, self.padding), mode='circular')

        # zero padding on the x-axis 
        x = F.pad(x, (self.padding, self.padding, 0, 0), mode='constant', value=0)
        # print(x.shape)
        return x


class CustomPool(torch.nn.Module):
    def __init__(self, output_size, p=3, eps=1e-6):
        # Custom pool layer that learns the 
        # via custom padding (cyclic in y-axis, zeros in x-axis)
        # The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        # - At p = infinity, one gets Max Pooling
        # - At p = 1, one gets Average Pooling
        # The output is of size H x W, for any input size.
        super(CustomPool, self).__init__()
        assert p > 0
        self.p = nn.Parameter(torch.ones(1)*p)
        self.output_size = output_size
        self.eps = eps
    
    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)



class custom_ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, layer_scale=1e-6):
        # Simple implementation of convnext block. See https://arxiv.org/abs/2201.03545 
        # Utilising custom layernorm (taken from https://pytorch.org/vision/main/_modules/torchvision/models/convnext.html)
        # See also https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py 
        # for alternative implementation details
        super(custom_ConvNeXtBlock, self).__init__()
        self.gelu = nn.GELU()

        self.circ_pad = CustomPad(kernel_size=kernel_size)
        # depthwise conv, now with custom padding
        self.conv_d9x9 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, groups=in_channels) 
        self.ln = LayerNorm2d(in_channels)
        # Separate "downsampling" layers 1x1 kernels
        self.conv_1x1_1 = nn.Conv2d(in_channels, in_channels*2, kernel_size=1, stride=1, padding=0, bias=True)
        self.gelu = nn.GELU()
        self.conv_1x1_2 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.layer_scale = nn.Parameter(torch.ones(in_channels, 1, 1) * layer_scale)

    def forward(self, inp):
        out1 = self.circ_pad(inp)
        out2 = self.conv_d9x9(out1)
        out3 = self.ln(out2)
        
        out4 = self.conv_1x1_1(out3)
        out5 = self.gelu(out4)

        out6 = self.layer_scale * self.conv_1x1_2(out5)

        ret = out6 + inp
        return ret

class custom_ConvNeXt_central(nn.Module):
    def __init__(self, num_channels=3, hidden_channels=12):
        super(custom_ConvNeXt_central, self).__init__()

        self.gelu = nn.GELU()
        # STEM BLOCK (125,49)->(125,49)
        self.pad1 = CustomPad(kernel_size=9)
        self.conv1 = nn.Conv2d(num_channels, hidden_channels, kernel_size=9, stride=1, bias=True)
        self.ln1 = LayerNorm2d(hidden_channels)

        # DOWN-RES 1 (62, 24)
        # self.pool = nn.MaxPool2d(2)
        self.pool1 = CustomPool((62,24), p=3, eps=1e-6)
        # BLOCK-2 at (62,24)
        # self.pad2 = CustomPad(kernel_size=7)
        self.block2 = custom_ConvNeXtBlock(in_channels=hidden_channels, out_channels=int(hidden_channels/2),kernel_size=7)

        # DOWN-RES 2 (31, 12)
        self.pool2 = CustomPool((31,12), p=3, eps=1e-6)
        # BLOCK-3 down to (28,28)
        self.pad3 = CustomPad(kernel_size=5)
        self.block3 = custom_ConvNeXtBlock(in_channels=hidden_channels, out_channels=int(hidden_channels/2),kernel_size=5)

        # UP-RES 1 (62, 24)
        # self.up1 = nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=2, stride=2, padding=0)
        self.block4 = custom_ConvNeXtBlock(in_channels=hidden_channels*2, out_channels=int(hidden_channels/2),kernel_size=5)

        # UP-RES 2 (125, 49)
        # self.up2 = nn.ConvTranspose2d(in_channels=hidden_channels*2, out_channels=hidden_channels*2, kernel_size=2, stride=2, padding=0, output_padding=(1,1))
        self.block5 = custom_ConvNeXtBlock(in_channels=hidden_channels*3, out_channels=hidden_channels,kernel_size=7)


    def forward(self,x):
        # print('Input shape',x.shape)

        # stem
        out1 = self.gelu(self.ln1(self.conv1(self.pad1(x))))
        # print('End of stem',out1.shape)

        # block 2
        out2 = self.pool1(out1)
        out2 = self.block2(out2)
        # print('End of block 2',out2.shape)

        # block 3
        out3 = self.pool2(out2)
        out3 = self.block3(out3)
        # print('End of block 3',out3.shape)

        # block 4
        out4 = F.interpolate(out3, size=[62,24], mode='bilinear', align_corners=True) 
        # out4 = self.up1(out3)
        out4 = torch.cat([out4,out2],dim=1)
        out4 = self.block4(out4)
        # print('End of block 4',out4.shape)
        
        # block 5
        out5 = F.interpolate(out4, size=[125,49], mode='bilinear', align_corners=True) 
        # out5 = self.up2(out4)
        out5 = torch.cat([out5,out1],dim=1)
        out5 = self.block5(out5)
        # print('End of block 5',out5.shape)
        
        return out5



class test_ConvNeXt_central(nn.Module):
    def __init__(self, num_channels=3, hidden_channels=12):
        super(test_ConvNeXt_central, self).__init__()

        self.gelu = nn.GELU()
        # STEM BLOCK (125,49)->(125,49)
        self.pad1 = CustomPad(kernel_size=9)
        self.conv1 = nn.Conv2d(num_channels, hidden_channels, kernel_size=9, stride=1, bias=False)
        self.ln1 = LayerNorm2d(hidden_channels)


        self.block2 = custom_ConvNeXtBlock(in_channels=hidden_channels, out_channels=int(hidden_channels/2),kernel_size=9)
        self.pad2 = CustomPad(kernel_size=9)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=9, stride=1, bias=True)
        self.ln2 = LayerNorm2d(hidden_channels)

        self.block3 = custom_ConvNeXtBlock(in_channels=hidden_channels, out_channels=hidden_channels*2,kernel_size=9)
        self.pad3 = CustomPad(kernel_size=9)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=9, stride=1, bias=True)
        self.ln3 = LayerNorm2d(hidden_channels)

        self.block4 = custom_ConvNeXtBlock(in_channels=hidden_channels*2, out_channels=hidden_channels*2,kernel_size=9)

    def forward(self,x):
        # print('Input shape',x.shape)

        # stem
        out1 = self.gelu(self.ln1(self.conv1(self.pad1(x))))
        # print('End of stem',out1.shape)

        # block 2
        out2 = self.ln2(self.conv2(self.pad2(self.block2(out1))))
        # print('End of block 2',out2.shape)

        # block 3
        out3 = self.ln3(self.conv3(self.pad3(self.block3(out2))))
        # print('End of block 3',out3.shape)

        # block 4
        out4 = torch.cat([out3,out1],dim=1)
        out4 = self.block4(out4)
        # print('End of block 4',out4.shape)
        
        return out4




class smallConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, layer_scale=1e-6):
        # Simple implementation of convnext block. See https://arxiv.org/abs/2201.03545 
        # Utilising custom layernorm (taken from https://pytorch.org/vision/main/_modules/torchvision/models/convnext.html)
        # See also https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py 
        # for alternative implementation details
        super(smallConvNeXtBlock, self).__init__()
        self.gelu = nn.GELU()

        # depthwise conv, now with circular padding
        self.conv_d9x9 = nn.Conv2d(in_channels, in_channels, kernel_size=9, stride=1, padding=4, padding_mode='circular', groups=in_channels) 
        self.ln = LayerNorm2d(in_channels)
        # Separate "downsampling" layers 1x1 kernels
        self.conv_1x1_1 = nn.Conv2d(in_channels, in_channels*2, kernel_size=1, stride=1, padding=0, bias=True)
        self.gelu = nn.GELU()
        self.conv_1x1_2 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.layer_scale = nn.Parameter(torch.ones(in_channels, 1, 1) * layer_scale)

    def forward(self, input):

        out = self.conv_d9x9(input)
        out = self.ln(out)
        
        out = self.conv_1x1_1(out)
        out = self.gelu(out)

        out = self.layer_scale * self.conv_1x1_2(out)

        ret = out + input
        return ret


class smallConvNeXt_central(nn.Module):
    def __init__(self, num_channels=3, hidden_channels=8, num_classes=1000):
        super(smallConvNeXt_central, self).__init__()
        
        self.gelu = nn.GELU()
        # STEM CELL BLOCK (125,49)->(125,49)
        self.conv1 = nn.Conv2d(num_channels, hidden_channels, kernel_size=9, stride=1, padding=4, padding_mode='circular', bias=True)
        self.ln1 = LayerNorm2d(hidden_channels)

        # BLOCK-2 (56,56)
        self.conv2_1 = smallConvNeXtBlock(in_channels=hidden_channels)
        self.channel_res2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1)

        # BLOCK-3 
        self.conv3_1 = smallConvNeXtBlock(in_channels=hidden_channels)
        self.channel_res3 = nn.Conv2d(hidden_channels, hidden_channels*2, kernel_size=1, stride=1)

        # BLOCK-4 
        self.conv4_1 = smallConvNeXtBlock(in_channels=hidden_channels*2)
        self.conv4_2 = smallConvNeXtBlock(in_channels=hidden_channels*2) # here's where we cut for SSD. reduce channels 32->24

        # in-between res4->res5 down sampling (14,14)->(7,7)
        self.down_ln4 = LayerNorm2d(hidden_channels*3)
        self.down_res4 = nn.Conv2d(hidden_channels*3, hidden_channels*8, kernel_size=1, stride=2) #should be kernel size 2

        # BLOCK-5 (7,7)
        self.conv5_1 = smallConvNeXtBlock(in_channels=hidden_channels*8)

        #global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.ln6 = LayerNorm2d(hidden_channels*8)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(hidden_channels*8,num_classes)

    def forward(self,x):
        # print('Input shape',x.shape)
        #stem
        out = self.gelu(self.ln1(self.conv1(x)))
        # print('End of stem',out.shape)

        #conv2
        out = self.conv2_1(out)
        out = self.channel_res2(out)
        # print('End of conv2',out.shape)

        #conv3
        out = self.conv3_1(out)              
        out = self.channel_res3(out)
        # print('End of conv3',out.shape)

        #conv4
        out = self.conv4_1(out)             
        out = self.conv4_2(out)                                    
        out = self.down_res4(self.down_ln4(out))
        # print('End of conv4',out.shape)

        #conv5
        out = self.conv5_1(out)            
        # print('End of conv5',out.shape)

        #output
        out = self.avg_pool(out)
        out = self.ln6(out)
        out = self.fc(self.flat(out))
        # print('Final output',out.shape)
        
        return out










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

        elif name=="custom_convnext_central":
            self.feature_extractor = custom_ConvNeXt_central(num_channels=in_channels,hidden_channels=hidden_channels)
            self.out_channels = [hidden_channels*3]

        elif name=="test_ConvNeXt_central":
            self.feature_extractor = test_ConvNeXt_central(num_channels=in_channels,hidden_channels=hidden_channels)
            self.out_channels = [hidden_channels*2]

        elif name=="smallconvnext_central":
            backbone = smallConvNeXt_central(num_channels=in_channels,hidden_channels=hidden_channels)
            self.out_channels = [hidden_channels*2] 
            # print(list(backbone.children())[:8]) # 16
            self.feature_extractor = torch.nn.Sequential(*list(backbone.children())[:8])
        
    def forward(self, x):
        return self.feature_extractor(x)



class SSD(torch.nn.Module):
    def __init__(self, backbone_name, in_channels=10, diamond_mask=True):
        super().__init__()

        # grab chosen feature extractor
        self.backbone_name = backbone_name
        self.feature_extractor = CustomFeatureExtractor(in_channels=in_channels*2, hidden_channels=20, name=backbone_name)
        print(f"Backbone model:    {sum(p.numel() for p in self.feature_extractor.parameters()):,} parameters.")

        self.label_num = 1 
        self._build_additional_features(self.feature_extractor.out_channels)
        self.num_defaults = [1]
        
        self.loc = nn.Conv2d(self.aux_channels, self.num_defaults[0] * 4, kernel_size=3, padding=1)
        self.conf = nn.Conv2d(self.aux_channels, self.num_defaults[0] * self.label_num, kernel_size=3, padding=1)
        print(f"Loc head:          {sum(p.numel() for p in self.loc.parameters()):,} parameters.")
        print(f"Conf head:         {sum(p.numel() for p in self.conf.parameters()):,} parameters.")

        custom_pool_mask = torch.tensor([[0, 0, 0, 1, 1, 1, 0, 0, 0],
                                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                                         [0, 1, 1, 1, 1, 1, 1, 1, 0],
                                         [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                         [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                         [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                         [0, 1, 1, 1, 1, 1, 1, 1, 0],
                                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                                         [0, 0, 0, 1, 1, 1, 0, 0, 0],],dtype=torch.float32)
        pool_mask = custom_pool_mask if diamond_mask else torch.ones(9,9)
        print("POOL MASK",pool_mask)
        self.sumpool = MaskSumPool(kernel_size=9, in_channels=in_channels, stride=1, pool_mask=pool_mask)
        print(f"PT map:            {sum(p.numel() for p in self.sumpool.parameters()):,} (frozen) parameters. Custom pooling mask kernel.")
        
        self._init_weights()

    def _build_additional_features(self, input_size, hidden_channels=12):
        
        # self.aux_channels = int(hidden_channels/2)
        # self.additional_blocks = nn.Sequential(
        #         nn.Conv2d(input_size[0], int(self.aux_channels*3), kernel_size=9, padding=4, padding_mode='circular', stride=1, bias=False),
        #         LayerNorm2d(int(self.aux_channels*3)),
        #         nn.GELU(),
        #         nn.Conv2d(int(self.aux_channels*3), self.aux_channels, kernel_size=7, padding=3, padding_mode='circular', stride=1, bias=False),
        #         LayerNorm2d(self.aux_channels),
        #         nn.Conv2d(self.aux_channels, self.aux_channels, kernel_size=3, padding=(1,0), padding_mode='circular', stride=2, bias=False),
        #         nn.GELU(),
        # )
        self.aux_channels = int(hidden_channels)
        aux_layers = nn.Sequential(
                nn.Conv2d(input_size[0], self.aux_channels, kernel_size=5, padding=2, padding_mode='circular', stride=1, bias=False),
                LayerNorm2d(self.aux_channels),
                nn.GELU(),
                nn.Conv2d(self.aux_channels, self.aux_channels, kernel_size=3, padding=1, padding_mode='circular', stride=1, bias=False),
                LayerNorm2d(self.aux_channels),
                nn.Conv2d(self.aux_channels, self.aux_channels, kernel_size=3, padding=(1,0), padding_mode='circular', stride=2, bias=False),
                nn.SELU(),
        )

        self.additional_blocks = nn.ModuleList([aux_layers])

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
        # print('input image shape',x.shape)

        ptmap = self.sumpool(x)
        # print('ptmap shape',ptmap.shape)
        x = torch.cat((x, ptmap), dim=1)
        # print('concat image+sumpool',x.shape)

        x = self.feature_extractor(x)
        # print('After feature ext.',x.shape)     

        for l in self.additional_blocks:
            x = l(x)
        # x = self.additional_blocks(x)
        # print('After additional blocks',x.shape)

        # print('feature maps [125,96] reshape->',125*96, 'but we downsample the image in aux layers to to reduce to [63,48]',63*48,'equal to our step_x, step_y=2' )
        locs, confs = self.bbox_view(x, self.loc, self.conf)
        # print('After bbox_view',confs.shape,locs.shape)

        return locs, confs, ptmap[:,0,:,:]




#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################


# instantiate 
# load trained model
model = SSD(backbone_name=backbone,in_channels=5,diamond_mask=True)
model = model.to(config["device"]) 
model_name = "jetSSD_{}_{}e".format(backbone,config["n_epochs"])
model_save_path = model_dir + f"/{model_name}.pth"
model.load_state_dict(torch.load(model_save_path, map_location=torch.device(config["device"])))   
total_params = sum(p.numel() for p in model.parameters())
print(model.backbone_name, f'\t{total_params:,} total! parameters.\n')
model.eval()


# default prior boxes
dboxes = data.DefaultBoxes(figsize=(24,63),scale=(3.84, 4.05),step_x=1,step_y=1)
print("Generated prior boxes, ",dboxes.dboxes.shape, ", default boxes", dboxes.dboxes.device)

# encoder and loss
encoder = data.Encoder(dboxes)


save_loc = output_dir + "/" + model_name + "/" + proc + "/" + time.strftime("%Y%m%d-%H") + "/"
print("Save location: ", save_loc)
if not os.path.exists(save_loc): os.makedirs(save_loc)

# let's infer on all events in the test set and store the results in a numpy structured array
# with the following data types:
# event_no: int, h5file: int, img: numpy array?, ground truth boxes: list, predicted_boxes: list, predicted_scores: list, predicted_pt (sumpool): list, extent
beginning = time.perf_counter()
dt = np.dtype([('event_no', 'i4'), ('event_weight', 'f4'), ('h5file', 'S2'), ('h5event', 'i4'), ('extent', 'f8', (4)),  #S2 for a string of length exactly 2
                ('tar_boxes', 'f4', (250,4)), ('tar_pt', 'f4', (250)), 
                ('tru_boxes', 'f4', (100,4)), ('tru_pt', 'f4', (100)), 
                ('p_boxes', 'f4', (config["max_num"], 4)), ('p_scores', 'f4', (config["max_num"])), ('p_pt', 'f4', (config["max_num"]))])
BS = config["BS"]
Large = np.zeros((len(test_loader)*BS), dtype=dt)    
with torch.inference_mode():
    for step, (batch_imgs,targets) in enumerate(test_loader):
        img_tensor = batch_imgs.to(config["device"]).float()

        locs,conf,ptmap = model(img_tensor)

        # define NMS scriteria, confidence threshold
        output = encoder.decode_batch(locs, conf, ptmap, 
                                        iou_thresh=0.25, #NMS
                                        confidence=0.45, #conf threshold
                                        max_num=config["max_num"]) #155

        boxes, labels, scores, pts = zip(*output)

        #remove from GPU
        tar_boxes,extents,h5files,h5events,event_nos,event_weights,tar_pt = [], [], [], [], [], [], []
        tru_boxes,tru_pt = [], []
        det_boxes, det_scores, det_pts = [], [], []
        for i in range(BS):
            extent_i = targets[i]["extent"].detach().cpu().numpy()
            # make target/truth boxes cover extent
            tar_boxes_ext = targets[i]['akt_boxes'].detach().cpu().numpy()
            tar_boxes_ext[:,(0,2)] = (tar_boxes_ext[:,(0,2)]*((extent_i[1]-extent_i[0])/img_tensor[i].shape[2]))+extent_i[0]
            tar_boxes_ext[:,(1,3)] = (tar_boxes_ext[:,(1,3)]*((extent_i[3]-extent_i[2])/img_tensor[i].shape[1]))+extent_i[2]
            tar_pts = targets[i]['akt_jet_pt']

            tru_boxes_ext = targets[i]['truth_boxes'].detach().cpu().numpy()
            tru_boxes_ext[:,(0,2)] = (tru_boxes_ext[:,(0,2)]*((extent_i[1]-extent_i[0])/img_tensor[i].shape[2]))+extent_i[0]
            tru_boxes_ext[:,(1,3)] = (tru_boxes_ext[:,(1,3)]*((extent_i[3]-extent_i[2])/img_tensor[i].shape[1]))+extent_i[2]
            tru_pts = targets[i]['truth_jet_pt']

            tar_boxes.append(tar_boxes_ext)
            tar_pt.append(tar_pts)
            tru_boxes.append(tru_boxes_ext)
            tru_pt.append(tru_pts)
            extents.append(extent_i)
            h5files.append(targets[i]["h5file"])
            h5events.append(targets[i]["h5event"])
            event_nos.append(targets[i]["event_no"])
            event_weights.append(targets[i]["event_weight"])

            # make pred boxes cover extent
            det_boxes_scr = scores[i].detach().cpu().numpy()
            det_boxes_pts = pts[i].detach().cpu().numpy()
            det_boxes_ext = boxes[i].detach().cpu().numpy()
            det_boxes_ext[:,(0,2)] = (det_boxes_ext[:,(0,2)]*((extent_i[1]-extent_i[0])))+extent_i[0]
            det_boxes_ext[:,(1,3)] = (det_boxes_ext[:,(1,3)]*((extent_i[3]-extent_i[2])))+extent_i[2]

            # remember ALL targets have width/height 0.8
            # mask out boxes that have width and height > 1.3 (== radius >0.65)
            mask_too_big = (det_boxes_ext[:,2] - det_boxes_ext[:,0] < 1.3) & (det_boxes_ext[:,3] - det_boxes_ext[:,1] < 1.3)
            det_boxes_ext = det_boxes_ext[mask_too_big]
            det_boxes_scr = det_boxes_scr[mask_too_big]
            det_boxes_pts = det_boxes_pts[mask_too_big]
            # new!
            # mask out boxes that have  height <0.5 (== radius <0.25)
            mask_too_small = (det_boxes_ext[:,3] - det_boxes_ext[:,1] > 0.5)
            det_boxes_ext = det_boxes_ext[mask_too_small]
            det_boxes_scr = det_boxes_scr[mask_too_small]
            det_boxes_pts = det_boxes_pts[mask_too_small]

            det_boxes.append(det_boxes_ext)
            det_scores.append(det_boxes_scr)
            det_pts.append(det_boxes_pts)

            # ###############################
            # import matplotlib.pyplot as plt
            # import matplotlib
            # # det_boxes_ext,det_boxes_scr,det_boxes_pts = wrap_check_NMS3(det_boxes_ext,det_boxes_scr,det_boxes_pts,iou_thresh=0.3)
            # # tru_boxes_ext,tru_pts = wrap_check_truth2(torch.tensor(tru_boxes_ext),torch.tensor(targets[i]['jet_pt']),MIN_CELLS_PHI,MAX_CELLS_PHI)
            # f,ax = plt.subplots(1,1,figsize=(10,12))   
            # img = img_tensor[i].detach().cpu().numpy()
            # # img = original_images[i].detach().cpu().numpy()
            # ax.imshow(img[0],cmap='binary_r',extent=extent_i,origin='lower')
        
            # ax.axhline(y=MIN_CELLS_PHI, color='red', alpha=0.6, linestyle='--',lw=0.7)
            # ax.axhline(y=MAX_CELLS_PHI, color='red', alpha=0.6, linestyle='--',lw=0.7)
    
            # for i in range(len(tru_boxes_ext)):
            #     bbx,pt = tru_boxes_ext[i],tru_pts[i]
            #     x,y=float(bbx[0]),float(bbx[1])
            #     w,h=float(bbx[2])-float(bbx[0]),float(bbx[3])-float(bbx[1])  
            #     ax.add_patch(matplotlib.patches.Rectangle((x,y),w,h,lw=1.8,ec='gold',fc='none'))
            #     ax.text(x+0.05,y+h-0.15, f"{pt:.0f}",color='gold',fontsize=8)
    
            # for k in range(len(tar_boxes_ext)):
            #     bbx,pt = tar_boxes_ext[k],tar_pts[k]
            #     x,y=float(bbx[0]),float(bbx[1])
            #     w,h=float(bbx[2])-float(bbx[0]),float(bbx[3])-float(bbx[1])  
            #     ax.add_patch(matplotlib.patches.Rectangle((x,y),w,h,lw=1.8,ec='limegreen',fc='none'))
            #     ax.text(x+0.05,y+h-0.15, f"{pt:.0f}",color='limegreen',fontsize=8)

            # for j in range(len(det_boxes_ext)):
            #     bbx,scr,pt = det_boxes_ext[j],det_boxes_scr[j],det_boxes_pts[j]
            #     x,y=float(bbx[0]),float(bbx[1])
            #     w,h=float(bbx[2])-float(bbx[0]),float(bbx[3])-float(bbx[1])  
            #     ax.add_patch(matplotlib.patches.Rectangle((x,y),w,h,lw=1.9,ec='red',fc='none'))
            #     ax.text(x+w-0.3,y+h-0.15, f"{scr.item():.2f}",color='red',fontsize=8)
            #     ax.text(x+0.05,y+h/20, f"{pt.item():.0f}",color='red',fontsize=8)

            # ax.set(xlabel='$\eta$',ylabel='$\phi$',xlim=(extent_i[0],extent_i[1]),ylim=(extent_i[2],extent_i[3]))
            # plt.tight_layout()
            # f.savefig(save_loc+f'ex-NMS-{step*BS + i}-ttbar.png',dpi=400)
            # plt.close()
            # print(step*BS + i)
            # print("\t",len(tru_boxes_ext),len(tar_boxes_ext),len(det_boxes_ext),len(det_boxes_pts))
            # if (step*BS + i) == 16:
            #     quit()
            # ###############################

        print(step)

        dataset_idx = step*BS
        Large['event_no'][dataset_idx:dataset_idx+BS] = event_nos
        Large['event_weight'][dataset_idx:dataset_idx+BS] = event_weights
        Large['h5file'][dataset_idx:dataset_idx+BS] = h5files
        Large['h5event'][dataset_idx:dataset_idx+BS] = h5events
        Large['extent'][dataset_idx:dataset_idx+BS] = extents  
    
        tar_boxes = [np.pad(tarb, ((0,250-len(tarb)),(0,0)), 'constant', constant_values=(0)) for tarb in tar_boxes]
        tar_pt = [np.pad(tarb, ((0,250-len(tarb))), 'constant', constant_values=(0)) for tarb in tar_pt]
        tru_jet_boxes = [np.pad(trub, ((0,100-len(trub)),(0,0)), 'constant', constant_values=(0)) for trub in tru_boxes]
        tru_jet_pt = [np.pad(trub, ((0,100-len(trub))), 'constant', constant_values=(0)) for trub in tru_pt]
        p_boxes = [np.pad(preb, ((0,config["max_num"]-len(preb)),(0,0)), 'constant', constant_values=(0)) for preb in det_boxes]
        p_scores = [np.pad(pres, ((0,config["max_num"]-len(pres))), 'constant', constant_values=(0)) for pres in det_scores]
        p_pts = [np.pad(prept, ((0,config["max_num"]-len(prept))), 'constant', constant_values=(0)) for prept in det_pts]
    
        Large['tar_boxes'][dataset_idx:dataset_idx+BS] = tar_boxes   
        Large['tar_pt'][dataset_idx:dataset_idx+BS] = tar_pt   
        Large['tru_boxes'][dataset_idx:dataset_idx+BS] = tru_jet_boxes   
        Large['tru_pt'][dataset_idx:dataset_idx+BS] = tru_jet_pt   
        Large['p_boxes'][dataset_idx:dataset_idx+BS] = p_boxes   
        Large['p_scores'][dataset_idx:dataset_idx+BS] = p_scores 
        Large['p_pt'][dataset_idx:dataset_idx+BS] = p_pts 

end = time.perf_counter()      
print(f"Time taken for entire test set: {(end-beginning)/60:.3f} mins, (or {(end-beginning):.3f}s), average {(end-beginning)/test_len:.4f} per image")

print('\n\n')
with open(save_loc+'struc_array.npy', 'wb') as f:
    print('Saving...')
    np.save(f, Large)
