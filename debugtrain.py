import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import v2
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import matplotlib
import matplotlib.pyplot as plt
import json

import time
from statistics import mean

import data





# https://pytorch.org/docs/stable/notes/randomness.html
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

# ann_file="/srv/beegfs/scratch/shares/atlas_caloM/mu_200_truthjets/central_2sig_images/anns_central_jets_JZcomb0.json"
# ann_file="/home/users/b/bozianu/work/data/mu200/anns_central_jets_20GeV.json"
# ann_file="/srv/beegfs/scratch/shares/atlas_caloM/mu_200_truthjets/central_2sig_images/anns_central_jets_JZ4.2.json" # works
ann_file="/srv/beegfs/scratch/shares/atlas_caloM/mu_200_truthjets/central_2sig_images/anns_central_jets_truth_JZcomb0_train.json"
save_dir="/home/users/b/bozianu/work/paperSSD/customSSD/saved_models/"
# backbone = "smallconvnext_central"
backbone = "custom_convnext_central"
config = {
    "seed"       : 0,
    "device"     : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "NW"         : 0,
    "BS"         : 8,
    "LR"         : 0.01,
    "WD"         : 0.01,
    "wup_epochs" : int(25/3),
    "n_epochs"   : int(25),
}
torch.manual_seed(config["seed"])






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



dataset = CustomDataset(annotation_file=ann_file, rnd_flips=True)
train_len = int(0.8 * len(dataset))
val_len = int(0.01 * len(dataset))
test_len = len(dataset) - train_len - val_len
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])
print('\ttrain / val / test size : ',train_len,'/',val_len,'/',test_len,'\n')

train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=dataset.collate_fn, batch_size=config["BS"], shuffle=True, drop_last=True, num_workers=config["NW"])
val_loader   = torch.utils.data.DataLoader(val_dataset, collate_fn=dataset.collate_fn, batch_size=config["BS"], shuffle=False, drop_last=True, num_workers=config["NW"])
test_loader  = torch.utils.data.DataLoader(test_dataset, collate_fn=dataset.collate_fn, batch_size=config["BS"], shuffle=False, drop_last=True, num_workers=config["NW"])


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




class NewLoss(torch.nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels taken into account, using focal loss with chosen alpha/gamma hyperparams
        See https://arxiv.org/pdf/1708.02002 or https://amaarora.github.io/posts/2020-06-29-FocalLoss.html
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """
    def __init__(self, dboxes, scalar=1.0, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(NewLoss, self).__init__()
        self.scalar = scalar
        self.device = device

        # REG LOSS
        self.sl1_loss = nn.SmoothL1Loss(reduction='none')
        self.giou_loss = torchvision.ops.generalized_box_iou_loss
        self.ciou_loss = torchvision.ops.complete_box_iou_loss
        self.diou_loss = torchvision.ops.distance_box_iou_loss

        # CLF LOSS
        self.con_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.foc_loss = torchvision.ops.focal_loss.sigmoid_focal_loss
        # self.vfoc_loss = VariFocalLoss(alpha=0.25,gamma=3,reduction='sum')

        self.dboxes = nn.Parameter(dboxes(order="xywh").transpose(0, 1).unsqueeze(dim = 0).to(device),requires_grad=False)

    def xywh2xyxy(self, boxes, i=0):
        """
            Convert boxes xywh to xyxy 
        """
        boxes = boxes.clone()
        boxes[..., i:i + 2] -= boxes[..., i + 2:i + 4] / 2
        boxes[..., i + 2:i + 4] += boxes[..., i:i + 2]
        return boxes

    def _loc_vec(self, loc):
        """
            Parameterise Location Vectors
        """
        gxy = (loc[:, :2, :] - self.dboxes[:, :2, :]) / self.dboxes[:, 2:, ]
        gwh = torch.log(loc[:, 2:, :]/self.dboxes[:, 2:, :])
        return torch.cat((gxy, gwh), dim=1).contiguous()

    def forward(self, ploc, plabel, gloc, glabel):
        """
            ploc, plabel: Nx4x8732, Nxlabel_numx8732
                predicted location and labels

            gloc, glabel: Nx4x8732, Nx8732
                ground truth location and labels
        """

        mask = glabel > 0 
        pos_num = mask.sum(dim=1) 
        num_mask = (pos_num > 0).float() # does the image contain any positive anchors
        # print(pos_num,glabel.sum(dim=1))

        # Box Regression Loss (Smooth L1)
        vec_gd = self._loc_vec(gloc)
        sl1 = self.sl1_loss(ploc, vec_gd).sum(dim=1)
        # print('2. SL1 normal ?',sl1)
        # print('3. SL1 * num_mask ?', sl1.sum(dim=1)*num_mask)
        sl1 = (mask.float()*sl1).sum(dim=1)
        # print('4.SL1 * mask ?',sl1)
        # sl1[sl1==0] = 10
        # print('5.SL1 +10 ?',sl1)
        # print(pos_num, num_mask, mask.sum(dim=1))
        # print()
        if torch.isnan(sl1).any():
            print('SL1 nan :(',sl1, sl1*num_mask)
            print(pos_num, num_mask, mask.sum(dim=1))
            print(ploc.shape,plabel.shape,gloc.shape,glabel.shape)
            quit()
        # print("SL1 loss:        ", sl1)
        # tensor([  0, 127,  68, 279,  97,  62,  54, 164], device='cuda:0') tensor([0., 1., 1., 1., 1., 1., 1., 1.], device='cuda:0') tensor([  0, 127,  68, 279,  97,  62,  54, 164], device='cuda:0')

        # 1. target boxes (first image in batch)!  tensor([[ 21.6451,  22.7307,  29.4852,  30.7722],

        plabel = plabel.squeeze(1).float()
        glabel = glabel.float()
        # Classification Loss (Focal)
        floss = self.foc_loss(plabel,glabel,alpha=0.25,gamma=3,reduction='sum')
        if torch.isnan(floss).any():
            print('focal loss nan :(',floss)
            print(pos_num, num_mask)
            print(ploc.shape,plabel.shape,gloc.shape,glabel.shape)
            quit()
        # normalise by the number of anchors assigned to a ground truth box
        floss = floss / pos_num
        # print("Focal loss:      ", floss)

        # Classification Loss (BCE)
        con = self.con_loss(plabel, glabel)
        con_neg = con.clone() 
        con_neg[mask] = 0
        _, con_idx = con_neg.sort(dim=1, descending=True)
        _, con_rank = con_idx.sort(dim=1)
        neg_num = torch.clamp(3*pos_num, max=mask.size(1)).unsqueeze(-1)
        neg_mask = con_rank < neg_num
        closs = (con*((mask + neg_mask).float())).sum(dim=1)
        # print("BCE loss:      ", closs)

        # user_response = input("Continue? Please enter y/n: ")
        # if user_response=="n": quit()

        # return (sl1*num_mask/pos_num).mean(dim=0), (floss*num_mask/pos_num).mean(dim=0)
        return (sl1*num_mask/pos_num).mean(dim=0), (closs*num_mask/pos_num).mean(dim=0)
    


#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################


# instantiate model
model = SSD(backbone_name=backbone,in_channels=5,diamond_mask=True)
model = model.to(config["device"]) 
total_params = sum(p.numel() for p in model.parameters())
print(model.backbone_name, f'\t{total_params:,} total! parameters.\n')
        
# optimizers & learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=config["LR"], weight_decay=config["WD"], amsgrad=True)  
scheduler = CosineAnnealingLR(optimizer, T_max=config["n_epochs"], eta_min=0.00)

# default prior boxes
dboxes = data.DefaultBoxes(figsize=(24,63),scale=(3.84, 4.05),step_x=1,step_y=1)
print("Generated prior boxes, ",dboxes.dboxes.shape, ", default boxes", dboxes.dboxes.device)

# encoder and loss
encoder = data.Encoder(dboxes)
# loss = models.Loss(dboxes)
newloss = NewLoss(dboxes,scalar=1.0)


print('Starting training...')
for epoch in range(config["n_epochs"]):
    beginning = time.perf_counter()

    model.train()
    running_loss = list()
    s_loss,g_loss,c_loss,f_loss,v_loss = list(),list(),list(),list(),list()
    for step, (images, target_dict) in enumerate(train_loader):
        # send data to gpu (annoying)
        images = images.to(config["device"],non_blocking=True)

        # forward pass
        plocs, plabel, ptmap = model(images) # plocs.shape(torch.Size([BS, 4, n_dfboxes])) and plabel.shape(torch.Size([BS, 1, n_dfboxes]))

        # # encode targets/default boxes
        gloc,glabel = encoder.encode_batch(target_dict, config["BS"])
        # print("1. target boxes (first image in batch)! ",target_dict[0]["boxes"])
        # print(target_dict[0]["h5file"],target_dict[0]["h5event"],target_dict[0]["event_no"])
        reg_loss, cls_loss = newloss(plocs, plabel, gloc, glabel)
        
        s_loss.append(reg_loss.item())
        f_loss.append(cls_loss.item())
        train_loss = reg_loss + cls_loss
        running_loss.append(train_loss.item())

        # back prop
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

# ###

#         output = encoder.decode_batch(plocs, plabel, ptmap, iou_thresh=0.25, confidence=0.45,max_num=155) 
#         boxes, labels, scores, pts = zip(*output)
#         # det_boxes_ext = boxes[i].detach().cpu().numpy()
#         # det_boxes_ext[:,(0,2)] = (det_boxes_ext[:,(0,2)]*((extent_i[1]-extent_i[0])))+extent_i[0]
#         # det_boxes_ext[:,(1,3)] = (det_boxes_ext[:,(1,3)]*((extent_i[3]-extent_i[2])))+extent_i[2]
#         f,ax = plt.subplots()
#         img = images.cpu().detach().numpy()
#         ii = ax.imshow(img[0][0],cmap='binary_r')
#         # ax.hlines([-np.pi,np.pi],-3,3,color='red',ls='dashed')
#         # ax.hlines([-1.9396086193266369,1.940238044375465],-3,3,color='orange',ls='dashed')
#         true_bboxes = target_dict[0]["boxes"]
#         for bbx in true_bboxes:
#             bbx = bbx.cpu().detach().numpy()
#             bb = matplotlib.patches.Rectangle((bbx[0],bbx[1]),bbx[2]-bbx[0],bbx[3]-bbx[1],lw=1,ec='limegreen',fc='none')
#             ax.add_patch(bb)
#         for bbx in boxes:
#             bbx = bbx.cpu().detach().numpy()
#             bb = matplotlib.patches.Rectangle((bbx[0],bbx[1]),bbx[2]-bbx[0],bbx[3]-bbx[1],lw=1,ec='red',fc='none')
#             ax.add_patch(bb)
#         cbar = f.colorbar(ii,ax=ax)
#         cbar.ax.get_yaxis().labelpad = 10
#         cbar.set_label('sum cell E', rotation=90)
#         ax.set(xlabel='eta',ylabel='phi')
#         f.savefig('central-image-example.png')
#         plt.close()
#         quit()
# ###

    print(f"\tEpoch {epoch} / {config['n_epochs']}: train loss {mean(running_loss):.4f}, train time {time.perf_counter() - beginning:.2f}s, LR: {optimizer.param_groups[0]['lr']:.4f}")
    # print(f"\t\tSL1 Loss: {mean(s_loss):.3f}, Focal Loss: {mean(f_loss):.3f}")
    print(f"\t\tSL1 Loss: {mean(s_loss):.3f}, BCE Loss: {mean(f_loss):.3f}")
    val_beginning = time.perf_counter()
    # validation step
    model.eval()
    with torch.inference_mode():
        running_val_loss = list()
        for step, (val_images, val_dict) in enumerate(val_loader):
            # send data to gpu (annoying)
            val_images = val_images.to(config["device"],non_blocking=True) 
        
            # forward pass
            plocs, plabel, ptmap = model(val_images) #plocs.shape(torch.Size([BS, 4, n_dfboxes])) and plabel.shape(torch.Size([BS, 1, n_dfboxes]))

            # encode val_targets/default boxes
            gloc,glabel = encoder.encode_batch(val_dict, config["BS"])

            # val_loss = loss(plocs, plabel, gloc, glabel) 
            reg_loss, cls_loss = newloss(plocs, plabel, gloc, glabel)
            val_loss = reg_loss + cls_loss
            running_val_loss.append(val_loss.item())
        
    print(f"\tEpoch {epoch} / {config['n_epochs']}: valid loss {mean(running_val_loss):.4f}, valid time {time.perf_counter() - val_beginning:.2f}s")
    
    # update LR scheduler
    scheduler.step()


# save trained model
model_name = "jetSSD_{}_{}e".format(model.backbone_name,config["n_epochs"])
print(f'Saving model now...\t{model_name}')
torch.save(model.state_dict(), save_dir+"/{}.pth".format(model_name))



########################################################################################################################



print("Finished training. Now let's look at one event and infer")
print(f"Using default {dboxes.dboxes.shape} boxes and Encoder")
model.eval()
with torch.no_grad():
    for i, (images,targets) in enumerate(test_loader):
        images = images.to(config["device"]).float()
        model = model.to(config["device"])

        locs,conf,ptmap = model(images)
        print("decoding:")
        # define NMS scriteria, confidence threshold
        output = encoder.decode_batch(locs, conf, ptmap,
                                        iou_thresh=0.3, #NMS
                                        confidence=0.4, #conf threshold
                                        max_num=150) 

        boxes, labels, scores, pts = zip(*output)

        # examine event by event
        for j, ((boxes, labels, scores, pts), img) in enumerate(zip(output, images)):
            print(j)
            print(min(scores.cpu()))
            
            # make truth boxes cover extent
            extent_j = targets[j]['extent']
            tru_boxes_ext = targets[j]['boxes'].cpu()
            tru_boxes_ext[:,(0,2)] = (tru_boxes_ext[:,(0,2)]*((extent_j[1]-extent_j[0])/img.shape[2]))+extent_j[0]
            tru_boxes_ext[:,(1,3)] = (tru_boxes_ext[:,(1,3)]*((extent_j[3]-extent_j[2])/img.shape[1]))+extent_j[2]
            
            tru_pt = targets[j]['jet_pt']
            event_number = targets[j]['event_no']
            pred_scores = scores.cpu()
            pred_pts = pts.cpu()
            pred_labels = labels.cpu()

            # make pred boxes cover extent
            det_boxes_ext = boxes.clone().cpu()
            det_boxes_ext[:,(0,2)] = (det_boxes_ext[:,(0,2)]*((extent_j[1]-extent_j[0])))+extent_j[0]
            det_boxes_ext[:,(1,3)] = (det_boxes_ext[:,(1,3)]*((extent_j[3]-extent_j[2])))+extent_j[2]
            print("\tAnti-kt jet pt: ", tru_pt)
            print("\tPred box jet pt: ", pred_pts)


            # plotting
            MIN_CELLS_PHI,MAX_CELLS_PHI = -3.1334076, 3.134037
            MIN_CELLS_ETA,MAX_CELLS_ETA = -2.5, 2.5

            f,ax = plt.subplots(1,2)
            ax[0].imshow(img[0].cpu().numpy(),cmap='binary_r',extent=extent_j,origin='lower')
            ax[1].imshow(img[1].cpu().numpy(),cmap='binary_r',extent=extent_j,origin='lower')
            
            ax[0].axhline(y=MIN_CELLS_PHI, color='red', alpha=0.6, linestyle='--',lw=0.7)
            ax[0].axhline(y=MAX_CELLS_PHI, color='red', alpha=0.6, linestyle='--',lw=0.7)
            ax[1].axhline(y=MIN_CELLS_PHI, color='red', alpha=0.6, linestyle='--',lw=0.7)
            ax[1].axhline(y=MAX_CELLS_PHI, color='red', alpha=0.6, linestyle='--',lw=0.7)

            for bbx in tru_boxes_ext:
                x,y=float(bbx[0]),float(bbx[1])
                w,h=float(bbx[2])-float(bbx[0]),float(bbx[3])-float(bbx[1])  
                ax[0].add_patch(matplotlib.patches.Rectangle((x,y),w,h,lw=1,ec='limegreen',fc='none'))
                ax[1].add_patch(matplotlib.patches.Rectangle((x,y),w,h,lw=1,ec='limegreen',fc='none'))

            for bbx,scr,pt in zip(det_boxes_ext,pred_scores,pred_pts):
                x,y=float(bbx[0]),float(bbx[1])
                w,h=float(bbx[2])-float(bbx[0]),float(bbx[3])-float(bbx[1])  
                ax[0].add_patch(matplotlib.patches.Rectangle((x,y),w,h,lw=1.25,ec='red',fc='none'))
                ax[1].add_patch(matplotlib.patches.Rectangle((x,y),w,h,lw=1.25,ec='red',fc='none'))
                ax[1].text(x+w,y+h, f"{scr.item():.2f}",color='red',fontsize=6)
                ax[1].text(x,y+h/20, f"{pt.item():.0f}",color='red',fontsize=6)

            ax[0].set(xlabel='$\eta$',ylabel='$\phi$')
            ax[1].set(xlabel='$\eta$',ylabel='$\phi$')
            plt.tight_layout()
            model_name = "jetSSD_{}_{}e".format(model.backbone_name,config["n_epochs"])
            f.savefig('{}-{}-{}.png'.format(model_name,j,event_number))
            plt.close()
            
            quit()
