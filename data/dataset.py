import torch
import torchvision
import json




class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, annotation_file):
        # Custom dataset that takes in the annotations folder 
        # returns 
        # img: pytorch tensor [5,125,49]
        # validation dict: contains lists/tensors of dict_keys(['boxes', 'labels', 'jet_pt', 'extent', 'h5file', 'h5event', 'event_no'])
        # here the channel order is: 
        # [H_sum_pt,H_max_pt,H_sum_signif,H_max_signif,H_max_noise]
        with open(annotation_file, 'r') as f:
            self.data = json.load(f)
        
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

        event_no = anns_i["image"]["id"]
        h5file   = anns_i["image"]["file"]
        h5event  = anns_i["image"]["event"]
        pT       = anns_i["anns"]["jet_pt"]
        extent   = anns_i["anns"]["extent"]
        extent_tensor = torch.tensor(extent).float()

        return img, {'boxes': bboxes, 'labels': labels, 'jet_pt': pT, 'extent': extent_tensor, 'h5file': h5file, 'h5event': h5event, 'event_no': event_no}

    def collate_fn(self,batch):
        images, targets = zip(*batch) 
        images = torch.stack(images, dim=0)
        return images, targets



if __name__=="__main__":

    dataset = CustomDataset(
        annotation_file="../../../data/mu200/anns_central_jets_20GeV.json"
    )

    img, val_dict = dataset[3]
    print(val_dict.keys())

    train_len = int(0.78 * len(dataset))
    val_len = int(0.02 * len(dataset))
    test_len = len(dataset) - train_len - val_len
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])
    print('\ttrain / val / test size : ',train_len,'/',val_len,'/',test_len,'\n')
