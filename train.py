import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR


import models
import data


config = {
    "seed"       : 0,
    "device"     : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "NW"         : 2,
    "BS"         : 4,
    "LR"         : 0.01,
    "WD"         : 0.01,
    "wup_epochs" : int(12/3),
    "n_epochs"   : 12,
}
torch.manual_seed(config["seed"])


dataset = data.CustomDataset(annotation_file="/home/users/b/bozianu/work/data/mu200/anns_central_jets_20GeV.json")
train_len = int(0.78 * len(dataset))
val_len = int(0.02 * len(dataset))
test_len = len(dataset) - train_len - val_len
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])
print('\ttrain / val / test size : ',train_len,'/',val_len,'/',test_len,'\n')

train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=dataset.collate_fn, batch_size=config["BS"], shuffle=True, drop_last=True, num_workers=config["NW"])
val_loader = torch.utils.data.DataLoader(val_dataset, collate_fn=dataset.collate_fn, batch_size=config["BS"], shuffle=False, drop_last=True, num_workers=config["NW"])
test_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=dataset.collate_fn, batch_size=config["BS"], shuffle=False, drop_last=True, num_workers=config["NW"])


# instantiate model
model = models.SSD(backbone_name="uconvnext_central",in_channels=5)
model = model.to(config["device"]) 
total_params = sum(p.numel() for p in model.parameters())
print(model.backbone_name, f'\t{total_params:,} total! parameters.\n')
        
# optimizers & learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=config["LR"], weight_decay=config["WD"], amsgrad=True)  
warmup_scheduler = LinearLR(optimizer, start_factor=1./3, end_factor=1.0, total_iters=config["wup_epochs"]) # Linear warmup
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=config["n_epochs"] - config["wup_epochs"]) # Cosine Annealing after warmup
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[config["wup_epochs"]])

# default prior boxes
dboxes = data.DefaultBoxes(figsize=(24,63),step_x=1,step_y=1) 
print("Generated prior boxes, ",dboxes.dboxes.shape, ", default boxes")

# encoder and loss
encoder = data.Encoder(dboxes)
loss = models.Loss(dboxes,scalar=1.0)


print('Starting training...')
for epoch in range(config["n_epochs"]):

    model.train()
    for step, (images, target_dict) in enumerate(train_loader):
        # send data to gpu (annoying)
        images = images.to(config["device"],non_blocking=True)





