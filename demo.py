import torch
import torchvision
import time
from tqdm.auto import tqdm



from ssd import SSD, MultiBoxLoss
from data.dataset import NewCOCODataset, NewCOCODataLoader
from utils.utils import move_dev


# Get data
dataset = NewCOCODataset(root_folder="/home/users/b/bozianu/work/data/val2017",
                            annotation_json="/home/users/b/bozianu/work/data/annotations/instances_val2017.json")
print('Images in dataset:',len(dataset))

train_size = int(0.8 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
print('\ttrain / val / test size : ',train_size,'/',val_size,'/',test_size)

torch.random.manual_seed(1)
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset,[train_size,val_size,test_size])


#config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NW = 2
EPOCH = 20
LR = 1e-3
BS = 8
momentum = 0.9  
weight_decay = 5e-4  
print_feq = 100


#dataloaders
train_init_dataloader = NewCOCODataLoader(train_dataset,batch_size=BS,num_workers=NW,shuffle=True,drop_last=True)
val_init_dataloader = NewCOCODataLoader(val_dataset,batch_size=BS,num_workers=NW,shuffle=True,drop_last=True)
train_dataloader = train_init_dataloader.loader()
val_dataloader = val_init_dataloader.loader()


#initialise model
model = SSD(device=device)
criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy)
optimizer = torch.optim.SGD(model.parameters(),lr=LR, momentum=momentum, weight_decay=weight_decay)


#training loop
for epoch in range(EPOCH):
    tock = time.perf_counter()
    model.train()
    train_loss_this_epoch = []
    valid_loss_this_epoch = []
    for i, data in enumerate(train_dataloader):
        
        img,boxes,labels = data 
        
        #send to GPU
        img = img.to(device)
        truth_boxes = move_dev(boxes,device)
        truth_labels = move_dev(labels,device)
        
        #pass through network
        pred_loc, pred_sco = model(img)
        
        #compute loss
        loss = criterion(pred_loc,pred_sco,truth_boxes,truth_labels)

        optimizer.zero_grad(set_to_none=True) # Reduce memory operations
        loss.backward()  # init backprop
        optimizer.step() # adjust weights

        train_loss_this_epoch.append(loss.item())
        # if i%print_feq == 0:
        #     print('epoch:', epoch, '\tstep', i, '/', len(train_dataloader),
        #           '\ttrain loss:', '{:.4f}'.format(loss.item()),
        #           '\telapsed time','{:.4f}'.format((time.perf_counter()-tock)),'s')
            
        
    
    with torch.no_grad():
        model.eval()
        for j, val_data in enumerate(val_dataloader):
            val_img, val_boxes, val_labels = val_data
            
            #send to GPU
            val_img.to(device)
            val_boxes = move_dev(val_boxes,device)
            val_labels = move_dev(val_labels,device)

            #pass through network
            pred_loc, pred_sco = model(img)
            
            val_loss = criterion(pred_loc,pred_sco,val_boxes,val_labels)
            valid_loss_this_epoch.append(val_loss.item())
    
    print('EPOCH:',epoch, '/', EPOCH, 
          '\ttrain loss: {:.4f}'.format(sum(train_loss_this_epoch)/len(train_loss_this_epoch)),
          '\tval loss: {:.4f}'.format(sum(valid_loss_this_epoch)/len(valid_loss_this_epoch)),
          '\telapsed time: {:.3f}'.format((time.perf_counter()-tock)),'s')

















