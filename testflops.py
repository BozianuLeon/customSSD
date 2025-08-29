import torch
from torch.profiler import profile, record_function, ProfilerActivity
import torchprofile

import models
import data

config = {
    "seed"       : 0,
    "device"     : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "NW"         : 0,
    "BS"         : 1,
    "n_epochs"   : 32,
    "max_num"    : 150,
}

torch.manual_seed(config["seed"])

# fake some data
fake_inputs = torch.randn(1, 5, 125, 49).to(config["device"])

# load trained model
model = models.SSD(backbone_name="custom_convnext_central",in_channels=5)
model = model.to(config["device"]) 
model_name = "jetSSD_{}_{}e".format(model.backbone_name,config["n_epochs"])
model_save_path = "/home/users/b/bozianu/work/paperSSD/customSSD/saved_models/" + f"/{model_name}.pth"
# model.load_state_dict(torch.load(model_save_path, map_location=torch.device(config["device"])))
model.load_state_dict(torch.load(model_save_path, weights_only=True,map_location=torch.device(config["device"])))
total_params = sum(p.numel() for p in model.parameters())
print(model.backbone_name,f'!total \t{total_params:,} parameters.\n')
model.eval()

# default prior boxes
dboxes = data.DefaultBoxes(figsize=(24,63),scale=(3.84,4.05),step_x=1,step_y=1) 
print("Generated prior boxes, ",dboxes.dboxes.shape, ", default boxes")
# encoder 
encoder = data.Encoder(dboxes)


# run fake inference
with torch.inference_mode():
    img_tensor = fake_inputs.to(config["device"]).float()
    locs, conf, ptmap = model(img_tensor)

    # define NMS scriteria, confidence threshold
    output = encoder.decode_batch(locs, conf, ptmap, 
                                    iou_thresh=0.25, #NMS
                                    confidence=0.45, #conf threshold
                                    max_num=config["max_num"]) #155

    boxes, labels, scores, pts = zip(*output)



# Profile the model
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],profile_memory=True,record_shapes=True,with_flops=True,with_modules=True) as prof:
    with record_function("model_inference"):
        model(fake_inputs)

print(prof.key_averages().table(row_limit=10))


print(prof.key_averages().table(sort_by="self_flops", row_limit=10))

macs = torchprofile.profile_macs(model,fake_inputs)
print(macs)