import torch
import torch.nn as nn
import torchvision
import matplotlib
import matplotlib.pyplot as plt


MIN_CELLS_ETA,MAX_CELLS_ETA = -2.5, 2.5
# EXTENT = [-4.82349586, 4.82349586, -6.21738815, 6.21801758] 
EXTENT = (-2.4999826, 2.4999774, -6.217388274177672, 6.2180176992265)

class DefaultBoxes(object):
    def __init__(self, figsize, scale, step_x, step_y):

        self.figsize = figsize
        self.step_x = step_x
        self.step_y = step_y
        self.scale = scale
        self.box_size = (self.scale[0]/self.figsize[0], self.scale[1]/self.figsize[1])


        self.default_boxes = []
        width, height = self.figsize

        i_vals, j_vals = torch.meshgrid(
            torch.arange(0, height, self.step_y),  
            torch.arange(0, width, self.step_x), 
            indexing='ij'
        )

        # Compute cx and cy in a vectorized way, taking into account the new pixel gap
        cx_vals = (j_vals + 0.5) / width   # Normalize by width for cx
        cy_vals = (i_vals + 0.5) / height  # Normalize by height for cy

        all_sizes = [(self.scale[0]/self.figsize[0], self.scale[1]/self.figsize[1])]

        for w, h in all_sizes:
            boxes = torch.stack([cx_vals, cy_vals, torch.full_like(cx_vals, w), torch.full_like(cy_vals, h)], axis=-1)
            self.default_boxes.extend(boxes.reshape(-1, 4))
        
        self.dboxes = torch.stack(self.default_boxes)
        self.dboxes.clamp_(min=0, max=1)
        self.dboxes = self.dboxes.float()

        # For IoU calculation
        self.dboxes_ltrb = self.dboxes.clone()
        self.dboxes_ltrb[:, 0] = self.dboxes[:, 0] - 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 1] = self.dboxes[:, 1] - 0.5 * self.dboxes[:, 3]
        self.dboxes_ltrb[:, 2] = self.dboxes[:, 0] + 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 3] = self.dboxes[:, 1] + 0.5 * self.dboxes[:, 3]

    @property
    def scale_xy(self):
        #Needs to be updated
        return self.box_size[0]

    @property
    def scale_wh(self):
        #Needs to be updated
        return self.box_size[1]

    @property
    def fig_size(self):
        return self.figsize

    def __call__(self, order="ltrb"):
        if order == "ltrb": return self.dboxes_ltrb
        if order == "xywh": return self.dboxes





actual_image_size  = [5,125,63]
central_image_size = [5,125,49]
figsize = (49,125)
scale = (4, 3.2*torch.pi)
step_x, step_y = 10,15
dboxes = DefaultBoxes(figsize, scale, step_x, step_y)
print('Figsize', figsize, figsize[0]*figsize[1],'Number of dboxes: ', dboxes.dboxes.shape)
print("width and height: ", (dboxes.scale[0]/dboxes.figsize[0], dboxes.scale[1]/dboxes.figsize[1]))
# print(dboxes.dboxes)
print("Each bin has height:", 4*torch.pi/125, "scale in height should be: ",0.8*4*torch.pi/125)
print("Each bin has width:", 5/49, "scale in width should be: ",0.8*5/49)


fig, ax = plt.subplots(figsize=(11,11))

for i in range(dboxes.dboxes.shape[0]):
    box_i = dboxes("ltrb")[i]
    rect = matplotlib.patches.Rectangle(
        (box_i[0], box_i[1]), 
        box_i[2] - box_i[0], 
        box_i[3] - box_i[1], 
        linewidth=1, edgecolor='r', facecolor='none'
    )
    ax.add_patch(rect)

ax.set_xlim(-0.2, 1.2)
ax.set_ylim(-0.2, 1.2)
fig.savefig("testing_default_boxes.png")

print()
print()
print()
print()





