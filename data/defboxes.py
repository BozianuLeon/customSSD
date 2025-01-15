import torch



class MyDefaultBoxes(object):
    def __init__(self, figsize, step_x, step_y):

        self.figsize = figsize
        self.step_x = step_x
        self.step_y = step_y

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

        all_sizes = [
                     (4.0/self.figsize[0], 4.0/self.figsize[1]), 
                    #  (6/self.figsize[0], 6/self.figsize[1])
                     ]

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
        return self.box_size[0]

    @property
    def fig_size(self):
        return self.figsize

    def __call__(self, order="ltrb"):
        if order == "ltrb": return self.dboxes_ltrb
        if order == "xywh": return self.dboxes





if __name__=="__main__":

    # pytorch tensor [5,125,49]
    # figsize = (49,125)
    figsize = (24,63) # (96,125)
    step_x, step_y = 1,1
    dboxes = MyDefaultBoxes(figsize, step_x, step_y)
    print('Number of dboxes: ', dboxes.dboxes.shape)





