import torch
from torchvision.ops import box_iou

def match_boxes(boxes1, boxes2, iou_threshold=0.5):
    iou_matrix = box_iou(boxes1, boxes2)
    max_iou_values, max_iou_indices = torch.max(iou_matrix, dim=1)
    valid_matches = max_iou_values >= iou_threshold
    boxes1_indices = torch.nonzero(valid_matches).squeeze(1)
    boxes2_indices = max_iou_indices[valid_matches]
    return boxes1_indices, boxes2_indices


def match_boxes(boxes1, boxes2, iou_threshold=0.5):
    iou_matrix = box_iou(boxes1, boxes2)
    
    # First mask out all IoUs below threshold
    iou_matrix[iou_matrix < iou_threshold] = 0.0
    
    # For each box2, get the index of the best matching box1
    max_iou_values, max_idx_boxes1 = torch.max(iou_matrix, dim=0)
    
    # Create an array of indices for boxes2
    idx_boxes2 = torch.arange(len(boxes2), device=boxes1.device)
    
    # Keep only the valid matches (above threshold)
    valid_matches = max_iou_values > 0
    max_idx_boxes1 = max_idx_boxes1[valid_matches]
    idx_boxes2 = idx_boxes2[valid_matches]
    
    return max_idx_boxes1, idx_boxes2



def match_boxes_periodic(boxes1, boxes2, iou_threshold=0.5, period=2*torch.pi):
    # Create three versions of boxes2: original, shifted up, and shifted down
    boxes2_original = boxes2.clone()
    boxes2_up = boxes2.clone()
    boxes2_down = boxes2.clone()
    
    # Shift y coordinates based on sign
    y_mask_positive = boxes2[:, 1] > 0  # y1 coordinate is positive
    y_mask_negative = boxes2[:, 1] < 0  # y1 coordinate is negative
    
    # For positive y, subtract period
    boxes2_up[y_mask_positive, 1] -= period  # y1
    boxes2_up[y_mask_positive, 3] -= period  # y2
    
    # For negative y, add period
    boxes2_down[y_mask_negative, 1] += period  # y1
    boxes2_down[y_mask_negative, 3] += period  # y2
    
    # Stack all versions of boxes2
    boxes2_all = torch.cat([boxes2_original, boxes2_up, boxes2_down], dim=0)
    
    # Compute IoU matrix for all versions
    iou_matrix = box_iou(boxes1, boxes2_all)
    
    # Mask out IoUs below threshold
    iou_matrix[iou_matrix < iou_threshold] = 0.0
    
    # Find best matches considering all versions
    max_iou_values, max_indices = torch.max(iou_matrix, dim=0)
    
    # Get the original box2 indices (before periodic copies)
    n_boxes2 = len(boxes2)
    original_box2_idx = torch.arange(n_boxes2, device=boxes1.device)
    
    # Keep only the valid matches (above threshold)
    valid_matches = max_iou_values[:n_boxes2] > 0  # only consider original boxes
    idx_boxes1 = max_indices[:n_boxes2][valid_matches]  # only consider original boxes
    idx_boxes2 = original_box2_idx[valid_matches]
    
    return idx_boxes1, idx_boxes2




# Create test boxes
boxes1 = torch.tensor([
    [0, 0, 10, 10],    # should match with first box in boxes2
    [20, 20, 30, 30],  # should match with second box in boxes2
    [40, 40, 50, 50],  # should not match with any box
    [0, 0, 9, 9],      # should also match with first box in boxes2
    [100, 100, 110, 110],  # should not match
    [19, 19, 31, 31],  # should match with second box in boxes2
], dtype=torch.float)

boxes2 = torch.tensor([
    [1, 1, 11, 11],     # matches with boxes1[0] and boxes1[3]
    [21, 21, 31, 31],   # matches with boxes1[1] and boxes1[5]
    [60, 60, 70, 70],   # matches with nothing
    [80, 80, 90, 90],   # matches with nothing
    [200, 200, 210, 210] # matches with nothing
], dtype=torch.float)

# Match boxes
idx1, idx2 = match_boxes(boxes1, boxes2, iou_threshold=0.5)
# idx1, idx2 = match_boxes_periodic(boxes1, boxes2, iou_threshold=0.5)

# Print results
print("Matched indices from boxes1:", idx1)
print("Matched indices from boxes2:", idx2)
print("\nMatched boxes from boxes1:")
print(boxes1[idx1])
print("\nMatched boxes from boxes2:")
print(boxes2[idx2])