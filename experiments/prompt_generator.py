import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

class PromptGenerator:
    def __init__(self) -> None:
        pass

    def generate_prompt(self,
                        label: np.ndarray,  # ground truth
                        point_prompt: str,
                        box_prompt: bool,
                        box_margin: int):
        # point prompt not implemented yet
        point_coords = None
        point_labels = None

        if box_prompt:
            boxes = self.generate_box_prompt(label, box_margin)
        else:
            boxes = None

        return point_coords, point_labels, boxes


    def generate_point_prompt(self,
                              label: np.ndarray,
                              point_prompt: str):
        "Not Implemented"
        pass

    def generate_box_prompt(self, 
                            label: np.ndarray, 
                            box_margin: int = 1):
        boxes = []

        for i in range(1, 14):
            mask_i = (label == i)

            rows, cols = np.where(mask_i)
            if len(rows) == 0:
                continue
            
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()
            
            min_row = max(0, min_row - box_margin)
            max_row = min(label.shape[0], max_row + box_margin)
            min_col = max(0, min_col - box_margin)
            max_col = min(label.shape[1], max_col + box_margin)            

            boxes.append([min_row, min_col, max_row, max_col])

        if len(boxes) == 0:
            return None
        
        return boxes