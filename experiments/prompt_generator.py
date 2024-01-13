import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

class PromptGenerator:
    def __init__(self) -> None:
        pass

    def prepare_grid(self, h, w, dist):
        x_coords = np.arange(0, w, dist)
        y_coords = np.arange(0, h, dist)
        xx, yy = np.meshgrid(x_coords, y_coords)
        self.grid_points: np.ndarray = np.stack([xx.flatten(), yy.flatten()], axis=1).astype(np.int32)

    def generate_prompt(self,
                        label: np.ndarray,  # ground truth
                        prompt_type: str):
        point_coords = []
        point_labels = []
        boxes = []
        targets = []

        if prompt_type == "grid_point_dense":
            self.prepare_grid(h=label.shape[0], w=label.shape[1], dist=1)
        elif prompt_type == "grid_point_sparse":
            self.prepare_grid(h=label.shape[0], w=label.shape[1], dist=10)

        for i in range(1, 14):
            mask_i = (label == i)
            if mask_i.sum() == 0:
                continue
            targets.append(i)

            if prompt_type == "single_point_center":  
                # 对每个 ground truth mask，生成单点 point prompt，位于 mask 的中心
                point_coord = self.generate_point_prompt(label=mask_i, num_points=1, center=True)
                point_label = np.ones((point_coord.shape[0], ), dtype=np.int8)
                point_coords.append(point_coord)
                point_labels.append(point_label)
            elif prompt_type == "single_point_random":
                # 对每个 ground truth mask，生成单点 point prompt，位于 mask 的随机位置
                point_coord = self.generate_point_prompt(label=mask_i, num_points=1, center=False)
                point_label = np.ones((point_coord.shape[0], ), dtype=np.int8)
                point_coords.append(point_coord)
                point_labels.append(point_label)
            elif prompt_type == "multi_point_center":
                # 对每个 ground truth mask，生成多点 point prompt，其中一点位于 mask 的中心，其他随机
                point_coord = self.generate_point_prompt(label=mask_i, num_points=3, center=True)
                point_label = np.ones((point_coord.shape[0], ), dtype=np.int8)
                point_coords.append(point_coord)
                point_labels.append(point_label)
            elif prompt_type == "multi_point_random":
                # 对每个 ground truth mask，生成多点 point prompt，都位于 mask 的随机位置
                point_coord = self.generate_point_prompt(label=mask_i, num_points=3, center=False)
                point_label = np.ones((point_coord.shape[0], ), dtype=np.int8)
                point_coords.append(point_coord)
                point_labels.append(point_label)
            elif prompt_type == "grid_point_dense":
                # 把每个像素点都作为 prompt （但这个要好多好多显存，估计跑不起来）
                point_coord, point_label = self.generate_grid_point_prompt(label=mask_i)
                point_coords.append(point_coord)
                point_labels.append(point_label)
            elif prompt_type == "grid_point_sparse":
                # 稍微有一些间距的 grid point prompt 
                # （间距目前默认设置为了 10，在当前函数 for 循环之前，调用 self.prepare_grid() 时设置）
                point_coord, point_label = self.generate_grid_point_prompt(label=mask_i)
                point_coords.append(point_coord)
                point_labels.append(point_label)
            elif prompt_type == "bounding_box_tight":
                # 对每个 ground truth mask，生成一个紧贴 mask 边界框的 bounding box prompt
                box = self.generate_box_prompt(label=mask_i, box_margin=1)
                boxes.append(box)
            elif prompt_type == "bounding_box_loose":
                # 对每个 ground truth mask，生成一个较 mask 边界框略大的 bounding box prompt
                box = self.generate_box_prompt(label=mask_i, box_margin=10)
                boxes.append(box)

        point_coords = np.stack(point_coords, axis=0) if len(point_coords) > 0 else None
        point_labels = np.stack(point_labels, axis=0) if len(point_labels) > 0 else None
        boxes = np.stack(boxes, axis=0) if len(boxes) > 0 else None

        return point_coords, point_labels, boxes, targets


    def generate_random_point_prompt(self,
                                     label: np.ndarray,
                                     num_points: int = 1) -> np.ndarray:
        indices = np.argwhere(label == 1)[:, ::-1]
        num_indices = indices.shape[0]

        if num_indices < num_points:
            random_indices = np.random.choice(num_indices, num_points, replace=True)
        else:
            random_indices = np.random.choice(num_indices, num_points, replace=False)

        selected_points = indices[random_indices].astype(np.int32)

        return selected_points
    

    def generate_center_point_prompt(self,
                                     label: np.ndarray,
                                     num_points: int = 1) -> np.ndarray:
        w = label.shape[1]
        dist = distance_transform_edt(label)
        maxpos = dist.argmax()
        
        centor_coord = np.array([[maxpos % w, maxpos // w]], dtype=np.int32)
        return np.tile(centor_coord, (num_points, 1))
        

    def generate_point_prompt(self,
                              label: np.ndarray,
                              num_points: int,
                              center: bool = True) -> np.ndarray:
        if center:
            center_points = self.generate_center_point_prompt(label=label, num_points=1)
            if num_points == 1:
                selected_points = center_points
            elif num_points > 1:
                random_points = self.generate_random_point_prompt(label=label, num_points=num_points-1)
                selected_points = np.concatenate([center_points, random_points], axis=0)
        else:
            random_points = self.generate_random_point_prompt(label=label, num_points=num_points)
            selected_points = random_points
        
        return selected_points

    def generate_grid_point_prompt(self, label: np.ndarray):
        coords = self.grid_points
        labels = np.zeros((coords.shape[0], ), dtype=np.int8)

        for i in range(coords.shape[0]):
            labels[i] = label[coords[i, 1], coords[i, 0]]

        return coords, labels

    def generate_box_prompt(self, 
                            label: np.ndarray, 
                            box_margin: int = 1) -> np.ndarray:
        rows, cols = np.where(label)
        assert(len(rows) > 0)
            
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()
            
        min_row = max(0, min_row - box_margin)
        max_row = min(label.shape[0], max_row + box_margin)
        min_col = max(0, min_col - box_margin)
        max_col = min(label.shape[1], max_col + box_margin)            
        
        return np.array([min_col, min_row, max_col, max_row], dtype=np.int32)