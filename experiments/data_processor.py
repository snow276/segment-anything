import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict, Tuple
import math
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from prompt_generator import PromptGenerator
import matplotlib.pyplot as plt

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

class SAMed:
    def __init__(self) -> None:
        pass


    def set_sam_model(self, 
                      model_type: str, 
                      checkpoint: str, 
                      device: str) -> None:
        print("Loading SAM model...")
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.sam.to(device)


    def test(self, 
            image_set: list, 
            label_set: list, 
            point_prompt: str, 
            multi_point: bool, 
            background_point: bool,
            box_prompt: bool,
            box_margin: int,
            batch_size: int,
            file,
            tag: str) -> None:
        print(f"Testing SAM on {tag}...")
        mdices = []

        for i, (images, labels) in enumerate(zip(image_set, label_set)):
            slice_num = images.shape[-1]
            print(f"----Sample {i}\t slice_num: {slice_num}\t batch_size: {batch_size}")

            batch_split = list(range(0, slice_num, batch_size)) + [slice_num]

            for batch_num in tqdm(range(len(batch_split) - 1)):
                batch_range = range(batch_split[batch_num], batch_split[batch_num + 1])

                batched_input, batched_targets = self.prepare_batched_input(
                    images, labels, batch_range, point_prompt, multi_point, background_point,
                    box_prompt, box_margin
                )

                batched_output = self.sam(batched_input, multimask_output=False)

                batched_mdice = self.calc_batched_dice(batched_output, batched_targets, labels[:, :, batch_range])

                mdices += batched_mdice
            
            self.print_dice(mdices, file)
            

    def prepare_batched_input(self, 
                              images: np.ndarray, 
                              labels: np.ndarray,
                              batch_range: range,
                              point_prompt: str = None,
                              multi_point: bool = False,
                              background_point: bool = False,
                              box_prompt: bool = False,
                              box_margin: int = 1) -> Tuple[List[Dict], List[List[int]]]:
        batched_input = [] 
        batched_targets = []

        resize_transform = ResizeLongestSide(self.sam.image_encoder.img_size)
        prompt_generator = PromptGenerator()

        for i in batch_range:
            # print(i)
            image_i: np.ndarray = images[..., i]
            label_i: np.ndarray = labels[..., i]
            input_i = {}
            input_i["image"] = self.prepare_image(image_i, resize_transform, self.sam)
            input_i["original_size"] = image_i.shape[:2]

            # print(f"box_prompt: {box_prompt}")
            point_coords, point_labels, boxes, targets = prompt_generator.generate_prompt(
                label=label_i, 
                point_prompt=point_prompt,
                box_prompt=box_prompt,
                box_margin=box_margin
            )

            # input_i['point_coords'] = point_coords
            # input_i['point_labels'] = point_labels
            if boxes is not None:
                # print(i)
                input_i['boxes'] = resize_transform.apply_boxes_torch(torch.from_numpy(boxes).to(self.sam.device), image_i.shape[:2])
            # input['boxes'] = resize_transform.apply_boxes_torch(boxes, image.shape[:2])
            batched_input.append(input_i)
            batched_targets.append(targets)

            # if i == 127:
            # # if True:
            #     print(batched_input[127])
            #     plt.figure(figsize=(10, 10))
            #     plt.imshow(images[..., 1, 127])
            #     for box in boxes:
            #         show_box(box, plt.gca())
            #     plt.axis('off')
            #     plt.savefig("plot.png")

        return batched_input, batched_targets


    def prepare_image(self, image, transform, device):
        image = transform.apply_image(image)
        image = torch.as_tensor(image, device=device.device)
        # permute: change shape from (Height, Weight, 3) to (3, Height, Weight) 
        return image.permute(2, 0, 1).contiguous()
    

    def calc_batched_dice(self, batched_output, batched_targets, labels):
        batched_dice = [[float('nan') for __ in range(14)] for _ in range(len(batched_output))]
        for i, (output, targets) in enumerate(zip(batched_output, batched_targets)):
            if len(targets) == 0:
                continue
            masks = output['masks'].cpu().numpy()
            # masks.shape = （mask个数 * 1 * Height * Weight）
            # 对 box prompt 而言，mask个数与targets个数一样
            # 对我们的数据集而言，Height = Weight = 512
            assert(len(masks) == len(targets))
            for j in range(len(targets)):
                mask = masks[j, 0, :, :]
                target = targets[j]
                truth = (labels[:, :, i] == target)

                dice = self.calc_dice(mask, truth)

                batched_dice[i][target] = dice

        # print(batched_dice)
        return batched_dice


    def calc_dice(self, pred, truth):
        pred = pred.astype(bool)
        truth = truth.astype(bool)

        intersection = np.logical_and(pred, truth)
        dice = 2 * np.sum(intersection) / (np.sum(pred) + np.sum(truth))

        return dice
    

    def print_dice(self, mdices, file):
        total_dice, total_cnt = 0, 0

        for target in range(1, 14):
            total = 0
            cnt = 0
            for j in range(len(mdices)):
                if not math.isnan(mdices[j][target]):
                    total += mdices[j][target]
                    cnt += 1
            if cnt != 0:
                dice = total / cnt
                file.write(f"target{target}\t mdice = {dice}\n")
                total_dice += dice
                total_cnt += 1

        avg_dice = total_dice / total_cnt
        file.write(f"For the sample, avg_dice = {avg_dice}\n\n")