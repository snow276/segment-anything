import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict
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
            tag: str) -> None:
        print(f"Testing SAM on {tag}...")

        for i, (images, labels) in enumerate(zip(image_set, label_set)):
            # print(f"Sample {i} depth: {len(images)}")
            batched_input = self.prepare_batched_input(
                images, labels, point_prompt, multi_point, background_point,
                box_prompt, box_margin
            )
            batched_output = self.sam(batched_input, multimask_output=False)

            if i == 0:
                plt.figure(figsize=(10, 10))
                plt.imshow(images[..., 127])
                for mask in batched_output[127]['masks']:
                    show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
                plt.axis('off')
                plt.show()
            

    def prepare_batched_input(self, 
                              images: np.ndarray, 
                              labels: np.ndarray,
                              point_prompt: str = None,
                              multi_point: bool = False,
                              background_point: bool = False,
                              box_prompt: bool = False,
                              box_margin: int = 1) -> List[Dict]:
        batched_input = [] 

        resize_transform = ResizeLongestSide(self.sam.image_encoder.img_size)
        prompt_generator = PromptGenerator()

        slice_num = images.shape[-1]
        for i in range(slice_num):
            image_i: np.ndarray = images[..., i]
            label_i: np.ndarray = labels[..., i]
            input_i = {}
            input_i["image"] = self.prepare_image(image_i, resize_transform, self.sam)
            input_i["original_size"] = image_i.shape[:2]

            # print(f"box_prompt: {box_prompt}")
            point_coords, point_labels, boxes = prompt_generator.generate_prompt(
                label=label_i, 
                point_prompt=point_prompt,
                box_prompt=box_prompt,
                box_margin=box_margin
            )

            # input_i['point_coords'] = point_coords
            # input_i['point_labels'] = point_labels
            if boxes is not None:
                input_i['boxes'] = resize_transform.apply_boxes_torch(torch.tensor(boxes, device=self.sam.device), image_i.shape[:2])
            # input['boxes'] = resize_transform.apply_boxes_torch(boxes, image.shape[:2])
            batched_input.append(input_i)

        if True:
            plt.figure(figsize=(10, 10))
            plt.imshow(images[..., 1, 127])
            for box in batched_input[127]['boxes']:
                show_box(box.cpu().numpy(), plt.gca())
            plt.axis('off')
            plt.savefig("plot.png")
        return batched_input


    def prepare_image(self, image, transform, device):
        image = transform.apply_image(image)
        image = torch.as_tensor(image, device=device.device)
        # permute: change shape from (Height, Weight, 3) to (3, Height, Weight) 
        return image.permute(2, 0, 1).contiguous
    
