import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict, Tuple
import math
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from prompt_generator import PromptGenerator
import matplotlib.pyplot as plt
import visualize

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
            prompt_type: str,
            batch_size: int,
            file,
            tag: str) -> None:
        """
        用 SAM 对一组三维的图像的每个切片进行分割，并计算 mDice 指数

        输入：
            image_set: 一个列表。每个元素的类型都是 np.ndarray，形状为 (Height, Width, 3, #slices)，
                       每个元素对应于一个 .nii.gz 数据文件，表示一个三维的图像
            label_set: 一个列表。每个元素的类型都是 np.ndarray，形状为 (Height, Width, #slices)，
                       每个元素对应于一个 .nii.gz 标签文件，表示一个三维的图像标签
                       image_set 的元素和 label_set 的元素是一一对应的
            prompt_type: 一个字符串，表示需要生成的 prompt 的类型
            batch_size: 单次输入给 SAM 的二维图像切片的个数
            file: 输出结果的文件
            tag: 当前正在用 SAM 测试的数据集的名字
        """
        print(f"Testing SAM on {tag}...")
        mdices = []

        # 处理数据集中的每一个三维图像和其对应的标签
        for i, (images, labels) in enumerate(zip(image_set, label_set)):
            slice_num = images.shape[-1]  # 当前三维图像的二维切片个数
            print(f"----Sample {i}\t slice_num: {slice_num}\t batch_size: {batch_size}")

            # 把当前三维图像的二维切片 batch 化
            batch_split = list(range(0, slice_num, batch_size)) + [slice_num]

            # 用于逐像素地记录整个三维图像的切割结果
            # segment_result_3d 的形状为 (14, Height, Width, #slices)
            # segment_result_3d[target, H, W, s] = 1 代表 SAM 把第 s 个切片的 (H, W) 坐标处
            # 认定为了属于器官 target
            segment_result_3d = np.zeros((14, *labels.shape), dtype=np.int8)

            # 用 SAM 对每个 batch 的二维图形进行推理
            for batch_num in tqdm(range(len(batch_split) - 1)):
                batch_range = range(batch_split[batch_num], batch_split[batch_num + 1])

                # 用 prepare_batched_input() 函数进行数据准备
                # batched_input: 用来喂给 SAM 的输入，包含了每张二维图片的数据、大小、为它生成的prompt等信息
                # batched_targets: 记录了每张图片中含有的器官种类，在之后计算每种器官的 mDice 的时候有用
                batched_input, batched_targets = self.prepare_batched_input(
                    images, labels, batch_range, prompt_type
                )

                # 使用 sam 进行推理，生成的 batched_output 中记录了分割产生的 mask
                # 分割得到的 mask 与 batched_input 中的 prompt（除了背景点） 是按顺序一一对应的关系
                # 注意，“一组 point prompt” 是算做一个 prompt 的，只产生一个 mask
                batched_output = self.sam(batched_input, multimask_output=False)

                # 把 SAM 的分割结果汇总到 segment_result_3d 中
                self.insert_into_3d_result(
                    segment_result_3d, batch_range[0], batched_targets, batched_output
                )

            # 对这个三维图像，对所有器官计算各自的 mDice 指标
            # 返回这个三维图像的各个出现了的器官的 mDice 指标平均值
            mdice_avg = self.calc_and_print_mdice(i, labels, segment_result_3d, file)
            if mdice_avg != float('nan'):
                mdices.append(mdice_avg)
        file.write(f"overall average mdice score: {sum(mdices) / len(mdices)}\n")
            

    def prepare_batched_input(self, 
                              images: np.ndarray, 
                              labels: np.ndarray,
                              batch_range: range,
                              prompt_type: str) -> Tuple[List[Dict], List[List[int]]]:
        """
        为三维图像的一个二维切片 batch 准备喂给 SAM 的输入
        具体的细节大量参考了 segment-anything/notebooks/predictor_example.ipynb

        输入：
            images: np.ndarray，形状为 (Height, Width, 3, #slices)，对应一个三维图像
            labels: np.ndarray，形状为 (Height, Width, #slices)，对应一个三维图像的标签   
            batch_range: 一个范围，表示本次调用需要处理的二维切片 batch 的在整个三维图像中的下标范围
            prompt_type: 表示需要为二维切片生成的 prompt 的类型

        输出：
            batched_input: 一个列表。每个元素是一个字典 input_i，代表了一张二维图片的输入数据。
                    我们用到的字段有：
                    input_i["image"]: 二维图片的数据
                    input_i["original_size"]: 图片尺寸
                    input_i["point_coords"]: point prompt 的点坐标。（可选）
                    input_i["point_labels"]: point prompt 的类型。1 表示前景点，0 表示背景点。（可选）
                    input_i["boxes"]: bounding box prompt 的坐标（用左下-右上坐标描述）。（可选）
            batched_targets: 一个列表。每个元素是一个列表，表示这个二维图像的标签中有哪些器官。
                    比如，如果一张图的标签中只有 2 和 6 两种器官，那么其对应的列表就是 [2, 6] 
                    实现保证 batched_targets 里的每个列表的元素都是升序排列的
        """
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

            # 调用 generate_prompt() 函数，根据 ground truth label 和 prompt type 生成 prompt 
            point_coords, point_labels, boxes, targets = prompt_generator.generate_prompt(
                label=label_i, 
                prompt_type=prompt_type
            )

            if point_coords is not None:
                input_i['point_coords'] = resize_transform.apply_coords_torch(
                    torch.from_numpy(point_coords).to(self.sam.device), image_i.shape[:2]
                )

            if point_labels is not None:
                input_i['point_labels'] = torch.from_numpy(point_labels).to(self.sam.device)

            if boxes is not None:
                input_i['boxes'] = resize_transform.apply_boxes_torch(
                    torch.from_numpy(boxes).to(self.sam.device), image_i.shape[:2]
                )

            batched_input.append(input_i)
            batched_targets.append(targets)

            # if i == 70:
            # # if True:
            #     # print(batched_input[127])
            #     plt.figure(figsize=(10, 10))
            #     plt.imshow(images[..., 1, i])
            #     # for box in boxes:
            #     #     show_box(box, plt.gca())
            #     print("point_coords :")
            #     print(point_coords.shape)
            #     print("point_labels:")
            #     print(point_labels.shape)
            #     print(np.argwhere(point_labels == 1))
            #     for i in range(point_coords.shape[0]):
            #         visualize.show_points(point_coords[i, :], point_labels[i, :], plt.gca())
            #     plt.axis('off')
            #     plt.savefig("plot.png")

        return batched_input, batched_targets


    def prepare_image(self, image, transform, device):
        image = transform.apply_image(image)
        image = torch.as_tensor(image, device=device.device)
        # permute: change shape from (Height, Weight, 3) to (3, Height, Weight) 
        return image.permute(2, 0, 1).contiguous()
    

    def insert_into_3d_result(self, segment_result_3d, range_left, batched_targets, batched_output):
        """
        根据对一批二维切片的 SAM 分割结果（即 batched_output 中的 masks 字段），把这些 mask
        按照器官种类分类，并且全部记录下来，形成三维的分割结果

        输出：
            batched_segmentation
        """
        assert(len(batched_targets) == len(batched_output))
        for i, (output, targets) in enumerate(zip(batched_output, batched_targets)):
            # masks.shape = （mask个数 * 1 * Height * Weight）
            # mask 个数与 targets 个数一样，二者按照同样顺序一一对应
            # 对我们的数据集而言，Height = Weight = 512
            masks = output['masks'].cpu().numpy()
            segment_result_3d[targets, :, :, i + range_left] = masks[:, 0, :, :]


    def calc_dice(self, pred, truth):
        pred = pred.astype(bool)
        truth = truth.astype(bool)

        intersection = np.logical_and(pred, truth)
        dice = 2 * np.sum(intersection) / (np.sum(pred) + np.sum(truth))

        return dice
    

    def calc_and_print_mdice(self, sample_num, labels, segment_result_3d, file):
        file.write(f"The segmentation result of sample {sample_num}:\n")
        mdice = {}
        mdice_sum, has_count = 0, 0
        for target in range(1, 14):
            target_truth = (labels == target)
            target_pred  = segment_result_3d[target, :, :, :]

            if np.sum(target_truth) == 0:
                mdice[target] = float('nan')
            else:
                mdice[target] = self.calc_dice(target_pred, target_truth)
                mdice_sum += mdice[target]
                has_count += 1

            file.write(f"organ {target}:\t mdice = {mdice[target]}\n")
        
        mdice_avg = float('nan') if has_count == 0 else mdice_sum / has_count
        file.write(f"The average mdice of different organs: {mdice_avg}\n\n")

        return mdice_avg
            