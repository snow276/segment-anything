import os

"""
命令行参数：
--prompt_type: 提示类型。目前支持的有：
    single_point_center:  对每个 ground truth mask 用一个点的 prompt ，在 mask 的中心（内距离变化最大值处）
    single_point_random:  对每个 ground truth mask 用一个点的 prompt ，在 mask 的内部随机位置
    multi_point_center:   对每个 ground truth mask 用多个点的 prompt ，其中一个是中心点
        （如果需要修改点的个数，需要在 prompt_generator.py 模块里面改一下。默认总点数是 3 个）
    multi_point_random:   对每个 ground truth mask 用多个点的 prompt ，全都是随机点
        （如果需要修改点的个数，需要在 prompt_generator.py 模块里面改一下。默认总点数是 3 个）
    bounding_box_tight:   对每个 ground truth mask 用一个 box prompt ，边界尽量紧
    bounding_box_loose:   对每个 ground truth mask 用一个 box prompt ，边界略大于 ground truth 边界框
    grid_point_sparse:    使用 grid point prompt，格点的间距默认为 10 像素
        （如果需要修改格点间距，需要在 prompt_generator.py 模块里面改一下）

--batch_size: 推理的时候的 batch size。实测下来模型大小大概有 20G ，每个 batch 大概占用 1G ，
              所以在 24G 的显卡上推荐设置 batch size 为 2

--output_file: 结果输出的文件

--device: 加载 SAM 模型的设备

其它参数大概率不需要改，具体可以参考 argparser.py 文件
"""

os.system('python3 experiments/task1.py --prompt_type single_point_center --batch_size 2 --device cuda:0 --output_file experiments/results/single_point_center.txt')
os.system('python3 experiments/task1.py --prompt_type single_point_random --batch_size 2 --device cuda:0 --output_file experiments/results/single_point_random.txt')
os.system('python3 experiments/task1.py --prompt_type multi_point_center --batch_size 2 --device cuda:0 --output_file experiments/results/multi_point_center.txt')
os.system('python3 experiments/task1.py --prompt_type multi_point_random --batch_size 2 --device cuda:0 --output_file experiments/results/multi_point_random.txt')
os.system('python3 experiments/task1.py --prompt_type bounding_box_tight --batch_size 2 --device cuda:0 --output_file experiments/results/bounding_box_tight.txt')
os.system('python3 experiments/task1.py --prompt_type bounding_box_loose --batch_size 2 --device cuda:0 --output_file experiments/results/bounding_box_loose.txt')
os.system('python3 experiments/task1.py --prompt_type grid_point_sparse --batch_size 2 --device cuda:0 --output_file experiments/results/grid_point_sparse.txt')