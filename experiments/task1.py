import argparse
import numpy
import torch
from tqdm import tqdm
from argparser import create_arg_parser
from segment_anything import sam_model_registry
from data_loader import BTCV_loader
from data_processor import SAMed

def main(args: argparse.Namespace) -> None:
    # 初始化随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 加载数据集
    btcv_loader = BTCV_loader(args.json_path, args.data_path)
    # train_image, train_label = btcv_loader.load_data_with_label("training")
    valid_image, valid_label = btcv_loader.load_data_with_label("validation")
    
    # 加载模型
    samed = SAMed()
    samed.set_sam_model(args.model_type, args.checkpoint, args.device)

    # 把运行结果输出到命令行指定的输出文件中
    file = open(args.output_file, "w")
    # 在训练集上进行测试
    # samed.test(train_image, train_label, args.prompt_type, args.batch_size, file, tag="training data")

    # 在测试集上进行测试
    samed.test(valid_image, valid_label, args.prompt_type, args.batch_size, file, tag="training data")

    file.close()

if __name__ == "__main__":
    # 解析参数
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()
    
    # main 函数
    main(args)
