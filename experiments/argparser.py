import argparse

def create_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Runs SAM on BTCV with different kinds of prompts."
        )
    )

    parser.add_argument(
        "--json_path",
        type=str,
        default="./dataset/RawData/dataset_0.json",
        help="The path to the json-formatted dataset-loading config file."
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="./dataset/RawData/",
        help="The path to the data directory."
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="./experiments/results/out.txt",
        help="The path to the output file."
    )

    parser.add_argument(
        "--prompt_type",
        type=str,
        default=None,
        choices=['single_point_center', 'single_point_random', 
                 'multi_point_center', 'multi_point_random', 
                 'bounding_box_tight', 'bounding_box_loose'],
        help="The type of point prompt, in [ \
                'single_point_center', 'single_point_random', \
                'multi_point_center', 'multi_point_random', \
                'bounding_box_tight', 'bounding_box_loose']."
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="The device to which data and model is loaded."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The number of 2D pics used in a single SAM segmentation."
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="vit_h",
        help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b'].",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/sam_vit_h_4b8939.pth",
        help="The path to the SAM checkpoint to use for mask generation.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="random seed."
    )

    return parser