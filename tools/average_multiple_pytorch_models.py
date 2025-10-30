import os
import torch
import argparse
from collections import defaultdict

def load_model_from_dir(directory):
    """从目录加载 pytorch_model.bin"""
    model_path = os.path.join(directory, "pytorch_model.bin")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"pytorch_model.bin not found in {directory}")
    print(f"Loading model from {model_path}")
    return torch.load(model_path, map_location="cpu")

def average_models(model_dirs, output_path, weights=None):
    """
    对多个目录中的 pytorch_model.bin 进行平均。

    Args:
        model_dirs (list of str): 包含 pytorch_model.bin 的目录列表
        output_path (str): 输出的平均模型路径
        weights (list of float, optional): 每个模型的权重。若为 None，则等权重平均。
    """
    num_models = len(model_dirs)
    if weights is None:
        weights = [1.0 / num_models] * num_models
    else:
        assert len(weights) == num_models, "Number of weights must match number of models"
        total = sum(weights)
        weights = [w / total for w in weights]  # 归一化

    # 加载所有模型
    models = [load_model_from_dir(d) for d in model_dirs]

    # 获取所有键（参数名）
    all_keys = set(models[0].keys())
    for model in models[1:]:
        all_keys &= set(model.keys())  # 取交集，只平均共有的参数

    print(f"Found {len(all_keys)} common parameters across all models.")

    # 初始化平均字典
    avg_state_dict = {}

    for key in all_keys:
        tensors = [model[key] for model in models]
        shapes = [t.shape for t in tensors]
        if len(set(shapes)) != 1:
            print(f"Warning: Shape mismatch for '{key}', skipping. Shapes: {shapes}")
            continue

        # 加权平均
        weighted_sum = sum(w * t for w, t in zip(weights, tensors))
        avg_state_dict[key] = weighted_sum

    # 保存结果
    torch.save(avg_state_dict, output_path)
    print(f"Averaged model saved to {output_path}")

def parse_weights(weight_str):
    """解析命令行权重字符串，如 '0.3,0.7' 或 '1,2,1'"""
    try:
        weights = list(map(float, weight_str.split(',')))
        return weights
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid weight format: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Average pytorch_model.bin from multiple directories")
    parser.add_argument(
        "--dirs",
        nargs="+",
        required=True,
        help="List of directories containing pytorch_model.bin"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for averaged model (e.g., averaged_pytorch_model.bin)"
    )
    parser.add_argument(
        "--weights",
        type=parse_weights,
        default=None,
        help="Optional comma-separated weights for each model (e.g., '1,2,1'). If not given, equal weights are used."
    )

    args = parser.parse_args()

    if args.weights and len(args.weights) != len(args.dirs):
        raise ValueError(f"Number of weights ({len(args.weights)}) must match number of directories ({len(args.dirs)})")

    average_models(args.dirs, args.output, weights=args.weights)
