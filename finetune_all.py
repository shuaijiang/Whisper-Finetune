import argparse
import functools
import os
import platform
import random
import numpy as np
import torch
#from peft import LoraConfig, get_peft_model, AdaLoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor

from utils.callback import SavePeftModelCallback
from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding
from utils.model_utils import load_from_checkpoint
from utils.reader import CustomDataset
from utils.utils import print_arguments, make_inputs_require_grad, add_arguments
torch._dynamo.config.suppress_errors = True

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("train_data",    type=str, default="dataset/train.json",       help="训练数据集的路径")
add_arg("test_data",     type=str, default="dataset/test.json",        help="测试数据集的路径")
add_arg("base_model",    type=str, default="openai/whisper-tiny",      help="Whisper的基础模型")
add_arg("output_dir",    type=str, default="output/",                  help="训练保存模型的路径")
add_arg("freeze_encoder",  type=bool, default=True,   help="是否freeze encoder")
add_arg("warmup_steps",  type=int, default=1000,      help="训练预热步数")
add_arg("logging_steps", type=int, default=100,     help="打印日志步数")
add_arg("eval_steps",    type=int, default=1000,    help="多少步数评估一次")
add_arg("save_steps",    type=int, default=1000,    help="多少步数保存模型一次")
add_arg("num_workers",   type=int, default=4,       help="读取数据的线程数量")
add_arg("learning_rate", type=float, default=1e-5,  help="学习率大小")
add_arg("min_audio_len", type=float, default=0.5,   help="最小的音频长度，单位秒")
add_arg("max_audio_len", type=float, default=30,    help="最大的音频长度，单位秒")
add_arg("fp16",          type=bool,  default=True,  help="是否使用fp16训练模型")
add_arg("use_8bit",      type=bool,  default=False, help="是否将模型量化为8位")
add_arg("timestamps",    type=bool,  default=False, help="训练时是否使用时间戳数据")
add_arg("local_files_only", type=bool, default=True, help="是否只在本地加载模型，不尝试下载")
add_arg("num_train_epochs", type=int, default=3,      help="训练的轮数")
add_arg("language",      type=str, default="Chinese", help="设置语言，可全称也可简写，如果为None则训练的是多语言")
add_arg("task",     type=str, default="transcribe", choices=['transcribe', 'translate'], help="模型的任务")
add_arg("augment_config_path",         type=str, default=None, help="数据增强配置文件路径")
add_arg("resume_from_checkpoint",      type=str, default=None, help="恢复训练的检查点路径")
add_arg("per_device_train_batch_size", type=int, default=8,    help="训练的batch size")
add_arg("per_device_eval_batch_size",  type=int, default=1,    help="评估的batch size")
add_arg("gradient_accumulation_steps", type=int, default=1,    help="梯度累积步数")
args = parser.parse_args()
print_arguments(args)


# 获取Whisper的数据处理器，这个包含了特征提取器、tokenizer
processor = WhisperProcessor.from_pretrained(args.base_model,
                                             language=args.language,
                                             task=args.task,
                                             no_timestamps=not args.timestamps,
                                             local_files_only=args.local_files_only)

# 读取数据
train_dataset = CustomDataset(data_list_path=args.train_data,
                              processor=processor,
                              language=args.language,
                              timestamps=args.timestamps,
                              min_duration=args.min_audio_len,
                              max_duration=args.max_audio_len,
                              augment_config_path=args.augment_config_path)
test_dataset = CustomDataset(data_list_path=args.test_data,
                             processor=processor,
                             language=args.language,
                             timestamps=args.timestamps,
                             min_duration=args.min_audio_len,
                             max_duration=args.max_audio_len)
print(f"训练数据：{len(train_dataset)}，测试数据：{len(test_dataset)}")
# 数据padding器
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# 获取Whisper模型
device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

# 获取模型
model = WhisperForConditionalGeneration.from_pretrained(args.base_model,
                                                        load_in_8bit=args.use_8bit,
                                                        device_map=device_map,
                                                        local_files_only=args.local_files_only)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# 注册forward，否则多卡训练会失败
model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)


if args.base_model.endswith("/"):
    args.base_model = args.base_model[:-1]

output_dir = args.output_dir
# 定义训练参数
training_args = \
    Seq2SeqTrainingArguments(output_dir=output_dir,  # 保存检查点和意志的目录
                             per_device_train_batch_size=args.per_device_train_batch_size,  # 训练batch_size大小
                             per_device_eval_batch_size=args.per_device_eval_batch_size,  # 评估batch_size大小
                             gradient_accumulation_steps=args.gradient_accumulation_steps,  # 训练梯度累计步数
                             learning_rate=args.learning_rate,  # 学习率大小
                             warmup_steps=args.warmup_steps,  # 预热步数
                             num_train_epochs=args.num_train_epochs,  # 微调训练轮数
                             save_strategy="steps",  # 指定按照步数保存检查点
                             eval_strategy="steps",  # 指定按照步数评估模型
                             save_safetensors=False,
                             load_best_model_at_end=False,  # 指定是否在结束时加载最优模型
                             fp16=args.fp16,  # 是否使用半精度训练
                             report_to=["tensorboard"],  # 指定使用tensorboard保存log
                             save_steps=args.save_steps,  # 指定保存检查点的步数
                             eval_steps=args.eval_steps,  # 指定评估模型的步数
                             save_total_limit=5,  # 只保存最新检查点的数量
                             optim='adamw_torch',  # 指定优化方法
                             ddp_find_unused_parameters=False if ddp else None,  # 分布式训练设置
                             dataloader_num_workers=args.num_workers,  # 设置读取数据的线程数量
                             logging_steps=args.logging_steps,  # 指定打印log的步数
                             remove_unused_columns=False,  # 删除模型不需要的数据列
                             label_names=["labels"])  # 与标签对应的输入字典中的键列表

if args.freeze_encoder:
    print('Model freeze encoder!')
    model.freeze_encoder()

# 使用Pytorch2.0的编译器
if torch.cuda.is_available() and torch.__version__ >= "2" and platform.system().lower() != 'windows':
    model = torch.compile(model)

# 定义训练器
trainer = Seq2SeqTrainer(args=training_args,
                         model=model,
                         train_dataset=train_dataset,
                         eval_dataset=test_dataset,
                         data_collator=data_collator,
                         tokenizer=processor.feature_extractor,
                         )

# =============== Robust Patch for RNG State Loading (Handles list/tensor) ===============
def _patched_load_rng_state(trainer_instance, checkpoint_path):
    local_rank = trainer_instance.args.local_rank
    if local_rank == -1:
        rng_file = os.path.join(checkpoint_path, "rng_state.pth")
    else:
        rng_file = os.path.join(checkpoint_path, f"rng_state_{local_rank}.pth")

    if not os.path.isfile(rng_file):
        print(f"Warning: {rng_file} not found. Skipping RNG state restoration.")
        return

    try:
        rng_dict = torch.load(rng_file, map_location="cpu", weights_only=False)

        # Helper: convert list to torch.ByteTensor if needed
        def to_tensor_if_list(x):
            if isinstance(x, list):
                return torch.tensor(x, dtype=torch.uint8)
            return x

        # Restore torch RNG state
        if "torch" in rng_dict:
            state = to_tensor_if_list(rng_dict["torch"])
            torch.set_rng_state(state)

        # Restore CUDA RNG state
        if "cuda" in rng_dict:
            if torch.cuda.is_available():
                state = to_tensor_if_list(rng_dict["cuda"])
                torch.cuda.set_rng_state(state)
            else:
                print("Warning: CUDA RNG state found but CUDA is not available.")

        # Restore NumPy and Python states (they are tuples, usually fine)
        if "numpy" in rng_dict:
            np.random.set_state(rng_dict["numpy"])
        if "python" in rng_dict:
            random.setstate(rng_dict["python"])

    except Exception as e:
        print(f"Warning: Failed to load RNG state from {rng_file}: {e}")

# Apply patch
trainer._load_rng_state = lambda cp: _patched_load_rng_state(trainer, cp)
# ======================================================================================


model.config.use_cache = False
trainer._load_from_checkpoint = load_from_checkpoint

# 开始训练
trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

# 保存最后的模型
trainer.save_state()
if training_args.local_rank == 0 or training_args.local_rank == -1:
    model.save_pretrained(os.path.join(output_dir, "checkpoint-final"))
