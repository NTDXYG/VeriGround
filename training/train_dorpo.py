from unsloth import FastVisionModel
import pandas as pd
from datasets import Dataset
from PIL import Image

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--decision_tokens", type=int, default=16)
parser.add_argument("--decision_weight", type=float, default=3.0)
args = parser.parse_args()

decision_tokens = args.decision_tokens
decision_weight = args.decision_weight

# 模型加载
model, tokenizer = FastVisionModel.from_pretrained(
    "/home/yg/code_projects/train_qwen3_5/4B-sft-anony",
    load_in_4bit=False,
    use_gradient_checkpointing=True,
)

# LoRA 配置
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
)

# 读取数据
df = pd.read_csv("orpo_train_data.csv")
dataset = Dataset.from_pandas(df)

formatted_dataset = dataset

# ORPO 训练配置
from trl import ORPOConfig
from orpo_vision_trainer import VisionORPOTrainer
from orpo_vision_collator import ORPOVisionDataCollator

FastVisionModel.for_training(model)

run_name = f"dt{decision_tokens}_dw{decision_weight}"

config = ORPOConfig(
  max_length=4096,
  max_prompt_length=2048,
  max_completion_length=2048,
  per_device_train_batch_size=1,
  gradient_accumulation_steps=8,
  beta=0.1,
  learning_rate=5e-6,
  logging_steps=1,
  optim="adamw_8bit",
  num_train_epochs=1,
  output_dir=f"dorpo_outputs/{run_name}",
  report_to="tensorboard",
  remove_unused_columns=False,
  label_pad_token_id=-100,
  # 注意：VisionORPOTrainer 不需要 dataset_num_proc
)


# 创建 DataCollator
data_collator = ORPOVisionDataCollator(
  model=model,
  tokenizer=tokenizer,
  max_length=4096,
)

trainer = VisionORPOTrainer(
  model=model,
  args=config,
  train_dataset=formatted_dataset,
  data_collator=data_collator,
  processing_class=tokenizer,
  use_dorpo=True,
  decision_tokens=decision_tokens,
  decision_weight=decision_weight,
)

trainer_stats = trainer.train()

# 保存时也用 run_name 区分
model.save_pretrained(f"dorpo-lora/{run_name}")
tokenizer.save_pretrained(f"dorpo-lora/{run_name}")
model.save_pretrained_merged(f"dorpo/{run_name}", tokenizer)