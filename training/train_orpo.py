from unsloth import FastVisionModel
import pandas as pd
from datasets import Dataset
from PIL import Image

# 模型加载
model, tokenizer = FastVisionModel.from_pretrained(
    "/home/yg/code_projects/train_qwen3_5/4B-sft-anoy",
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
df = pd.read_csv("train_orpo.csv")
dataset = Dataset.from_pandas(df)

formatted_dataset = dataset

# ORPO 训练配置
from trl import ORPOConfig
from orpo_vision_trainer import VisionORPOTrainer
from orpo_vision_collator import ORPOVisionDataCollator

FastVisionModel.for_training(model)

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
  output_dir="orpo_outputs",
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
)

trainer_stats = trainer.train()

# 保存
model.save_pretrained("4B-orpo-lora")
tokenizer.save_pretrained("4B-orpo-lora")
model.save_pretrained_merged("4B-orpo", tokenizer)