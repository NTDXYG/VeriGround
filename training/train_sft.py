from unsloth import FastVisionModel # FastLanguageModel for LLMs
import pandas as pd
from datasets import Dataset
from PIL import Image

model, tokenizer = FastVisionModel.from_pretrained(
    "/model/Qwen3.5-4B-unsloth",
    load_in_4bit = False, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = True, # True or "unsloth" for long context
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 16,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)

# 读取本地 CSV
df = pd.read_csv("train_data.csv")
# 转换为 HuggingFace Dataset 格式
dataset = Dataset.from_pandas(df)


def convert_to_conversation(sample):
    image_path = sample["image"]
    image = Image.open(image_path).convert("RGB")  # 转为 RGB
  
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "image", "image" : image},
            {"type" : "text",  "text"  : f"""Please write a Verilog module based on the provided circuit diagram image.
Return only the verilog code, without any explanation.

For example:
```verilog
your Verilog code here
```

Module header (must not be changed):
{sample["module_header"]}
"""}]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : "```verilog\n" + sample["verilog"] + "\n```"}]
        },
    ]
    return { "messages" : conversation }

# 转换数据集
converted_dataset = [convert_to_conversation(sample) for sample in dataset]

total = len(converted_dataset)
train_size = total - 2500
train_dataset = converted_dataset.select(range(train_size))
eval_dataset = converted_dataset.select(range(train_size, total))

print(f"总样本数: {len(converted_dataset)}")
print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(eval_dataset)}")

from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

FastVisionModel.for_training(model) # Enable for training!

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 5, # Set this instead of max_steps for full training runs
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "sft_outputs",
        report_to = "tensorboard",     # For Weights and Biases

        # Evaluation settings:
        eval_strategy = "steps",
        eval_steps = 1000,

        # Save best model settings:
        save_strategy = "steps",
        save_steps = 1000,
        load_best_model_at_end = True,
        metric_for_best_model = "eval_loss",
        greater_is_better = False,

        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_length = 4096,
    ),
)

trainer_stats = trainer.train()

model.save_pretrained("4B-sft-original-lora")  # Local saving
tokenizer.save_pretrained("4B-sft-original-lora")

model.save_pretrained_merged("4B-sft-original", tokenizer,)