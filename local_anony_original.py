import json
import os
import re
import anony_utils
from typing import Dict, Any, List
from dataclasses import dataclass
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoProcessor
from PIL import Image


# 设置工作目录为当前文件所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class ImageProblem:
    prompt: str
    module_header: str
    norm_module_header: str
    module_name: str
    image_path: str  # jpg 图片路径
    source: str
    level: str
    rename_map: Dict[str, str]

class LocalImageVerilogGenerator:
    def __init__(self, model_path: str, max_tokens: int = 40960):
        """Initialize the generator with local multimodal LLM using vllm"""
        self.model = LLM(
            model=model_path, 
            gpu_memory_utilization=0.9, 
            max_model_len=max_tokens, 
            trust_remote_code=True,
            # 多模态模型需要启用 vision 相关配置
            limit_mm_per_prompt={"image": 1}  # 每个 prompt 最多 1 张图
        )
        # 输出的最大长度
        self.max_tokens = 2048
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    def _create_messages(self, problem: ImageProblem) -> List[Dict[str, Any]]:
        """创建包含图片的多模态消息"""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image"
                    },
                    {
                        "type": "text",
                        "text": f"""Please write a Verilog module based on the provided circuit diagram image.
Return only the verilog code, without any explanation.

For example:
```verilog
your Verilog code here
```

Module header (must not be changed):
{problem.norm_module_header}
"""
                    }
                ]
            }
        ]
        return messages

    def generate_solutions(self, problems: List[ImageProblem], k: int) -> List[Dict[str, Any]]:
        """Generate k solutions for multiple problems using local LLM with vllm"""
        prompts = []
        for problem in problems:
            messages = self._create_messages(problem)
            prompt = self.processor.apply_chat_template(
                messages, 
                tokenize=False,  
                add_generation_prompt=True,
                enable_thinking=False
            )
            # 加载图片并转换为 RGB
            image_obj = Image.open('benchmark/' + problem.image_path).convert("RGB")
            prompts.append({
                "prompt": prompt,
                "multi_modal_data": {
                    "image": image_obj
                }
            })

        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7,  
            max_tokens=self.max_tokens,
            n=k  # Generate k samples for each prompt
        )
        
        # Generate samples for all prompts at once
        outputs = self.model.generate(
            prompts=prompts,
            sampling_params=sampling_params
        )
        all_solutions = []
        # Process each generated output
        for i, output in enumerate(outputs):
            problem = problems[i]
            solutions = []
            for sample in output.outputs:
                generated_text = sample.text
                verilog_code = self._extract_verilog_code(generated_text)
                verilog_code = anony_utils.restore_verilog(verilog_code, problem.rename_map)
                solutions.append({"solution": verilog_code, "pass": ""})
            result = {
                "module_name": problem.module_name,
                "solutions": solutions,
                "source": problem.source,
                "level": problem.level
            }
            all_solutions.append(result)

        return all_solutions

    def _extract_verilog_code(self, content: str) -> str:
        code_block_pattern = r"```verilog\n([\s\S]*?)\n```"
        match = re.search(code_block_pattern, content, re.IGNORECASE)
        return match.group(1) if match else content


def generate_solutions(config):
    
    all_problems = []
    # 加载问题数据（支持 .json 和 .jsonl）
    with open(config["prompt_file"], "r", encoding="utf-8") as f:
        if config["prompt_file"].endswith(".jsonl"):
            records = [json.loads(line) for line in f if line.strip()]
        else:
            records = json.load(f)

    for item in records:
        rename_map, normalize_module_header = anony_utils.analyze_and_normalize(item.get("module_header", ""))
        image_path=item.get("image", "")
        image_path = image_path.replace("images_normal/", "images_anony/")
        problem = ImageProblem(
            prompt=item.get("prompt", ""),
            module_header=item.get("module_header", ""),
            norm_module_header=normalize_module_header,
            module_name=item.get("module_name", ""),
            image_path=image_path,
            source=item.get("source", ""),
            level=item.get("level", ""),
            rename_map=rename_map
        )
        all_problems.append(problem)

    print(f"Generating {config['k']} solutions for {len(all_problems)} problems...")

    # Initialize the local model generator
    generator = LocalImageVerilogGenerator(model_path=config["model_path"])

    # Generate solutions
    all_solutions = generator.generate_solutions(all_problems, config["k"])

    # 保存结果
    out_dir = "results_anony"
    os.makedirs(out_dir, exist_ok=True)
    output_file_name = f"{out_dir}/pass{config['k']}_{config['model_name']}_original.json"
    with open(output_file_name, "w", encoding="utf-8") as f:
        json.dump(all_solutions, f, ensure_ascii=False, indent=4)

    print(f"All solutions generated and saved to {output_file_name}")


if __name__ == "__main__":

    model_dict = {
        "Qwen3.5-4B": "/model/Qwen3.5-4B",
        "Qwen3.5-9B": "/model/Qwen3.5-9B",
        "EGM-4B": "/model/EGM-4B",
        "EGM-8B": "/model/EGM-8B",
        "4B-sft": "/home/yg/code_projects/train_qwen3_5/4B-sft-original",
        "4B-sft-anony": "/home/yg/code_projects/train_qwen3_5/4B-sft-anony",
        "4B-orpo": "/home/yg/code_projects/train_qwen3_5/4B-orpo",
        "4B-dorpo": "/home/yg/code_projects/train_qwen3_5/4B-dorpo",
        "9B-sft-anony": "/home/yg/code_projects/train_qwen3_5/9B-sft-anony",
    }
    
    dorpo_model_dict = {}
    for dt in [8]:
        for dw in [2.0]:
            dorpo_model_dict[f"4B-dorpo-dt{dt}-dw{dw}"] = f"/home/yg/code_projects/train_qwen3_5/dorpo/dt{dt}_dw{dw}"

    model_names = dorpo_model_dict.keys()
    k_values = [5]
    for model_name in model_names:
        for k in k_values:
            config = {
                "model_path": dorpo_model_dict[model_name],
                "model_name": model_name,
                "prompt_file": "benchmark/problems.json",
                "k": k
            }
            generate_solutions(config)