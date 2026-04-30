import json
import os
import asyncio
import re
import base64
from typing import Dict, Any, List
from tqdm.asyncio import tqdm as async_tqdm
from openai import OpenAI
from dataclasses import dataclass


os.chdir(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class ImageProblem:
    prompt: str
    module_header: str
    module_name: str
    image_path: str
    source: str
    level: str

class ImageVerilogGenerator:
    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _agent_call(self, messages):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=1.0,
            max_tokens=4096,
            stream=False,
            timeout=120
        )
        return response.choices[0].message.content

    def _create_messages(self, problem: ImageProblem) -> List[Dict[str, Any]]:
        base64_image = self._encode_image('benchmark/' + problem.image_path)
        user_content = [
            {
                "type": "text",
                "text": f"""Please write a Verilog module based on the provided circuit diagram image.
Return only the verilog code, without any explanation.

For example:
```verilog
your Verilog code here
```

Module header (must not be changed):
{problem.module_header}
"""
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high"
                }
            }
        ]
        return [{"role": "user", "content": user_content}]

    async def _call_llm(self, problem: ImageProblem) -> str:
        try:
            messages = self._create_messages(problem)
            output_content = await asyncio.to_thread(self._agent_call, messages)
            return output_content
        except Exception as e:
            print(f"Error calling LLM for {problem.module_name}: {e}")
            return ""

    def _extract_verilog_code(self, content: str) -> str:
        code_block_pattern = r"```verilog\n([\s\S]*?)\n```"
        match = re.search(code_block_pattern, content, re.IGNORECASE)
        return match.group(1) if match else content

    async def process_problem(self, problem: ImageProblem, k: int) -> Dict[str, Any]:
        solutions = []
        for _ in range(k):
            output_content = await self._call_llm(problem)
            verilog_code = self._extract_verilog_code(output_content)
            solutions.append({"solution": verilog_code, "pass": ""})
        return {
            "module_name": problem.module_name,
            "solutions": solutions,
            "source": problem.source,
            "level": problem.level
        }

async def main(config: Dict[str, Any]):
    generator = ImageVerilogGenerator(
        config["api_key"],
        config["base_url"],
        config["model_path"]
    )
    all_problems = []

    with open(config["prompt_file"], "r", encoding="utf-8") as f:
        if config["prompt_file"].endswith(".jsonl"):
            records = [json.loads(line) for line in f if line.strip()]
        else:
            records = json.load(f)

    for item in records:
        problem = ImageProblem(
            prompt=item.get("prompt", ""),
            module_header=item.get("module_header", ""),
            module_name=item.get("module_name", ""),
            image_path=item.get("image", ""),
            source=item.get("source", ""),
            level=item.get("level", ""),
        )
        all_problems.append(problem)

    semaphore = asyncio.Semaphore(config["max_concurrent"])

    async def process_with_semaphore(problem: ImageProblem):
        async with semaphore:
            return await generator.process_problem(problem, config["k"])

    all_results = await async_tqdm.gather(
        *[process_with_semaphore(problem) for problem in all_problems],
        desc="Generating Verilog from images",
    )

    out_dir = "results_normal"
    os.makedirs(out_dir, exist_ok=True)
    output_file_name = f"{out_dir}/pass{config['k']}_{config['model_name']}_original.json"
    with open(output_file_name, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    print(f"\nResults saved to: {output_file_name}")

if __name__ == "__main__":
    model_dict = {
        'gpt-4o': 'gpt-4o',
        'mimo-v2-omni': 'mimo-v2-omni',
        'opus-4-6': 'claude-opus-4-6',
    }

    for model_name in model_dict:
        config = {
            "base_url": "https://www.dmxapi.cn/v1",
            "api_key": "sk-5OymMCZ7s3hiElaStFzjdJcJqyCFjLkWSki8UjiyoRVu3mcV",
            "model_path": model_dict[model_name],
            "model_name": model_name,
            "prompt_file": "benchmark/problems.json",
            "max_concurrent": 4,
            "k": 1,
        }
        asyncio.run(main(config))
