# From Mirage to Grounding: Towards Reliable Multimodal Circuit-to-Verilog Code Generation

Code, **C2VEval** benchmark, and **VeriGround** training scripts for the paper of the same title.

## ⚡ Overview

We study **Mirage**: MLLMs can ignore circuit images and latch onto **identifier semantics** in `module_header` instead. Swapping diagrams for blanks barely hurts Pass@k (sometimes helps).

This repo provides:

1. **C2VEval** (169 tasks): paired **Normal / Anony** protocols.
2. **VeriGround**: anonymization, refusal augmentation, and **D-ORPO** (Decision-Focused ORPO).

## 🤗 Released models (Hugging Face)

| Model | Stage / role |
| :--- | :--- |
| [RTL-Series/VeriGround-mixed-sft](https://huggingface.co/RTL-Series/VeriGround-mixed-sft) | Stage 1: mixed supervised fine-tuning (Normal + Anony) |
| [RTL-Series/VeriGround-orpo](https://huggingface.co/RTL-Series/VeriGround-orpo) | Stage 2 baseline: standard ORPO|
| [RTL-Series/VeriGround-dorpo](https://huggingface.co/RTL-Series/VeriGround-dorpo) | Stage 2: **D-ORPO** aligned checkpoint |

Load with Hugging Face `transformers` or point local eval scripts (`local_*.py`) at a downloaded snapshot.

## 📁 Repository layout

```
code/
├── benchmark/                   # C2VEval
│   ├── problems.json
│   ├── problems_mismatch_*.json # preference pairs
│   ├── ground_truth.json
│   ├── testbench.jsonl
│   ├── images_normal/
│   ├── images_anony/
│   └── white.jpg                # blank image (Mirage)
├── training/
│   ├── train_mixed_sft.py
│   ├── train_orpo.py
│   ├── train_dorpo.py
│   ├── orpo_vision_trainer.py
│   └── orpo_vision_collator.py
├── anony_utils.py
├── functional_correctness.py
├── stat_test.py
├── api_*_*.py                   # OpenAI-compatible API eval
├── local_*_*.py                  # vLLM local eval
└── local_mismatch.py
```

## 📊 C2VEval (169 samples)

| Category | Count | Share |
|---|---|---|
| Basic combinational logic | 81 | 48.5% |
| Sequential building blocks | 43 | 25.7% |
| Finite state machines | 32 | 19.2% |
| Math / algorithms | 11 | 6.6% |

Each sample is $(I, H, V^*, \mathcal{T}, D)$ with anonymized pair $(I_{\text{anon}}, H_{\text{anon}})$. **Anony** renames module and ports to placeholders so models must ground on the image.

## 🧪 Evaluation

**Deps**

```bash
pip install openai tqdm vllm transformers pillow
# functional sim: iverilog (https://github.com/steveicarus/iverilog)
```

**API**

```bash
python api_normal_original.py    # Normal, real images
python api_normal_mirage.py      # Normal, blank (Mirage)
python api_anony_original.py     # Anony, anonymized images
python api_anony_mirage.py       # Anony, blank
```

Configure API key and model inside each script; outputs are JSON.

**Local ([vLLM](https://github.com/vllm-project/vllm))**

```bash
python local_normal_original.py
python local_normal_mirage.py
python local_anony_original.py
python local_anony_mirage.py
python local_mismatch.py         # refusal / mismatch
```

Set `model_path` to a local dir or HF snapshot.

**Pass@k / simulation**

```bash
python functional_correctness.py   # edit solutions_file path
python stat_test.py                 # e.g., McNemar
```

**Anonymization**

```python
from anony_utils import anonymize_verilog
anon_code, mapping = anonymize_verilog(verilog_code)
```

## 📈 Paper numbers (VeriGround 4B setting)

| Metric | Normal | Anony |
|---|---|---|
| Functional Pass@1 | 46.11% | 42.51% |
| False refusal rate | 1.20% | 0.00% |
| Refusal rate (blank images) | ≥92% | ≥97% |

## 📚 Citation

```bibtex
@article{yang2026mirage,
  title={From Mirage to Grounding: Towards Reliable Multimodal Circuit-to-Verilog Code Generation},
  author={Yang, Guang and Hu, Xing and Chen, Xiang and Xia, Xin},
  journal={ArXiv},
  year={2026}
}
```
