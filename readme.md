# From Mirage to Grounding: Towards Reliable Multimodal Circuit-to-Verilog Code Generation

Code, **C2VEval** benchmark, and **VeriGround** training/evaluation scripts for the paper *"From Mirage to Grounding: Towards Reliable Multimodal Circuit-to-Verilog Code Generation"*.

## ⚡ Overview

We study the **Mirage** phenomenon in multimodal circuit-to-Verilog code generation: existing MLLMs largely bypass the circuit diagram and instead exploit **identifier semantics** in `module_header` to retrieve canonical RTL templates. Consequently, replacing the circuit diagram with a blank image leaves Pass@$k$ unchanged—or even **higher** when complex schematics confuse the model.

This repository provides:

1. **C2VEval** (167 tasks across 4 categories): paired **Normal / Anony** and **Original / Mirage** evaluation protocols, plus five rounds of image–header mismatch inputs.
2. **VeriGround (4B)**: a grounded circuit-to-Verilog model trained with identifier anonymization, refusal augmentation, and **D-ORPO** (Decision-Focused ORPO).
3. **Fixed-header topology pairs**: 22 original/mutant combinational RTL pairs with identical anonymized interfaces and exhaustive binary-input functional checks recorded in the metadata. Mutant diagram rendering and synthesized-topology checks are still pending.

---

## 🔍 Motivating Examples of the Mirage Phenomenon (Section I)

In standard evaluation, the model receives a circuit diagram $I$ together with its `module_header` $H$. The first two examples below use **Opus-4.6** to illustrate how $H$ acts as a textual shortcut and why providing the real diagram can even hurt performance. The third illustrates the anonymized protocol.

### Example 1: Identical Correct Generation With or Without the Diagram (`hadd` / Fig. 1 in Paper)

Even when the module name is generic (`TopModule`), descriptive port identifiers (`a`, `b`, `sum`, `cout`) in `module_header` uniquely reveal a half-adder:

```verilog
module TopModule (
    input  a,
    input  b,
    output cout,
    output sum
);
```

- **With Real Circuit Diagram (`Original` — Passed ✅):**
  ```verilog
  assign sum  = a ^ b;
  assign cout = a & b;
  ```
- **With Blank White Image (`Mirage` — Passed ✅):**
  ```verilog
  assign sum  = a ^ b;
  assign cout = a & b;
  ```
- **Takeaway:** The model emits the exact same correct RTL without seeing any gates or wires, relying entirely on port names (`sum`, `cout`).

---

### Example 2: Visual Interference — Failing With the Real Diagram but Succeeding With a Blank Image (`fsm_3state`)

For a 3-state cyclic finite state machine (`fsm_3state`, image at `benchmark/images_normal/fsm_3state.jpg`), the header is:

```verilog
module fsm_3state (
    input clk, reset,
    output [1:0] state
);
```

- **With Real Circuit Diagram (`Original` — Failed ❌):**  
  When shown the Yosys/`netlistsvg` gate-level schematic (which decomposes the state transitions into flip-flops, inverters, muxes, and feedback wires), Opus-4.6 attempts to trace each primitive gate and misreads the next-state and output wiring:
  ```verilog
  module fsm_3state (
      input clk, reset,
      output [1:0] state
  );
      reg q0, q1;
      wire d0, d1, nq0, nq1;
      assign nq0 = ~q0;
      assign nq1 = ~q1;
      assign d0  = nq0;
      assign d1  = q0 ^ q1;

      always @(posedge clk) begin
          if (reset) begin q0 <= 0; q1 <= 0; end
          else       begin q0 <= d0; q1 <= d1; end
      end

      assign state[1] = q0 ^ q1;
      assign state[0] = q0 & q1; // Incorrect transition & output mapping
  endmodule
  ```
- **With Blank White Image (`Mirage` — Passed ✅):**  
  When the diagram is replaced with a blank image (`benchmark/white.jpg`), the model falls back on the identifier `fsm_3state` and retrieves the canonical 3-state cyclic FSM template (`2'b00 -> 2'b01 -> 2'b10 -> 2'b00`), passing all testbench assertions:
  ```verilog
  module fsm_3state (
      input clk, reset,
      output [1:0] state
  );
      reg [1:0] current_state;
      always @(posedge clk or posedge reset) begin
          if (reset)
              current_state <= 2'b00;
          else begin
              case (current_state)
                  2'b00:   current_state <= 2'b01;
                  2'b01:   current_state <= 2'b10;
                  2'b10:   current_state <= 2'b00;
                  default: current_state <= 2'b00;
              endcase
          end
      end
      assign state = current_state;
  endmodule
  ```
- **Takeaway:** On structurally dense sequential and FSM schematics, unadapted MLLMs suffer from **visual interference** (`Mirage > Original`): they cannot reliably parse gate-level feedback loops, whereas removing the image lets them guess the canonical textbook template from the module header.

---

### Example 3: How `Anony` Eliminates Header Shortcuts Under Identical Headers

Under the **Anony** protocol, all module and port names are replaced with positional placeholders (`module_name`, `val_0`, `val_1`, ...). Distinct circuits with the same port signature now share the **exact same header**:

```verilog
module module_name (
    input  val_0,
    input  val_1,
    output val_2
);
```

- For `andgate`, `norgate`, and `xnorgate`, the header above is identical; only the gate symbol inside $I_{\text{anon}}$ differs.
- Zero-shot baselines collapse under `Anony`, whereas **VeriGround** reads the depicted gate symbol and generates `val_0 ? val_1 : 1'b0` (AND), `~val_0 & ~val_1` (NOR), and `(val_0 == val_1)` (XNOR), respectively.

---

## 🤗 Released Models & Datasets (Hugging Face)

The following model checkpoints are hosted on Hugging Face ([https://huggingface.co/RTL-Series](https://huggingface.co/RTL-Series)). The C2VEval benchmark is included under `benchmark/`; training scripts also require local CSV files and their referenced images (see the training section below).

| Resource | Stage / Role |
| :--- | :--- |
| [RTL-Series/VeriGround-mixed-sft](https://huggingface.co/RTL-Series/VeriGround-mixed-sft) | Stage 1: Mixed supervised fine-tuning (Normal + Anony) |
| [RTL-Series/VeriGround-orpo](https://huggingface.co/RTL-Series/VeriGround-orpo) | Stage 2 baseline: Standard ORPO |
| [RTL-Series/VeriGround-dorpo](https://huggingface.co/RTL-Series/VeriGround-dorpo) | Stage 2: **D-ORPO** aligned checkpoint ($K=8, \alpha=2.0$) |

Load with Hugging Face `transformers` or point local evaluation scripts (`local_*.py`) at a downloaded snapshot.

---

## 📁 Repository Layout

```
code/
├── benchmark/                    # C2VEval benchmark (167 tasks)
│   ├── problems.json             # Problem metadata, headers, and categories
│   ├── problems_mismatch_*.json  # 5 rounds of mismatched image-header pairs
│   ├── topology_pairs.json       # 22 pairs: RTL, mutations, truth tables, testbenches
│   ├── problems_topology_original.json # Original-side inference inputs
│   ├── problems_topology_mutant.json   # Mutant-side inputs; images pending
│   ├── ground_truth.json         # Reference RTL in solutions[].solution
│   ├── testbench.jsonl           # Simulation testbenches
│   ├── images_normal/            # Normal netlistsvg schematics
│   ├── images_anony/             # Anonymized netlistsvg schematics
│   └── white.jpg                 # Blank image for Mirage evaluation
├── all_results/                  # Archived model outputs with simulation annotations
│   ├── results_normal/           # Normal Original/Mirage, pass1 and pass5
│   ├── results_anony/            # Anony Original/Mirage, pass1 and pass5
│   └── results_mismatch/         # Five mismatch rounds, pass1
├── training/                     # SFT, ORPO, and D-ORPO training scripts
│   ├── train_sft.py
│   ├── train_mixed_sft.py
│   ├── train_orpo.py
│   ├── train_dorpo.py
│   ├── orpo_vision_trainer.py
│   └── orpo_vision_collator.py
├── anony_utils.py                # Header normalization & identifier restoration
├── build_topology_pairs.py       # Fixed-header RTL mutation & exhaustive validation
├── functional_correctness.py     # Icarus Verilog compilation & simulation
├── stat_test.py                  # Paired McNemar tests with Holm–Bonferroni correction
├── requirements.txt              # Python dependencies (mostly minimum versions)
├── api_*_*.py                    # OpenAI-compatible API evaluation scripts
├── local_*_*.py                  # vLLM local evaluation scripts
└── local_mismatch.py             # Mismatch refusal evaluation script
```

---

## 📊 C2VEval Benchmark (167 Samples)

| Category | Count | Ratio |
| :--- | :---: | :---: |
| Basic Combinational Logic (BCL) | 81 | 48.5% |
| Sequential Building Blocks (SBB) | 43 | 25.7% |
| Finite State Machines (FSM) | 32 | 19.2% |
| Mathematical Operations & Algorithms (MOA) | 11 | 6.6% |
| **Total** | **167** | **100%** |

Each sample is a tuple $(I, H, V^*, \mathcal{T}, D)$ paired with its anonymized counterpart $(I_{\text{anon}}, H_{\text{anon}})$.

`problems.json` contains the prompt, module header, image path, source, and category (`level`). Reference RTL is stored in `ground_truth.json` under `solutions[].solution`; simulation testbenches are in `testbench.jsonl`. Join these records by `module_name`.

| Protocol | Header / identifiers | Image |
| :--- | :--- | :--- |
| Normal–Original | Original identifiers | `images_normal/` circuit diagram |
| Normal–Mirage | Original identifiers | Blank `white.jpg` |
| Anony–Original | Positional module/port placeholders | `images_anony/` circuit diagram |
| Anony–Mirage | Positional module/port placeholders | Blank `white.jpg` |
| Mismatch | Header paired with a mismatched diagram | Five input files, `problems_mismatch_1.json` through `problems_mismatch_5.json` |

The Anony scripts normalize the input header and restore generated identifiers for evaluation against the original testbenches. Mismatch evaluates responses to inconsistent image–header inputs; it is separate from the fixed-header topology experiment below.

---

## 🧪 Environment & Evaluation

**1. Install dependencies**

Run the following evaluation commands from `code/` so the relative benchmark and output paths resolve correctly:

```bash
cd code
pip install -r requirements.txt
# Install Icarus Verilog (iverilog & vvp) for functional simulation:
# https://github.com/steveicarus/iverilog
```

**2. Run API evaluation**

Before running, edit each script's `__main__` configuration: select the models in `model_dict`, configure your OpenAI-compatible `base_url` and `api_key`, and set `prompt_file`, `k`, and concurrency as needed. Supply your own credentials securely; do not commit credentials. These scripts use source-level configuration rather than command-line model/API flags.

```bash
python api_normal_original.py    # Normal, real circuit diagrams
python api_normal_mirage.py      # Normal, blank image (Mirage)
python api_anony_original.py     # Anony, anonymized diagrams
python api_anony_mirage.py       # Anony, blank image (Mirage)
```

**3. Run local evaluation ([vLLM](https://github.com/vllm-project/vllm))**

Configure the local checkpoint paths and active model loop in each script before launching. Some entry points iterate over `dorpo_model_dict` rather than `model_dict`, including multiple D-ORPO settings. Also select `k_values` and `prompt_file`; these are not exposed as command-line flags. Use a GPU environment compatible with vLLM and the selected checkpoint.

```bash
python local_normal_original.py
python local_normal_mirage.py
python local_anony_original.py
python local_anony_mirage.py
python local_mismatch.py         # Mismatch refusal evaluation
```

Generation writes `results_normal/`, `results_anony/`, or `results_mismatch/` under the working directory. The checked-in experiment outputs are archived separately under `all_results/`. Filenames follow `pass{k}_{model_name}_{original|mirage}.json`, or `pass{k}_{model_name}_mismatch_{round}.json`. A `pass5` file contains five candidate solutions per task; not every model has both sample counts.

**4. Compute Pass@$k$ and statistical tests**

For example, evaluate a newly generated Normal–Original result file:

```bash
python functional_correctness.py \
    --solutions results_normal/pass1_4B-dorpo-dt8-dw2.0_original.json \
    --problems benchmark/testbench.jsonl \
    --levels-json benchmark/problems.json \
    --timeout 5
python stat_test.py
```

`functional_correctness.py` compiles and simulates each candidate with `iverilog`/`vvp`, **updates the supplied solutions JSON in place** with per-solution `pass` annotations, and prints overall and per-category syntax/functional Pass@$k$. Use a copy to preserve archived annotations. The reported $k$ follows the number of candidate solutions in the input file.

`stat_test.py` reads the archived `all_results/results_normal/` and `all_results/results_anony/` Pass@1 Original files. It compares `4B-dorpo-dt8-dw2.0` with MiMo-v2-omni, GPT-4o, GPT-5.4, and Opus-4.6, aligns tasks by `module_name`, and applies Holm–Bonferroni correction over eight paired McNemar comparisons. To analyze a new run, update its configured result paths/model list or place the evaluated outputs at those paths.

---

## 🔬 Fixed-Header Topology Pairs (Preparation Stage)

`build_topology_pairs.py` prepares an additional controlled experiment: within each pair, the anonymized `module_header` stays exactly the same while one supported scalar binary operator or two-input primitive gate in the reference RTL changes. The goal is to test whether generated behavior follows the depicted circuit when the interface is held fixed.

The builder uses `benchmark/problems.json` and `benchmark/ground_truth.json`, considers only Basic Combinational Logic, and restricts candidates to single-module continuous combinational RTL. It checks the original reference against its source testbench, checks anonymization equivalence, and exhaustively compares original/mutant truth tables over all binary input combinations. Functionally equivalent mutations and designs with undefined outputs are rejected. Selection uses a seed and does not read model evaluation results.

The current snapshot contains **22 pairs**:

| File | Contents / use |
| :--- | :--- |
| `benchmark/topology_pairs.json` | Reference and anonymized RTL, mutation locations, rename maps, original/mutant testbenches, truth tables, distinguishing inputs, and verification status |
| `benchmark/problems_topology_original.json` | 22 inference records reusing existing `images_anony/` images |
| `benchmark/problems_topology_mutant.json` | 22 inference records referring to `images_topology_mutant/topology_*.jpg`; those images are not yet included |

Both inference files use neutral task IDs (`topology_000`, etc.) and contain matching headers/prompts for each pair. Keep `topology_pairs.json`, including its reference RTL and oracle testbenches, outside model inputs.

To rebuild with the current script, prepare a **separate benchmark directory** containing copies of `problems.json` and `ground_truth.json`, then run from `code/`:

```bash
python build_topology_pairs.py \
    --benchmark /path/to/new-benchmark \
    --count 30 \
    --seed 3407 \
    --max-input-bits 12
```

`iverilog` and `vvp` must be on `PATH`. The defaults request up to 30 pairs with seed 3407 and at most 12 input bits per circuit; fewer pairs are returned if fewer modules qualify. `--max-input-bits` accepts 1–16. The script writes four files: the three JSON files above plus `topology_sampling_report.json` with sampling settings, exclusions, and input hashes. The report is not included in the current snapshot. **The builder refuses to run if any target output already exists**, so running it against the checked-in `benchmark/` directory will stop without rebuilding.

The pair metadata records source-testbench, compilation, anonymization, and exhaustive functional checks as passed. **Synthesis, confirmation of a single-gate change after synthesis, mutant rendering, and visual-token-budget checks remain pending.** The script performs RTL preparation and simulation, not diagram rendering. Original images reuse existing anonymized diagrams; their correspondence to the reference RTL must also be checked. This snapshot is not yet a complete image-based topology benchmark. Its neutral task IDs and variant-specific oracle testbenches also require a matching evaluation setup; the standard `benchmark/testbench.jsonl` cannot be used directly for these pairs.

---

## 🏋️ Training

Training uses Unsloth, the shared vision ORPO trainer/collator, local CSV files, and the image paths referenced by those files. Configure checkpoint paths, CSV locations, and output directories in the scripts before running them in a compatible GPU environment. CSV paths are resolved relative to the working directory; the files below are not included in this code snapshot.

| Script | Local CSV input | Role |
| :--- | :--- | :--- |
| `training/train_sft.py` | `train_data.csv` | Standard supervised fine-tuning |
| `training/train_mixed_sft.py` | `train_anony_merged.csv` | Mixed Normal/Anony supervised fine-tuning |
| `training/train_orpo.py` | `train_orpo.csv` | Standard ORPO baseline |
| `training/train_dorpo.py` | `orpo_train_data.csv` | Decision-focused ORPO |

Run the desired stage from a directory containing its configured CSV and image paths. For example, from `code/` after preparing the inputs:

```bash
python training/train_mixed_sft.py
python training/train_dorpo.py --decision_tokens 8 --decision_weight 2.0
```

The explicit D-ORPO arguments reproduce the reported setting $K=8, \alpha=2.0$; the script's defaults are `--decision_tokens 16 --decision_weight 3.0`. Set its input checkpoint to the prepared SFT model. For the example above, D-ORPO saves trainer outputs to `dorpo_outputs/dt8_dw2.0`, LoRA weights to `dorpo-lora/dt8_dw2.0`, and the merged model to `dorpo/dt8_dw2.0`.

---

## 📈 Main Results (VeriGround-4B, $K=8, \alpha=2.0$)

| Metric | Normal | Anony |
| :--- | :---: | :---: |
| Functional Pass@1 (`Original`) | 46.11% | 42.51% |
| Functional Pass@1 (`Mirage`) | 0.00% | 0.00% |
| False Refusal Rate (FRR $\downarrow$) | 1.20% (2/167) | 0.00% (0/167) |
| Blank-Image Refusal Rate (RR $\uparrow$) | 92.81% | 97.60% |

---

## 📚 Citation

```bibtex
@article{yang2026mirage,
  title={From Mirage to Grounding: Towards Reliable Multimodal Circuit-to-Verilog Code Generation},
  author={Yang, Guang and Hu, Xing and Chen, Xiang and Xia, Xin},
  journal={arXiv preprint},
  year={2026}
}
```
