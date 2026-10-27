# From Mirage to Grounding: Towards Reliable Multimodal Circuit-to-Verilog Code Generation

Code, **C2VEval** benchmark, and **VeriGround** training/evaluation scripts for the paper *"From Mirage to Grounding: Towards Reliable Multimodal Circuit-to-Verilog Code Generation"*.

## ⚡ Overview

We study the **Mirage** phenomenon in multimodal circuit-to-Verilog code generation: existing MLLMs largely bypass the circuit diagram and instead exploit **identifier semantics** in `module_header` to retrieve canonical RTL templates. Consequently, replacing the circuit diagram with a blank image leaves Pass@$k$ unchanged—or even **higher** when complex schematics confuse the model.

This repository provides:

1. **C2VEval** (167 tasks across 4 categories): paired **Normal / Anony** and **Original / Mirage** evaluation protocols.
2. **VeriGround (4B)**: a grounded circuit-to-Verilog model trained with identifier anonymization, refusal augmentation, and **D-ORPO** (Decision-Focused ORPO).

---

## 🔍 Motivating Examples of the Mirage Phenomenon (Section I)

In standard evaluation, the model receives a circuit diagram $I$ together with its `module_header` $H$. Below are two representative examples from **Opus-4.6** illustrating how $H$ acts as a textual shortcut and why providing the real diagram can even hurt performance.

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

Because trained model checkpoints and image-paired corpora are large in size, all models and datasets are hosted on Hugging Face ([https://huggingface.co/RTL-Series](https://huggingface.co/RTL-Series)):

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
│   ├── ground_truth.json         # Verified reference implementations
│   ├── testbench.jsonl           # Simulation testbenches
│   ├── images_normal/            # Normal netlistsvg schematics
│   ├── images_anony/             # Anonymized netlistsvg schematics
│   └── white.jpg                 # Blank image for Mirage evaluation
├── all_results/                  # Model outputs on Normal, Anony, and Mismatch
├── training/                     # SFT, ORPO, and D-ORPO training scripts
│   ├── train_sft.py
│   ├── train_mixed_sft.py
│   ├── train_orpo.py
│   ├── train_dorpo.py
│   ├── orpo_vision_trainer.py
│   └── orpo_vision_collator.py
├── anony_utils.py                # Identifier anonymization utilities
├── functional_correctness.py     # Icarus Verilog compilation & simulation
├── stat_test.py                  # Paired McNemar significance tests
├── requirements.txt              # Pinned Python dependencies
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

---

## 🧪 Environment & Evaluation

**1. Install dependencies**

```bash
pip install -r requirements.txt
# Install Icarus Verilog (iverilog & vvp) for functional simulation:
# https://github.com/steveicarus/iverilog
```

**2. Run API evaluation**

```bash
python api_normal_original.py    # Normal, real circuit diagrams
python api_normal_mirage.py      # Normal, blank image (Mirage)
python api_anony_original.py     # Anony, anonymized diagrams
python api_anony_mirage.py       # Anony, blank image (Mirage)
```

**3. Run local evaluation ([vLLM](https://github.com/vllm-project/vllm))**

```bash
python local_normal_original.py
python local_normal_mirage.py
python local_anony_original.py
python local_anony_mirage.py
python local_mismatch.py         # Mismatch refusal evaluation
```

**4. Compute Pass@$k$ and statistical tests**

```bash
python functional_correctness.py --solutions <path_to_results.json>
python stat_test.py
```

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
