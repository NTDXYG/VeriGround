"""
McNemar's test for Functional Pass@1 on Original inputs.

Compares our model (4B-dorpo-dt8-dw2.0) against four baseline models:
  MiMo-v2-omni, GPT-4o, GPT-5.4, Opus-4.6

for both Normal and Anony conditions.

Multiple-testing correction: Holm–Bonferroni across all 8 comparisons.

Results:
Our model (Normal): 77/167 pass  (0.4611)
Our model (Anony): 71/167 pass  (0.4251)

======================================================================
  McNemar's Test Results (Holm–Bonferroni corrected, α=0.05)
======================================================================
Comparison               Condition       b       c     p-value    Rejected
----------------------------------------------------------------------
ours vs MiMo-v2-omni        Normal      32      18      0.0660          no
ours vs MiMo-v2-omni         Anony      48       9      0.0000         YES
ours vs GPT-4o              Normal      23      12      0.0910          no
ours vs GPT-4o               Anony      50       6      0.0000         YES
ours vs GPT-5.4             Normal      22      21      1.0000          no
ours vs GPT-5.4              Anony      42      12      0.0001         YES
ours vs Opus-4.6            Normal      22      33      0.1775          no
ours vs Opus-4.6             Anony      56       4      0.0000         YES

======================================================================
  LaTeX table rows
======================================================================
\tool vs.\ MiMo-v2-omni    & n.s. & 0.000$^{**}$ \\
\tool vs.\ GPT-4o          & n.s. & 0.000$^{**}$ \\
\tool vs.\ GPT-5.4         & n.s. & 0.000$^{**}$ \\
\tool vs.\ Opus-4.6        & n.s. & 0.000$^{**}$ \\

======================================================================
  Pass@1 Summary
======================================================================
  Model                        Normal     Anony
  ---------------------------------------------
  ours (4B-dorpo-dt8-dw2.0)    0.4611    0.4251
  MiMo-v2-omni                 0.3772    0.1916
  GPT-4o                       0.3952    0.1617
  GPT-5.4                      0.4551    0.2455
  Opus-4.6                     0.5269    0.1138

Normal 条件（真实电路图）
所有比较均为 n.s.（不显著），说明：
我们的模型在 Normal 条件下的 pass@1（0.461）与各 baseline 没有统计显著差异
特别是 vs Opus-4.6（0.527），虽然我们数值更低，但差异未达显著（p=0.178）
vs GPT-5.4（0.455）几乎持平（b=22, c=21，几乎一半一半），p=1.0
结论：在有图的标准任务上，我们 4B 的小模型与这些大模型（包括 GPT-5.4、Opus-4.6 等闭源强模型）没有显著差距。
Anony 条件（匿名化电路图）
所有比较均 p < 0.01，显著，且 b >> c，说明：
对比	b（我赢）	c（它赢）
vs MiMo-v2-omni	48	9
vs GPT-4o	50	6
vs GPT-5.4	42	12
vs Opus-4.6	56	4
b 远大于 c，说明是我们的模型单方面占优，而非互有胜负
Anony 条件下 baseline 们 pass@1 都大幅下滑（MiMo: 0.19, GPT-4o: 0.16, Opus-4.6: 0.11），而我们只从 0.461 降到 0.425，鲁棒性显著更强
结论：匿名化条件是专门考验模型是否真正理解电路语义（而非依赖信号名称记忆），baseline 模型在此条件下大幅退化，而我们的模型保持了性能，差距在统计上高度显著。
总体解读
这个结果非常适合写进论文，核心论点是：
> 我们的方法在标准条件下与 SOTA 闭源大模型不相上下，在匿名化条件下则显著优于所有 baseline（p < 0.01），证明模型学到了真正的电路理解能力，而非表面的信号命名偏好。
"""


from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import chi2

os.chdir(os.path.dirname(os.path.abspath(__file__)))


# ── data loading ─────────────────────────────────────────────────────────────

def load_pass_vector(path: Path) -> Dict[str, int]:
    """Return {module_name: 1/0} for a pass1 result file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    vec: Dict[str, int] = {}
    for entry in data:
        name = entry.get("module_name", "")
        sols = entry.get("solutions", [])
        if not name or not sols:
            continue
        vec[name] = 1 if sols[0].get("pass") == "true" else 0
    return vec


def align(a: Dict[str, int], b: Dict[str, int]) -> Tuple[List[int], List[int]]:
    """Return paired lists over the intersection of module names."""
    keys = sorted(set(a) & set(b))
    return [a[k] for k in keys], [b[k] for k in keys]


# ── McNemar's test (with continuity correction) ───────────────────────────────

def mcnemar_test(y_a: List[int], y_b: List[int]) -> Tuple[float, float, int, int]:
    """
    McNemar's test between two binary vectors.

    Returns (statistic, p_value, b, c) where
      b = #(A=1, B=0),  c = #(A=0, B=1).
    Uses mid-p correction when b+c <= 25, otherwise standard continuity correction.
    """
    arr_a = np.array(y_a)
    arr_b = np.array(y_b)
    b = int(((arr_a == 1) & (arr_b == 0)).sum())   # model wins
    c = int(((arr_a == 0) & (arr_b == 1)).sum())   # baseline wins

    if b + c == 0:
        return 0.0, 1.0, b, c

    # Mid-p McNemar for small discordant counts, standard otherwise
    if b + c <= 25:
        from scipy.stats import binom
        # exact mid-p: P(X >= b | p=0.5) - 0.5 * P(X == b | p=0.5), two-sided
        p_one = binom.sf(b - 1, b + c, 0.5) - 0.5 * binom.pmf(b, b + c, 0.5)
        p_two = min(2 * p_one, 1.0)
        stat = float(chi2.ppf(1 - p_two, df=1)) if p_two < 1 else 0.0
        return stat, p_two, b, c

    # Continuity-corrected chi-squared
    stat = (abs(b - c) - 1.0) ** 2 / (b + c)
    p_val = float(chi2.sf(stat, df=1))
    return stat, p_val, b, c


# ── Holm–Bonferroni correction ────────────────────────────────────────────────

def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Return a boolean list: True if H0 is rejected after Holm–Bonferroni correction.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    rejected = [False] * m
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if p_values[idx] <= threshold:
            rejected[idx] = True
        else:
            break   # Holm: stop at first non-rejection
    return rejected


def significance_label(p: float, rejected: bool) -> str:
    if not rejected:
        return "n.s."
    if p < 0.01:
        return f"{p:.3f}$^{{**}}$"
    return f"{p:.3f}$^{{*}}$"


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OUR_MODEL = "4B-dorpo-dt8-dw2.0"
    BASELINES = [
        ("MiMo-v2-omni", "mimo-v2-omni"),
        ("GPT-4o",        "gpt-4o"),
        ("GPT-5.4",       "gpt-5.4"),
        ("Opus-4.6",      "opus-4-6"),
    ]
    CONDITIONS = [
        ("Normal", "all_results/results_normal"),
        ("Anony",  "all_results/results_anony"),
    ]

    # Load our model's vectors
    our: Dict[str, Dict[str, int]] = {}
    for cond_name, cond_dir in CONDITIONS:
        path = Path(cond_dir) / f"pass1_{OUR_MODEL}_original.json"
        our[cond_name] = load_pass_vector(path)
        n = len(our[cond_name])
        n_pass = sum(our[cond_name].values())
        print(f"Our model ({cond_name}): {n_pass}/{n} pass  ({n_pass/n:.4f})")

    # Compute all p-values (8 total: 4 baselines × 2 conditions)
    records: List[Dict] = []
    for label, model_name in BASELINES:
        row: Dict = {"label": label}
        for cond_name, cond_dir in CONDITIONS:
            path = Path(cond_dir) / f"pass1_{model_name}_original.json"
            baseline_vec = load_pass_vector(path)
            ya, yb = align(our[cond_name], baseline_vec)
            stat, p_val, b, c = mcnemar_test(ya, yb)
            n_base = sum(baseline_vec.values())
            row[cond_name] = {"p": p_val, "stat": stat, "b": b, "c": c,
                              "n_base": n_base, "n_total": len(ya)}
        records.append(row)

    # Collect all p-values for Holm–Bonferroni
    all_p: List[float] = []
    all_keys: List[Tuple[int, str]] = []          # (record index, condition)
    for i, row in enumerate(records):
        for cond_name, _ in CONDITIONS:
            all_p.append(row[cond_name]["p"])
            all_keys.append((i, cond_name))

    rejected = holm_bonferroni(all_p)

    # Attach rejection decision back to records
    for idx, (i, cond_name) in enumerate(all_keys):
        records[i][cond_name]["rejected"] = rejected[idx]

    # ── Print detailed results ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  McNemar's Test Results (Holm–Bonferroni corrected, α=0.05)")
    print("=" * 70)
    fmt = "{:22s}  {:>10}  {:>6}  {:>6}  {:>10}  {:>10}"
    print(fmt.format("Comparison", "Condition", "b", "c", "p-value", "Rejected"))
    print("-" * 70)
    for row in records:
        for cond_name, _ in CONDITIONS:
            r = row[cond_name]
            print(fmt.format(
                f"ours vs {row['label']}",
                cond_name,
                str(r["b"]),
                str(r["c"]),
                f"{r['p']:.4f}",
                "YES" if r["rejected"] else "no",
            ))

    # ── LaTeX table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  LaTeX table rows")
    print("=" * 70)
    for row in records:
        n_cell = significance_label(row["Normal"]["p"], row["Normal"]["rejected"])
        a_cell = significance_label(row["Anony"]["p"],  row["Anony"]["rejected"])
        print(f"\\tool vs.\\ {row['label']:15s} & {n_cell} & {a_cell} \\\\")

    # ── Baseline pass@1 summary ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Pass@1 Summary")
    print("=" * 70)
    print(f"  {'Model':25s}  {'Normal':>8}  {'Anony':>8}")
    print("  " + "-" * 45)
    for cond_name, cond_dir in CONDITIONS:
        pass  # printed per model below
    all_models = [(f"ours ({OUR_MODEL})", OUR_MODEL, )] + \
                 [(label, mname) for label, mname in BASELINES]
    for label, mname in all_models:
        row_p: List[str] = []
        for cond_name, cond_dir in CONDITIONS:
            path = Path(cond_dir) / f"pass1_{mname}_original.json"
            vec = load_json_pass(path)
            n = len(vec); s = sum(vec.values())
            row_p.append(f"{s/n:.4f}" if n else "N/A")
        print(f"  {label:25s}  {row_p[0]:>8}  {row_p[1]:>8}")


def load_json_pass(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {}
    return load_pass_vector(path)


if __name__ == "__main__":
    main()
