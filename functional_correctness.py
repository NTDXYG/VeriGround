"""iverilog/vvp functional check with source-aware routing by module_name."""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

solutions_file = "results_anony/pass5_4B-dorpo-dt8-dw2.0_mirage.json"

VE, RB, AX, RT = "verilogeval_v2", "resbench", "archxbench", "rtllm_v2"
SOURCE_ALIASES = {"resbench": RB, "verilogeval_v2": VE, "archxbench": AX, "rtllm_v2": RT}
IVERILOG = ["iverilog", "-Wall", "-Winfloop", "-Wno-timescale", "-g2012"]


def norm_source(raw: object) -> str | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    return SOURCE_ALIASES.get(raw.strip().lower().replace("-", "_"))


def tb_top_name(source: str, tb_code: str) -> str | None:
    if source == VE:
        return "tb"
    for line in tb_code.splitlines():
        m = re.search(r"\s*module\s+(\w+)\s*[\(;]", line.strip())
        if m:
            return m.group(1)
    return None


def pass_at_k(n: int, c: int, k: int) -> float:
    if n == 0 or c == 0:
        return 0.0
    if k >= n:
        return 1.0 if c > 0 else 0.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                raise ValueError(f"bad JSON line {i} in {path}") from None
    return out


def merged_tb_map(rows: list[dict]) -> dict[str, dict[str, str | None]]:
    m: dict[str, dict[str, str | None]] = {}
    for p in rows:
        mn, tb = p.get("module_name"), p.get("testbench")
        src = norm_source(p.get("source"))
        if mn and tb and src:
            m[mn] = {
                "testbench": tb,
                "source": src,
            }
    return m


def load_level_map(path: Path) -> dict[str, str]:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}
    out: dict[str, str] = {}
    if isinstance(rows, list):
        for r in rows:
            if not isinstance(r, dict):
                continue
            name = r.get("module_name")
            level = r.get("level")
            if isinstance(name, str) and name and isinstance(level, str) and level:
                out[name] = level
    return out


def parse_out(source: str, sim_returncode: int, stdout: str, stderr: str) -> tuple[bool, str]:
    if source == VE:
        mm = re.search(r"Mismatches: ([0-9]*) in ([0-9]*) samples", stdout)
        if mm:
            mis, tot = (int(x) for x in mm.groups())
            return (True, "true") if mis == 0 else (False, f"Test failed: {mis} mismatch(es) in {tot} sample(s)")
        if "Timeout" in stdout or "Timeout" in stderr:
            return False, "Test failed: simulation timeout"
        if stderr.strip():
            return False, f"Simulation error: {stderr.strip()}"
        return False, "Test failed: could not parse simulation output"
    if source == AX:
        merged = f"{stdout}\n{stderr}".lower()
        fail_markers = [
            "[fail]",
            "some tests failed",
            "test failed",
            "mismatch",
            "mismatches found",
            "compilation failed",
            "simulation error",
            "timeout",
        ]
        pass_markers = [
            "[pass]",
            "all tests passed",
            "your design passed",
            "pass - all cycles matched",
        ]
        if sim_returncode != 0:
            return False, f"Simulation failed (returncode={sim_returncode})"
        if any(marker in merged for marker in fail_markers):
            return False, "Testbench reported failure"
        if any(marker in merged for marker in pass_markers):
            return True, "true"
        return True, "true"
    if "All tests passed" in stdout or "Your Design Passed" in stdout:
        return True, "true"
    if stderr:
        return False, f"Simulation error: {stderr}"
    return False, "Test failed: tests did not pass"


def run(
    sol_path: Path,
    merged_probs: Path,
    level_json: Path,
    timeout: int = 5,
) -> None:
    try:
        data: list[dict] = json.loads(sol_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"Error: not found: {sol_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: invalid JSON: {sol_path}")
        return

    try:
        merged_map = merged_tb_map(load_jsonl(merged_probs))
    except FileNotFoundError as e:
        print(f"Error: not found: {e.filename}")
        return

    level_map = load_level_map(level_json)
    module_stats = defaultdict(lambda: {"total": 0, "compiled": 0, "passed": 0, "level": "Unknown"})
    with tempfile.TemporaryDirectory(prefix="fc_iverilog_") as temp_root_str:
        temp_root = Path(temp_root_str)
        work_dirs = {VE: temp_root / VE, RB: temp_root / RB, AX: temp_root / AX, RT: temp_root / RT}
        for d in work_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        for entry in tqdm(data, desc="sim"):
            name = entry.get("module_name")
            if not name:
                print("Warning: missing module_name, skip")
                continue
            info = merged_map.get(name)
            if info is None:
                print(f"Warning: no testbench for {name}, skip")
                continue
            src = info.get("source")
            if not isinstance(src, str):
                print(f"Warning: bad source in merged problems for {name}, skip")
                continue
            tb = str(info["testbench"])
            sols = entry.get("solutions") or []
            if not sols:
                continue

            work = work_dirs[src]
            tv, ttb, vvp = work / "temp.v", work / "testbench.v", work / "test.vvp"
            st = module_stats[name]
            st["level"] = level_map.get(name, "Unknown")

            for s in sols:
                st["total"] += 1
                code = s.get("solution", "")
                if src == VE and not code:
                    s["pass"] = "Error: empty solution"
                    continue
                try:
                    tv.write_text(code, encoding="utf-8")
                    ttb.write_text(tb, encoding="utf-8")
                except OSError as e:
                    s["pass"] = f"File write error: {e}"
                    continue

                top = tb_top_name(src, tb)
                if not top:
                    s["pass"] = "Error: could not extract testbench module."
                    continue

                cp = subprocess.run(
                    [*IVERILOG, "-s", top, "-o", str(vvp), str(tv), str(ttb)],
                    cwd=str(work),
                    capture_output=True,
                    text=True,
                )
                if cp.returncode != 0:
                    s["pass"] = f"Compilation failed: {cp.stderr.strip()}"
                    continue
                st["compiled"] += 1

                try:
                    sp = subprocess.run(
                        ["vvp", "-n", str(vvp)],
                        cwd=str(work),
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
                    out, err = sp.stdout, sp.stderr
                except subprocess.TimeoutExpired:
                    out = err = "Timeout"
                except Exception as e:
                    out, err = "Error", str(e)
                    sim_rc = 1
                else:
                    sim_rc = sp.returncode

                ok, msg = parse_out(src, sim_rc, out, err)
                s["pass"] = msg
                if ok:
                    st["passed"] += 1

    try:
        sol_path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")
    except OSError as e:
        print(f"Warning: save failed: {e}")

    _summarize_all(module_stats)
    _summarize_by_level(module_stats)


def _summarize_all(module_stats: dict[str, dict[str, int]]) -> None:
    if not module_stats:
        return

    k = next(iter(module_stats.values()))["total"]
    if k == 0:
        return

    n_mod = 0
    syntax_acc = 0.0
    functional_acc = 0.0
    for _, r in sorted(module_stats.items()):
        n = r["total"]
        if n <= 0:
            continue
        n_mod += 1
        syntax_acc += pass_at_k(n, r["compiled"], k)
        functional_acc += pass_at_k(n, r["passed"], k)

    if n_mod == 0:
        return

    print(f"\n{'=' * 60}")
    print(f"Average syntax pass@{k} ({n_mod} modules): {syntax_acc / n_mod:.4f}")
    print(f"Average functional pass@{k} ({n_mod} modules): {functional_acc / n_mod:.4f}")
    print("=" * 60)


def _summarize_by_level(module_stats: dict[str, dict[str, int]]) -> None:
    grouped = defaultdict(list)
    for name, rec in module_stats.items():
        grouped[str(rec.get("level", "Unknown"))].append((name, rec))

    if not grouped:
        return

    print("\nPer-level pass@k")
    print("=" * 60)
    for level in sorted(grouped.keys()):
        rows = grouped[level]
        k = rows[0][1]["total"] if rows else 0
        if k <= 0:
            continue
        n_mod = 0
        syntax_acc = 0.0
        functional_acc = 0.0
        for _, r in rows:
            n = r["total"]
            if n <= 0:
                continue
            n_mod += 1
            syntax_acc += pass_at_k(n, r["compiled"], k)
            functional_acc += pass_at_k(n, r["passed"], k)
        if n_mod == 0:
            continue
        print(f"{level}:")
        print(f"  syntax pass@{k} ({n_mod} modules): {syntax_acc / n_mod:.4f}")
        print(f"  functional pass@{k} ({n_mod} modules): {functional_acc / n_mod:.4f}")
    print("=" * 60)


def main() -> None:
    
    ap = argparse.ArgumentParser(description="Functional correctness via iverilog/vvp.")
    ap.add_argument("--solutions", default=solutions_file, help="Results JSON path")
    ap.add_argument(
        "--problems",
        default="benchmark/testbench.jsonl"    
    )
    ap.add_argument("--levels-json", default="benchmark/problems.json")
    ap.add_argument("--timeout", type=int, default=5)
    args = ap.parse_args()

    try:
        run(
            Path(args.solutions),
            Path(args.problems),
            Path(args.levels_json),
            args.timeout,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception:
        raise


if __name__ == "__main__":
    main()
