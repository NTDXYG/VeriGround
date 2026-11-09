"""Prepare fixed-header combinational RTL pairs for later Linux rendering.

Run: python code/build_topology_pairs.py
Outputs are new JSON files in code/benchmark; existing files are never overwritten.
Only explicit two-input scalar gates or scalar binary &, |, ^ expressions are
mutated. Selection never reads model evaluation results. Rendering and synthesized
netlist topology remain to be checked on Linux.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path


IDENT = r"[A-Za-z_][A-Za-z_0-9$]*"
TOKEN = re.compile(
    r"\d+'[sS]?[bBoOdDhH][0-9a-fA-F_xXzZ?]+|'[01xXzZ]|\d+|"
    + IDENT + r"|===|!==|>>>|<<<|==|!=|<=|>=|&&|\|\||<<|>>|\^~|~\^|~&|~\||[^\s]"
)
PRECEDENCE = {
    '||': 1, '&&': 2, '|': 3, '^': 4, '^~': 4, '~^': 4, '&': 5,
    '==': 6, '!=': 6, '===': 6, '!==': 6,
    '<': 7, '>': 7, '<=': 7, '>=': 7, '<<': 8, '>>': 8,
    '+': 9, '-': 9, '*': 10, '/': 10, '%': 10,
}
GATE_MUTATIONS = {'and': 'or', 'or': 'and', 'xor': 'xnor', 'xnor': 'xor', 'nand': 'nor', 'nor': 'nand'}
OP_MUTATIONS = {'&': '|', '|': '&', '^': '~^', '~^': '^', '^~': '^'}
CATEGORY = 'Basic Combinational Logic'


class Unsupported(ValueError):
    """The reference lies outside this experiment's conservative RTL subset."""


def strip_comments(code: str) -> str:
    return re.sub(r'/\*.*?\*/|//[^\n]*', '', code, flags=re.S)


def unique_records(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for row in rows:
        name = row['module_name']
        if name in result:
            raise ValueError(f'Duplicate module_name: {name}')
        result[name] = row
    return result


def interface(code: str) -> tuple[str, str, list[dict], dict[str, int]]:
    """Accept one module, ANSI ports, and literal packed ranges; reject ambiguity."""
    if len(re.findall(r'\bmodule\b', code)) != 1 or len(re.findall(r'\bendmodule\b', code)) != 1:
        raise Unsupported('multiple_modules')
    match = re.match(r'\s*module\s+(' + IDENT + r')\s*\((.*?)\)\s*;', code, re.S)
    if not match:
        raise Unsupported('non_ansi_or_parameterized_header')
    ports = []
    direction = None
    packed = ''
    signed = False
    width = 1
    for segment in match[2].split(','):
        segment = segment.strip()
        start = re.match(r'(input|output|inout)\b', segment)
        if start:
            direction = start[1]
            segment = segment[start.end():].strip()
            segment = re.sub(r'^(?:wire|reg|logic)\s+', '', segment)
            signed = bool(re.match(r'signed\b', segment))
            segment = re.sub(r'^(?:signed|unsigned)\s+', '', segment)
            dim = re.match(r'\[\s*(\d+)\s*:\s*(\d+)\s*\]', segment)
            packed, width = '', 1
            if dim:
                packed = dim[0]
                width = abs(int(dim[1]) - int(dim[2])) + 1
                segment = segment[dim.end():].strip()
        if direction not in ('input', 'output') or not re.fullmatch(IDENT, segment):
            raise Unsupported('unsupported_port_declaration')
        ports.append({'name': segment, 'direction': direction, 'width': width,
                      'packed_range': packed, 'signed': signed})
    if len({p['name'] for p in ports}) != len(ports):
        raise Unsupported('duplicate_ports')
    widths = {p['name']: p['width'] for p in ports}
    body = code[match.end():]
    for decl in re.finditer(r'\b(?:wire|reg|logic)\s+(?:signed\s+)?(\[\s*\d+\s*:\s*\d+\s*\])?\s*([^;]+);', body):
        dim = decl[1]
        w = abs(int(re.findall(r'\d+', dim)[0]) - int(re.findall(r'\d+', dim)[1])) + 1 if dim else 1
        for item in decl[2].split(','):
            name = item.split('=', 1)[0].strip()
            if not re.fullmatch(IDENT, name):
                raise Unsupported('unsupported_internal_declaration')
            widths[name] = w
    return match[1], match[0].strip(), ports, widths


class Expression:
    """Pratt traversal identifies binary nodes and their self-determined widths.

    Mutations add parentheses around the changed node so a replacement cannot
    accidentally regroup adjacent operators of a different precedence.
    """

    def __init__(self, text: str, widths: dict[str, int], offset: int):
        self.tokens = list(TOKEN.finditer(text))
        self.widths = widths
        self.offset = offset
        self.i = 0
        self.candidates = []

    def peek(self) -> str:
        return self.tokens[self.i][0] if self.i < len(self.tokens) else ''

    def take(self, expected: str | None = None):
        if self.i >= len(self.tokens):
            raise Unsupported('unsupported_expression')
        token = self.tokens[self.i]
        self.i += 1
        if expected is not None and token[0] != expected:
            raise Unsupported('unsupported_expression')
        return token

    def parse(self, minimum: int = 0) -> tuple[int, int, int | None]:
        token = self.take()
        value, start = token[0], token.start()
        if value in ('~', '!', '+', '-', '&', '|', '^', '~&', '~|', '~^', '^~'):
            _, end, width = self.parse(11)
            width = 1 if value in ('!', '&', '|', '^', '~&', '~|', '~^', '^~') else width
        elif value == '(':
            _, _, width = self.parse()
            end = self.take(')').end()
        elif value == '{':
            widths = [self.parse()[2]]
            while self.peek() == ',':
                self.take(',')
                widths.append(self.parse()[2])
            end = self.take('}').end()
            width = sum(widths) if all(w is not None for w in widths) else None
        elif re.fullmatch(IDENT, value):
            width, end = self.widths.get(value), token.end()
        elif re.fullmatch(r"\d+'[sS]?[bBoOdDhH][0-9a-fA-F_xXzZ?]+", value):
            width, end = int(value.split("'")[0]), token.end()
        elif value in ("'0", "'1", "'x", "'X", "'z", "'Z"):
            width, end = 1, token.end()
        elif value.isdigit():
            width, end = max(32, int(value).bit_length()), token.end()
        else:
            raise Unsupported('unsupported_expression')
        if self.peek() == '[':
            self.take('[')
            left = self.take()[0]
            if not left.isdigit():
                raise Unsupported('dynamic_select')
            width = 1
            if self.peek() == ':':
                self.take(':')
                right = self.take()[0]
                if not right.isdigit():
                    raise Unsupported('dynamic_select')
                width = abs(int(left) - int(right)) + 1
            end = self.take(']').end()
        while self.peek() in PRECEDENCE and PRECEDENCE[self.peek()] >= minimum:
            op = self.take()
            _, right_end, right_width = self.parse(PRECEDENCE[op[0]] + 1)
            if op[0] in OP_MUTATIONS and width == right_width == 1:
                self.candidates.append({'kind': 'scalar_binary_operator', 'before': op[0],
                    'after': OP_MUTATIONS[op[0]], 'start': self.offset + start,
                    'end': self.offset + right_end, 'operator_start': self.offset + op.start(),
                    'operator_end': self.offset + op.end()})
            if op[0] in ('==', '!=', '===', '!==', '<', '>', '<=', '>=', '&&', '||'):
                width = 1
            elif op[0] not in ('<<', '>>'):
                width = max(width, right_width) if width is not None and right_width is not None else None
            end = right_end
        if minimum == 0 and self.peek() == '?':
            self.take('?')
            _, _, yes = self.parse()
            self.take(':')
            _, end, no = self.parse()
            width = max(yes, no) if yes is not None and no is not None else None
        return start, end, width


def mutations(code: str, widths: dict[str, int]) -> list[dict]:
    candidates = []
    for match in re.finditer(r'\bassign\s+[^;=]+=(?!=)([^;]+);', code):
        parser = Expression(match[1], widths, match.start(1))
        try:
            parser.parse()
            if parser.i != len(parser.tokens):
                continue
        except Unsupported:
            continue
        candidates.extend(parser.candidates)
    gate_re = r'\b(and|or|xor|xnor|nand|nor)\s+(?:' + IDENT + r'\s*)?\(([^;]+)\)\s*;'
    # Unnamed primitives may have no whitespace before their opening parenthesis.
    gate_re = gate_re.replace(r'\s+(?:', r'\s*(?:', 1)
    for match in re.finditer(gate_re, code):
        args = [v.strip() for v in match[2].split(',')]
        if len(args) != 3:
            continue
        def scalar(v):
            return widths.get(v) == 1 or bool(re.fullmatch(IDENT + r'\[\d+\]', v))
        if all(scalar(v) for v in args):
            candidates.append({'kind': 'two_input_gate', 'before': match[1],
                'after': GATE_MUTATIONS[match[1]], 'start': match.start(1), 'end': match.end(1),
                'operator_start': match.start(1), 'operator_end': match.end(1)})
    return sorted(candidates, key=lambda x: (x['operator_start'], x['kind']))


def mutate(code: str, mutation: dict) -> str:
    a, b = mutation['operator_start'], mutation['operator_end']
    if mutation['kind'] == 'two_input_gate':
        return code[:a] + mutation['after'] + code[b:]
    start, end = mutation['start'], mutation['end']
    expression = code[start:a] + mutation['after'] + code[b:end]
    return code[:start] + '(' + expression + ')' + code[end:]


def anonymize(code: str, ports: list[dict], module: str, widths: dict[str, int]):
    rename = {module: 'module_name'}
    rename.update({p['name']: f'val_{i}' for i, p in enumerate(ports)})
    for name in widths:
        if name not in rename:
            rename[name] = f'net_{len([n for n in rename if n not in {module, *(p["name"] for p in ports)}])}'
    # Replace identifiers simultaneously: original names may already be val_0/net_0.
    pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in sorted(rename, key=len, reverse=True)) + r')\b')
    return pattern.sub(lambda m: rename[m[0]], code), rename


def run_process(command: list[str], cwd: Path, timeout: int = 20):
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                          encoding='utf-8', errors='replace', timeout=timeout)


def simulate(code: str, tb: str, top: str, tools: dict, timeout: int = 20) -> str:
    with tempfile.TemporaryDirectory(prefix='topology_') as temp:
        work = Path(temp)
        (work / 'design.v').write_text(code, encoding='utf-8')
        (work / 'tb.v').write_text(tb, encoding='utf-8')
        cp = run_process([tools['iverilog'], '-g2012', '-s', top, '-o', str(work / 'sim.vvp'),
                          str(work / 'design.v'), str(work / 'tb.v')], work, timeout)
        if cp.returncode:
            raise Unsupported('compile_failed: ' + cp.stderr[:300])
        sp = run_process([tools['vvp'], '-n', str(work / 'sim.vvp')], work, timeout)
        if sp.returncode:
            raise Unsupported('simulation_failed: ' + sp.stderr[:300])
        return sp.stdout


def source_check(code: str, problem: dict, tools: dict) -> None:
    tb = problem['testbench']
    top = 'tb' if problem['source'] == 'verilogeval_v2' else re.search(r'\bmodule\s+(' + IDENT + ')', strip_comments(tb))[1]
    output = simulate(code, tb, top, tools)
    mismatches = re.search(r'Mismatches:\s*(\d+)\s+in\s+(\d+)\s+samples', output)
    if mismatches:
        ok = int(mismatches[1]) == 0 and int(mismatches[2]) > 0
    else:
        failures = re.search(r'\[FAIL\]|\bERROR\b|\bTIMEOUT\b|\bTEST FAILED\b|Some tests failed', output, re.I)
        failed_json = re.search(r'"failed"\s*:\s*(\d+)', output)
        success = re.search(r'All tests passed|Your Design Passed|\[PASS\]', output, re.I)
        ok = not failures and ((failed_json is not None and int(failed_json[1]) == 0) or success is not None)
    if not ok:
        raise Unsupported('source_testbench_failed_or_no_success_marker')


def vector_tb(module: str, ports: list[dict]) -> str:
    inputs = [p for p in ports if p['direction'] == 'input']
    outputs = [p for p in ports if p['direction'] == 'output']
    declarations = [f"  {'reg' if p['direction'] == 'input' else 'wire'} {p['packed_range']} {p['name']};" for p in ports]
    connections = ', '.join(f".{p['name']}({p['name']})" for p in ports)
    ins = ', '.join(p['name'] for p in inputs)
    outs = ', '.join(p['name'] for p in outputs)
    count = 1 << sum(p['width'] for p in inputs)
    return '\n'.join(['`timescale 1ns/1ps', 'module tb;', *declarations,
        f'  {module} dut ({connections});', '  integer i;', '  initial begin',
        f'    for (i=0; i<{count}; i=i+1) begin', f'      {{{ins}}} = i;',
        '      #10;', f'      $display("VECTOR %0d %b", i, {{{outs}}});',
        '    end', '    $finish;', '  end', 'endmodule', ''])


def truth_table(code: str, module: str, ports: list[dict], tools: dict) -> list[str]:
    output = simulate(code, vector_tb(module, ports), 'tb', tools)
    vectors = re.findall(r'^VECTOR\s+(\d+)\s+([01xXzZ]+)\s*$', output, re.M)
    count = 1 << sum(p['width'] for p in ports if p['direction'] == 'input')
    width = sum(p['width'] for p in ports if p['direction'] == 'output')
    if len(vectors) != count or [int(i) for i, _ in vectors] != list(range(count)):
        raise Unsupported('incomplete_truth_table')
    if any(len(bits) != width or re.search('[^01]', bits) for _, bits in vectors):
        raise Unsupported('undefined_output')
    return [bits for _, bits in vectors]


def evaluation_tb(ports: list[dict], expected: list[str]) -> str:
    """A self-contained oracle table; generated model outputs never see this text."""
    template = vector_tb('module_name', ports)
    width = sum(p['width'] for p in ports if p['direction'] == 'output')
    outs = ', '.join(p['name'] for p in ports if p['direction'] == 'output')
    cases = '\n'.join(f"        {i}: expected = {width}'b{bits};" for i, bits in enumerate(expected))
    template = template.replace('  integer i;', f'  integer i;\n  integer errors=0;\n  reg [{width-1}:0] expected;')
    template = template.replace(f'      $display("VECTOR %0d %b", i, {{{outs}}});',
        '      case (i)\n' + cases + "\n      endcase\n" +
        f'      if ({{{outs}}} !== expected) errors=errors+1;')
    template = template.replace('    $finish;',
        f'    $display("Mismatches: %0d in {len(expected)} samples", errors);\n' +
        '    if (errors==0) $display("All tests passed");\n' +
        '    else begin $display("Test failed"); $fatal(1, "Functional mismatch"); end\n' +
        '    $finish;')
    return template


def build(args) -> dict:
    tools = {name: shutil.which(name) for name in ('iverilog', 'vvp')}
    if not all(tools.values()):
        raise SystemExit('iverilog and vvp must be on PATH; no files have been written.')
    benchmark = args.benchmark.resolve()
    names = ['topology_pairs.json', 'problems_topology_original.json',
             'problems_topology_mutant.json', 'topology_sampling_report.json']
    if any((benchmark / name).exists() for name in names):
        raise SystemExit('An output file already exists; choose a new benchmark/output directory before rebuilding.')
    problem_file, reference_file = benchmark / 'problems.json', benchmark / 'ground_truth.json'
    problems = unique_records(json.loads(problem_file.read_text(encoding='utf-8')))
    references = unique_records(json.loads(reference_file.read_text(encoding='utf-8')))
    seed = random.Random(args.seed)
    eligible, excluded = [], []
    for name in sorted(problems):
        problem = problems[name]
        if problem['level'] != CATEGORY:
            continue
        try:
            reference = references[name]
            if len(reference['solutions']) != 1:
                raise Unsupported('multiple_reference_solutions')
            code = strip_comments(reference['solutions'][0]['solution']).strip() + '\n'
            # Exclude procedural blocks, hierarchy, arrays and timing from this minimal study.
            if re.search(r'\b(always|always_comb|always_ff|always_latch|initial|generate|function|task|parameter|localparam)\b|`|#|\$', code):
                raise Unsupported('outside_continuous_combinational_subset')
            module, header, ports, widths = interface(code)
            input_bits = sum(p['width'] for p in ports if p['direction'] == 'input')
            if not 1 <= input_bits <= args.max_input_bits or not any(p['direction'] == 'output' for p in ports):
                raise Unsupported('input_budget_or_no_output')
            candidates = mutations(code, widths)
            if not candidates:
                raise Unsupported('no_supported_scalar_gate')
            source_check(code, problem, tools)
            original_truth = truth_table(code, module, ports, tools)
            anon, rename = anonymize(code, ports, module, widths)
            _, anon_header, anon_ports, _ = interface(anon)
            if truth_table(anon, 'module_name', anon_ports, tools) != original_truth:
                raise Unsupported('anonymization_changed_function')
            # A per-module PRNG keeps candidate choice independent of other exclusions.
            rng = random.Random(f'{args.seed}:{name}')
            rng.shuffle(candidates)
            chosen = None
            for candidate in candidates:
                changed = mutate(code, candidate)
                try:
                    mutant_truth = truth_table(changed, module, ports, tools)
                except (Unsupported, subprocess.TimeoutExpired):
                    continue
                witnesses = [i for i, (a, b) in enumerate(zip(original_truth, mutant_truth)) if a != b]
                if not witnesses:
                    continue
                mutated_anon, mutated_map = anonymize(changed, ports, module, widths)
                if mutated_map != rename or interface(mutated_anon)[1] != anon_header:
                    raise Unsupported('mutated_interface_changed')
                if truth_table(mutated_anon, 'module_name', anon_ports, tools) != mutant_truth:
                    raise Unsupported('mutant_anonymization_changed_function')
                chosen = (candidate, mutated_anon, mutant_truth, witnesses)
                break
            if chosen is None:
                raise Unsupported('no_functionally_distinct_supported_mutation')
            mutation, mutant, mutant_truth, witnesses = chosen
            pair_id = f'topology_{len(eligible):03d}'
            original_tb = evaluation_tb(anon_ports, original_truth)
            mutant_tb = evaluation_tb(anon_ports, mutant_truth)
            for design, tb in [(anon, original_tb), (mutant, mutant_tb)]:
                if 'All tests passed' not in simulate(design, tb, 'tb', tools):
                    raise Unsupported('generated_testbench_failed')
            record = {'pair_id': pair_id, 'module_name': name, 'source': problem['source'],
                'level': CATEGORY, 'module_header': anon_header, 'rename_map': rename,
                'ports': [{**p, 'original_name': ports[i]['name']} for i, p in enumerate(anon_ports)],
                'input_bits': input_bits, 'mutation': {**mutation,
                    'location_basis': 'comment-stripped original reference RTL; zero-based character offsets',
                    'line': code[:mutation['operator_start']].count('\n') + 1,
                    'original_fragment': code[mutation['start']:mutation['end']]},
                'reference_rtl': code,
                'original': {'rtl': anon, 'image': f'images_topology_original/{pair_id}.jpg',
                             'testbench': original_tb},
                'mutant': {'rtl': mutant, 'image': f'images_topology_mutant/{pair_id}.jpg',
                           'testbench': mutant_tb},
                'verification': {'source_testbench': 'passed', 'compilation': 'passed',
                    'anonymization_equivalence': 'passed', 'functional_difference': 'exhaustively_verified',
                    'input_domain': 'all binary input combinations; no X/Z inputs',
                    'vectors_checked': len(original_truth), 'distinguishing_vector_count': len(witnesses),
                    'witnesses': [{'packed_input': format(i, f'0{input_bits}b'),
                                   'original_output': original_truth[i], 'mutant_output': mutant_truth[i]}
                                  for i in witnesses[:16]],
                    'packing_order': 'ports in header order, concatenated most-significant first',
                    'original_truth_table': original_truth, 'mutant_truth_table': mutant_truth,
                    'synthesis': 'pending_linux', 'single_gate_after_synthesis': 'pending_linux',
                    'rendering': 'pending_linux', 'visual_token_budget': 'pending_rendering'}}
            eligible.append(record)
            print(f'VALID {name}: {input_bits} input bits, {mutation["before"]}->{mutation["after"]}, {len(witnesses)} differing vectors', flush=True)
        except (Unsupported, subprocess.TimeoutExpired) as exc:
            excluded.append({'module_name': name, 'reason': str(exc)})
    if not eligible:
        raise SystemExit('No validated pairs; no output files written.')
    selected = seed.sample(eligible, min(args.count, len(eligible)))
    selected.sort(key=lambda p: p['module_name'])
    # Neutral filenames expose neither the original function nor the mutation type.
    for i, pair in enumerate(selected):
        pair['pair_id'] = f'topology_{i:03d}'
        for variant in ('original', 'mutant'):
            pair[variant]['image'] = (problems[pair['module_name']]['image'].replace('images_normal/', 'images_anony/')
                                      if variant == 'original' else
                                      f'images_topology_mutant/{pair["pair_id"]}.jpg')
    inputs = {}
    for variant in ('original', 'mutant'):
        inputs[variant] = [{'pair_id': p['pair_id'], 'module_name': p['pair_id'],
                           'module_header': p['module_header'],
                           'prompt': 'Implement the circuit shown in the provided diagram.',
                           'source': p['source'], 'level': p['level'], 'image': p[variant]['image']}
                          for p in selected]
    report = {'schema_version': 1, 'seed': args.seed, 'requested_pairs': args.count,
              'max_input_bits': args.max_input_bits, 'category': CATEGORY,
              'candidate_scope': 'single-module continuous combinational RTL; one scalar binary operator or two-input primitive gate changed',
              'selection': 'uniform seeded sample of all validated eligible modules; one mutant per module',
              'model_results_used': False, 'eligible_count': len(eligible), 'selected_count': len(selected),
              'eligible_modules': [p['module_name'] for p in eligible],
              'selected_modules': [p['module_name'] for p in selected], 'excluded': excluded,
              'exclusion_counts': dict(Counter(x['reason'].split(':')[0] for x in excluded)),
              'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (problem_file, reference_file)},
              'rendering': 'Original images reuse images_anony; render mutants on Linux and check diagram/reference correspondence, visibility and image token budget'}
    payloads = [selected, inputs['original'], inputs['mutant'], report]
    encoded = [json.dumps(p, ensure_ascii=False, indent=2) + '\n' for p in payloads]
    for name, content in zip(names, encoded):
        with (benchmark / name).open('x', encoding='utf-8', newline='\n') as handle:
            handle.write(content)
    print(json.dumps({'eligible_pairs': len(eligible), 'selected_pairs': len(selected),
                      'files': names, 'excluded': report['exclusion_counts']}, ensure_ascii=False), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, default=Path(__file__).parent / 'benchmark')
    parser.add_argument('--count', type=int, default=30)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--max-input-bits', type=int, default=12)
    args = parser.parse_args()
    if args.count < 1 or not 1 <= args.max_input_bits <= 16:
        parser.error('count must be positive and max-input-bits must be between 1 and 16')
    build(args)


if __name__ == '__main__':
    main()
