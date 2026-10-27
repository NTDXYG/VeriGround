import re
from typing import Dict, List, Optional, Tuple


_IDENT_RE = re.compile(r"[A-Za-z_]\w*")
_DECL_HEAD_RE = re.compile(
    r"^(?:input|output|inout|wire|reg|logic|signed|unsigned|const|var)\b"
)


def _strip_comments(code: str) -> str:
    """Remove // and /* */ comments for safer text analysis."""
    no_block = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    no_line = re.sub(r"//.*?$", "", no_block, flags=re.M)
    return no_line


def _extract_module_name(code: str) -> Optional[str]:
    """
    Best-effort module name extraction.
    Works even when code is partially broken, as long as 'module <id>' exists.
    """
    cleaned = _strip_comments(code)
    m = re.search(r"\bmodule\s+([A-Za-z_]\w*)\b", cleaned)
    return m.group(1) if m else None


def _extract_header_port_block(code: str) -> str:
    """
    Best-effort extraction of module port block (...).
    Returns empty string when not found or heavily malformed.
    """
    cleaned = _strip_comments(code)
    m = re.search(r"\bmodule\b", cleaned)
    if not m:
        return ""

    start = cleaned.find("(", m.end())
    if start == -1:
        return ""

    # Parenthesis matching with tolerance for malformed input.
    depth = 0
    end = -1
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
        elif ch == ";":
            # Encountered ; before matching ')': likely malformed header.
            if depth <= 0:
                break

    if end == -1:
        # Fall back to text before first ';' when ')' is missing.
        semi = cleaned.find(";", start)
        if semi == -1:
            return cleaned[start + 1 :]
        return cleaned[start + 1 : semi]

    return cleaned[start + 1 : end]


def _extract_names_from_header_ports(ports_block: str) -> List[str]:
    """
    Extract variable names from header block by comma-separated declarations.
    Handles cases like:
      input [3:0] a, b
      output logic gt, eq, lt
      inout wire signed [7:0] bus
    """
    names: List[str] = []

    for raw in ports_block.split(","):
        item = raw.strip()
        if not item:
            continue

        # Remove leading declaration keywords repeatedly.
        while True:
            new_item = re.sub(_DECL_HEAD_RE, "", item).strip()
            if new_item == item:
                break
            item = new_item

        # Remove leading packed dimensions repeatedly: [7:0][1:0] name
        while True:
            new_item = re.sub(r"^\[[^\]]+\]\s*", "", item)
            if new_item == item:
                break
            item = new_item

        # Keep text before assignment, if any: "name = 1'b0"
        item = item.split("=", 1)[0].strip()

        m = _IDENT_RE.match(item)
        if m:
            names.append(m.group(0))

    # De-duplicate while preserving order.
    dedup: List[str] = []
    seen = set()
    for name in names:
        if name not in seen:
            dedup.append(name)
            seen.add(name)
    return dedup


def build_rename_map(code: str) -> Dict[str, str]:
    """
    Build normalization map:
      <real_module_name> -> module_name
      <var_name_i> -> val_i
    """
    module_name = _extract_module_name(code)
    ports_block = _extract_header_port_block(code)
    var_names = _extract_names_from_header_ports(ports_block)

    rename_map: Dict[str, str] = {}
    if module_name:
        rename_map[module_name] = "module_name"
    for i, name in enumerate(var_names):
        rename_map[name] = f"val_{i}"
    return rename_map


def normalize_verilog(code: str, rename_map: Dict[str, str]) -> str:
    """Apply safe whole-identifier replacements using word boundaries."""
    normalized = code

    # Replace longer identifiers first to reduce accidental overlaps.
    for src in sorted(rename_map.keys(), key=len, reverse=True):
        dst = rename_map[src]
        normalized = re.sub(rf"\b{re.escape(src)}\b", dst, normalized)
    return normalized


def restore_verilog(normalized_code: str, rename_map: Dict[str, str]) -> str:
    """Restore code from normalized names using inverse mapping."""
    inverse_map = {v: k for k, v in rename_map.items()}
    restored = normalized_code
    for src in sorted(inverse_map.keys(), key=len, reverse=True):
        dst = inverse_map[src]
        restored = re.sub(rf"\b{re.escape(src)}\b", dst, restored)
    return restored


def analyze_and_normalize(code: str) -> Tuple[Dict[str, str], str]:
    """Convenience function for one-shot analysis + normalization."""
    rename_map = build_rename_map(code)
    normalized = normalize_verilog(code, rename_map)
    return rename_map, normalized


def anonymize_verilog(code: str) -> Tuple[str, Dict[str, str]]:
    """Alias returning (anonymized_code, rename_map) as documented in README."""
    rename_map, normalized = analyze_and_normalize(code)
    return normalized, rename_map


if __name__ == "__main__":
    # Example with valid header.
    verilog = r"""
module shifter(val_0, val_1, val_2);
    input val_0;
    input val_2;
    output [7:0] val_1;
    reg [7:0] val_1;
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n)
            val_1 <= 8'b0;
        else if (val_0)
            val_1 <= {val_1[6:0], val_2};
        else
            val_1 <= {val_2, val_1[6:0]};
    end
endmodule
"""
    rename_map, normalized = analyze_and_normalize(verilog)
    bad_norm = r"""
module sad(val_0, val_1, val_2);
    input val_0;
    input val_2;
    output [7:0] val_1;
    reg [7:0] val_1;
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n)
            val_1 <= 8'b0;
        else if (val_0)
            val_1 <= {val_1[6:0], val_2};
        else
            val_1 <= {val_2, val_1[6:0]};
    end
endmodule
"""
    restored = restore_verilog(normalized, rename_map)
    print("rename_map:", rename_map)
    print("normalized:\n", normalized)
    print("restored:\n", restored)
    bad_restored = restore_verilog(bad_norm, rename_map)
    print("bad_restored:\n", bad_restored)
    print("bad_restored == bad_norm:", bad_restored == bad_norm)