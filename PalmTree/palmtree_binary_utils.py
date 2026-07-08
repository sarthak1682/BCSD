import json
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


OPT_PATTERN = re.compile(r"(?<![A-Za-z0-9])(O[0-3s])(?![A-Za-z0-9])", re.IGNORECASE)
RIP_REL_PATTERN = re.compile(r"\[\s*rip\s*([+-])\s*(0x[0-9a-fA-F]+|\d+)\s*\]")
HEX_TOKEN_PATTERN = re.compile(r"^0x[0-9a-fA-F]+$")
TOKEN_SPLIT_PATTERN = re.compile(r"(0x[0-9a-fA-F]+|[A-Za-z_][A-Za-z0-9_]*|\d+)")

DEFAULT_EXCLUDED_FUNCTIONS: Set[str] = {
    "_start",
    "deregister_tm_clones",
    "register_tm_clones",
    "__do_global_dtors_aux",
    "frame_dummy",
    "__libc_csu_init",
    "__libc_csu_fini",
    "__libc_start_main",
    "__libc_start_call_main",
    "__gmon_start__",
    "_dl_relocate_static_pie",
}


def infer_opt_label(path: Path) -> str:
    match = OPT_PATTERN.search(str(path))
    if not match:
        return "unknown"
    opt = match.group(1)
    return "Os" if opt.lower() == "os" else opt.upper()


def collect_binary_files(path: Path) -> List[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(
            file_path
            for file_path in path.rglob("*")
            if file_path.is_file() and not file_path.is_symlink()
        )
    raise FileNotFoundError(f"No such binary input path: {path}")


def _load_angr():
    try:
        import angr  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "angr is required for PalmTree binary extraction. "
            "Install angr in the environment that runs finetune_palmtree.py / eval_palmtree.py."
        ) from exc
    return angr


def _relative_binary_name(binary_path: Path, input_root: Path) -> str:
    if input_root.is_file():
        return binary_path.name
    try:
        return str(binary_path.relative_to(input_root))
    except ValueError:
        return binary_path.name


def _canonicalize_binary_key(relative_binary: str) -> str:
    path = Path(relative_binary)
    cleaned_parts: List[str] = []

    for part in path.parts[:-1]:
        if part.upper() in {"O0", "O1", "O2", "O3", "OS"}:
            continue
        cleaned_parts.append(part)

    stem = re.sub(r"(?i)(^|[-_.])O[0-3s](?=$|[-_.])", r"\1", path.stem)
    stem = re.sub(r"[-_.]{2,}", "-", stem).strip("-_.")
    filename = (stem or path.stem) + path.suffix
    cleaned_parts.append(filename)

    return str(Path(*cleaned_parts))


def _build_symbol_map(project) -> Dict[int, str]:
    symbol_map: Dict[int, str] = {}
    for symbol in getattr(project.loader.main_object, "symbols", []):
        rebased_addr = getattr(symbol, "rebased_addr", None)
        if rebased_addr is None:
            continue
        symbol_map[int(rebased_addr)] = getattr(symbol, "name", "") or "symbol"
    return symbol_map


def _resolve_special_address(target_addr: int, symbol_map: Dict[int, str], string_addrs: Set[int]) -> str:
    if target_addr in symbol_map:
        return "symbol"
    if target_addr in string_addrs:
        return "string"
    return "address"


def _normalize_operand(
    operand: str,
    addr: int,
    size: int,
    symbol_map: Dict[int, str],
    string_addrs: Set[int],
) -> List[str]:
    operand = operand.replace("ptr", "")
    operand = operand.replace("*", " * ")

    rip_match = RIP_REL_PATTERN.search(operand.lower())
    if rip_match:
        sign, offset_text = rip_match.groups()
        offset = int(offset_text, 16) if offset_text.lower().startswith("0x") else int(offset_text)
        if sign == "-":
            offset = -offset
        target_addr = addr + size + offset
        replacement = _resolve_special_address(target_addr, symbol_map, string_addrs)
        operand = operand[: rip_match.start()] + replacement + operand[rip_match.end() :]

    operand = operand.replace("[", " [ ").replace("]", " ] ")
    raw_tokens = [piece.strip() for piece in TOKEN_SPLIT_PATTERN.split(operand) if piece and piece.strip()]

    normalized: List[str] = []
    for token in raw_tokens:
        if HEX_TOKEN_PATTERN.match(token) and 6 <= len(token) <= 18:
            normalized.append(_resolve_special_address(int(token, 16), symbol_map, string_addrs))
        else:
            normalized.append(token)
    return normalized


def normalize_instruction(
    mnemonic: str,
    op_str: str,
    addr: int,
    size: int,
    symbol_map: Dict[int, str],
    string_addrs: Set[int],
) -> str:
    tokens: List[str] = [mnemonic]
    operands = [part.strip() for part in op_str.split(",") if part.strip()]
    for operand in operands:
        tokens.extend(_normalize_operand(operand, addr, size, symbol_map, string_addrs))
    return " ".join(token for token in tokens if token)


def _capstone_fields(insn) -> Tuple[str, str, int, int]:
    base = getattr(insn, "insn", insn)
    mnemonic = getattr(base, "mnemonic", "")
    op_str = getattr(base, "op_str", "")
    addr = int(getattr(base, "address", getattr(insn, "address", 0)))
    size = int(getattr(base, "size", getattr(insn, "size", 0)))
    return mnemonic, op_str, addr, size


def _should_keep_function(func, excluded_functions: Set[str]) -> bool:
    name = getattr(func, "name", None)
    if not name or name in excluded_functions:
        return False
    if getattr(func, "is_simprocedure", False):
        return False
    if getattr(func, "is_plt", False):
        return False
    return True


def _extract_function_record(
    project,
    binary_path: Path,
    input_root: Path,
    func,
    symbol_map: Dict[int, str],
) -> Optional[Dict[str, object]]:
    try:
        string_addrs = {int(addr) for addr, _value in func.string_references()}
    except Exception:
        string_addrs = set()

    blocks: List[List[str]] = []
    instructions: List[str] = []
    block_iter = sorted(func.blocks, key=lambda block: block.addr)

    for block in block_iter:
        block_instructions: List[str] = []
        try:
            capstone_insns = list(block.capstone.insns)
        except Exception:
            capstone_insns = []

        for insn in capstone_insns:
            mnemonic, op_str, insn_addr, insn_size = _capstone_fields(insn)
            if not mnemonic:
                continue
            normalized = normalize_instruction(
                mnemonic=mnemonic,
                op_str=op_str,
                addr=insn_addr,
                size=insn_size,
                symbol_map=symbol_map,
                string_addrs=string_addrs,
            )
            block_instructions.append(normalized)
            instructions.append(normalized)

        if block_instructions:
            blocks.append(block_instructions)

    if not instructions:
        return None

    relative_binary = _relative_binary_name(binary_path, input_root)
    canonical_binary = _canonicalize_binary_key(relative_binary)
    opt = infer_opt_label(binary_path)
    function_name = getattr(func, "name", "unknown_func")
    function_addr = int(getattr(func, "addr", 0))

    return {
        "id": f"{canonical_binary}::{function_name}",
        "opt": opt,
        "source": relative_binary,
        "binary": relative_binary,
        "binary_key": canonical_binary,
        "function": function_name,
        "function_addr": function_addr,
        "instructions": instructions,
        "blocks": blocks,
        "num_instructions": len(instructions),
    }


def extract_palmtree_corpus(
    binary_input: Path,
    output_path: Path,
    min_instructions: int = 1,
    clean: bool = True,
    include_runtime_functions: bool = False,
    extra_excluded_functions: Optional[Iterable[str]] = None,
) -> Dict[str, int]:
    angr = _load_angr()
    binary_input = Path(binary_input)
    output_path = Path(output_path)

    if clean and output_path.exists():
        if output_path.is_file():
            output_path.unlink()
        else:
            shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    excluded_functions = set(extra_excluded_functions or [])
    if not include_runtime_functions:
        excluded_functions |= DEFAULT_EXCLUDED_FUNCTIONS

    binary_files = collect_binary_files(binary_input)
    stats = {
        "scanned_files": 0,
        "valid_binaries": 0,
        "emitted_functions": 0,
    }

    with output_path.open("w", encoding="utf-8") as fh:
        for binary_path in binary_files:
            stats["scanned_files"] += 1

            try:
                project = angr.Project(str(binary_path), load_options={"auto_load_libs": False})
                cfg = project.analyses.CFGFast(normalize=True, data_references=True)
            except Exception:
                continue

            stats["valid_binaries"] += 1
            symbol_map = _build_symbol_map(project)

            for func in cfg.kb.functions.values():
                if not _should_keep_function(func, excluded_functions):
                    continue

                record = _extract_function_record(
                    project=project,
                    binary_path=binary_path,
                    input_root=binary_input,
                    func=func,
                    symbol_map=symbol_map,
                )
                if record is None:
                    continue
                if int(record["num_instructions"]) < min_instructions:
                    continue

                fh.write(json.dumps(record) + "\n")
                stats["emitted_functions"] += 1

    return stats


def load_function_records(data_path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with Path(data_path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records
