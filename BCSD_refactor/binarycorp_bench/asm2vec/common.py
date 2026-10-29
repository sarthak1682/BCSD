import re


LABEL_RE = re.compile(r"^<label-([0-9]+)>$")


def asm_to_asm2vec_text(asm):
    raw_lines = asm if isinstance(asm, list) else asm.split("\n")
    out_lines = ["_block_label:"]

    for raw in raw_lines:
        line = raw.strip()
        if not line:
            continue

        match = LABEL_RE.match(line)
        if match:
            out_lines.append(f"LABEL{match.group(1)}:")
            continue

        # Branch targets in the bench split are written as <label-N>. Asm2Vec's
        # parser only links labels when instruction arguments match label names.
        line = re.sub(r"<label-([0-9]+)>", r"LABEL\1", line)
        out_lines.append(f"    {line}")

    return "\n".join(out_lines)
