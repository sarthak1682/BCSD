
"""
Nova Dataset Extraction Pipeline

Extracts BinaryCorp .pkl files for use with the Nova-1.3b model.

Normalization logic (normalize_binarycorp_function) is adapted from the authors'
normalize.py in the official HF repo:
https://huggingface.co/spaces/ejschwartz/nova-6.7b/tree/main

Changes:
- Jump targets replaced with <label-N> tokens (or <unk> if external)
- Sequences hard-truncated to 256 labels (Nova tokenizer limit)
- AT&T regex cleanups: hex→decimal, strip '%', space punctuation
- Added handling for BinaryCorp hex format (e.g. '38h')

"""


import os
import tarfile
import pickle
import json
import re
from tqdm import tqdm

DATA_DIR = "binarycorp_3m"
TRAIN_TAR = os.path.join(DATA_DIR, "small_train.tar.gz")
TEST_TAR = os.path.join(DATA_DIR, "small_test.tar.gz")

def hex_to_decimal(match):
    # Use match.group(1) to grab JUST the hex digits, stripping the '0x' or 'h'
    return str(int(match.group(1), 16))


def normalize_binarycorp_function(asm_list):
    # Nova strictly requires a maximum of 256 labels per sequence
    asm_list = asm_list[:256]
    
    addr2label = {}
    parsed_lines = []
    
    for i, instr in enumerate(asm_list):
        label_tok = f"<label-{i+1}>"
        
        # Strip comments immediately (IDA uses ';', sometimes '#')
        instr = instr.split(';')[0].split('#')[0].strip()
        
        parts = instr.split(None, 1)
        if len(parts) == 2 and parts[0].startswith("0x"):
            addr = parts[0]
            if addr.endswith(':'): addr = addr[:-1]
            addr_stripped = addr[2:] 
            content = parts[1]
            addr2label[addr_stripped] = label_tok
        else:
            content = instr
            
        parsed_lines.append((content.strip(), label_tok))
        
    normalized_asm = ""
    for content, label in parsed_lines:
        
        if content.startswith('j') or content.startswith('loop') or content.startswith('call'):
            parts = content.split()
            if len(parts) == 2:
                inst, target_addr = parts[0], parts[1]
                
                if target_addr.startswith('0x'): 
                    target_addr = target_addr[2:]
                
                # Strip the 'h' suffix from target addresses if they are purely IDA hex
                if target_addr.endswith('h') and target_addr[:-1].isalnum():
                    target_addr = target_addr[:-1]
                
                if target_addr in addr2label:
                    content = f"{inst} {addr2label[target_addr]}"
                else:
                    content = f"{inst} <unk>"
                    
        # Apply regex cleanups
        # 1. Official 0x prefix hex
        content = re.sub(r"0x([0-9A-Fa-f]+)", hex_to_decimal, content)
        # 2. IDA 'h' suffix hex (Must start with a digit to avoid matching 'bh', 'ch' registers)
        content = re.sub(r"\b([0-9][0-9A-Fa-f]*)h\b", hex_to_decimal, content)
        
        content = content.replace('%', '')
        content = re.sub(r"([,(])|([),])", r' \1\2 ', content)
        content = re.sub(r' +', ' ', content).strip()
        
        if content: # Only append if the line wasn't just a comment
            normalized_asm += f"{content}\n{label}\n"
        
    return normalized_asm

def extract_and_convert(tar_path, output_jsonl, opts=['O0', 'O3'], limit=None):
    """Extract tar.gz and convert to jsonl in one step."""
    
    # Extract
    extract_dir = tar_path.replace('.tar.gz', '')
    if not os.path.exists(extract_dir):
        print(f"Extracting {tar_path}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(os.path.dirname(tar_path))
    
    # Find the actual extracted folder (may be nested)
    if os.path.isdir(extract_dir):
        data_dir = extract_dir
    else:
        data_dir = os.path.join(os.path.dirname(tar_path), os.listdir(os.path.dirname(tar_path))[0])
    
    print(f"Converting from {data_dir}...")
    
    pair_count = 0
    total_samples = 0
    
    with open(output_jsonl, 'w') as out:
        projects = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        for proj_name in tqdm(projects, desc="Projects"):
            if limit and pair_count >= limit:
                break
            
            proj_path = os.path.join(data_dir, proj_name)
            
            # Find pkl files for each optimization
            opt_files = {}
            for filename in os.listdir(proj_path):
                if filename == 'saved_index.pkl' or not filename.endswith('.pkl'):
                    continue
                for opt in opts:
                    if f'-{opt}-' in filename:
                        opt_files[opt] = os.path.join(proj_path, filename)
                        break
            
            if len(opt_files) < len(opts):
                continue
            
            # Load pkl files
            opt_data = {}
            try:
                for opt, pkl_path in opt_files.items():
                    with open(pkl_path, 'rb') as f:
                        opt_data[opt] = pickle.load(f)
            except:
                continue
            
            # Find common functions
            common_funcs = set(opt_data[opts[0]].keys())
            for opt in opts[1:]:
                common_funcs &= set(opt_data[opt].keys())
            
            # Export
            for func_name in common_funcs:
                if limit and pair_count >= limit:
                    break
                
                for opt in opts:
                    func_data = opt_data[opt][func_name]
                    func_addr, asm_list = func_data[0], func_data[1]
                    
                    # Normalizes jump targets, converts hex, spacing, and assigns unique labels
                    labeled_asm = normalize_binarycorp_function(asm_list)
                    
                    row = {"id": f"{proj_name}__{func_name}", "opt": opt, "asm": labeled_asm}
                    out.write(json.dumps(row) + "\n")
                    total_samples += 1
                
                pair_count += 1
    
    print(f"✓ Wrote {total_samples} samples ({pair_count} pairs) to {output_jsonl}")

# Convert test set
extract_and_convert(TEST_TAR, "binarycorp3m_test_nova.jsonl")

# Convert train set
extract_and_convert(TRAIN_TAR, "binarycorp3m_train_nova.jsonl")
