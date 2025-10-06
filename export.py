# export_layerskip.py

import argparse
import json
import math
import os
import struct
from pathlib import Path
from typing import Dict, List

import numpy as np
from jinja2 import Template

# LayerSkip-Llama3.2-1B Configuration (must match config.h)
SEQ_LEN = 4096
VOCAB_SIZE = 128256
DIM = 2048
HIDDEN_DIM = 8192
N_LAYERS = 16
N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 64
Q_DIM = N_HEADS * HEAD_DIM
KV_DIM = N_KV_HEADS * HEAD_DIM

def bytes_to_unicode():
    """Reference GPT-2/Llama byte→Unicode map."""
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))

def internal_to_bytes(U2B: Dict, token_str: str) -> bytes:
    """Convert token string to bytes using Unicode→Byte mapping."""
    return b''.join(
        bytes([U2B[ch]]) if ch in U2B else ch.encode('utf-8')
        for ch in token_str
    )

def build_tokenizer(model, output_dir: str):
    """Export tokenizer to binary format for C++ inference."""
    B2U = bytes_to_unicode()
    U2B = {u: b for b, u in B2U.items()}

    tokenizer = model.tokenizer

    # Get ID → token mapping
    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    all_tokens = [id_to_token[i] for i in sorted(id_to_token)]

    # Get tokenizer backend data
    tokenizer_data = json.loads(tokenizer.backend_tokenizer.to_str())

    # Extract vocab and merge rules (for BPE-based tokenizers)
    model_type = tokenizer_data.get("model", {}).get("type", "BPE")
    
    if model_type == "BPE":
        vocab_dict = tokenizer_data["model"]["vocab"]
        merges = tokenizer_data["model"].get("merges", [])
        
        # Build merge rank table
        merge_rank = {}
        for i, merge in enumerate(merges):
            merge_key = ''.join(tuple(merge if isinstance(merge, list) else merge.split()))
            merge_rank[merge_key] = i

        # Create pseudo-score dictionary
        # Tokens from initial vocab get score 0 (unmerged tokens)
        # Merged tokens get scores based on merge rank
        pseudo_scores = {}
        for token_id, token in enumerate(all_tokens):
            rank = merge_rank.get(token)
            if rank is not None:
                score = -math.log(rank + 1)
            else:
                score = -1e6  # Initial vocab tokens
            pseudo_scores[token] = score
    else:
        # For non-BPE tokenizers, use uniform scores
        print(f"Warning: Tokenizer type {model_type} - using uniform scores")
        pseudo_scores = {token: -1e6 for token in all_tokens}

    max_token_length = max(len(t) for t in all_tokens)
    tokenizer_path = os.path.join(output_dir, "tokenizer.bin")

    with open(tokenizer_path, "wb") as out_f:
        # Header: max_token_length, vocab_size, bos_token_id, eos_token_id
        out_f.write(struct.pack("<I", max_token_length))
        out_f.write(struct.pack("<I", len(all_tokens)))
        out_f.write(struct.pack("<I", model.bos_token_id if model.bos_token_id else 0))
        out_f.write(struct.pack("<I", model.eos_token_id if model.eos_token_id else 0))

        # Write each token
        for token_id, token in enumerate(all_tokens):
            token_bytes = internal_to_bytes(U2B, token)
            out_f.write(struct.pack("f", pseudo_scores[token]))  # merge score
            out_f.write(struct.pack("<I", len(token_bytes)))     # token length
            out_f.write(token_bytes)                              # UTF-8 bytes

    print(f"[OK] Written tokenizer to {tokenizer_path}")
    print(f"  Vocab size: {len(all_tokens)}")
    print(f"  Max token length: {max_token_length}")
    print(f"  BOS token ID: {model.bos_token_id}")
    print(f"  EOS token ID: {model.eos_token_id}")

def build_prompts(model, output_dir: str):
    """Generate chat template files for different conversation modes."""
    
    # Check if model has chat template
    if not hasattr(model.tokenizer, 'chat_template') or not model.tokenizer.chat_template:
        print("Warning: No chat template found. Creating basic templates.")
        
        # Create basic Llama 3.2 style templates
        basic_user = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        basic_system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        templates = {
            'template_user.txt': basic_user,
            'template_system.txt': basic_system,
        }
        
        for filename, template_text in templates.items():
            with open(os.path.join(output_dir, filename), 'w', encoding='utf-8', newline='') as f:
                f.write(template_text)
        
        print(f"[OK] Written basic prompt templates to '{output_dir}'")
        return

    template = Template(model.tokenizer.chat_template)

    templates_to_generate = []

    # Template 1: User only
    messages_user = [{"role": "user", "content": "%s"}]
    try:
        rendered = template.render(messages=messages_user, add_generation_prompt=True)
        templates_to_generate.append(('template_user.txt', rendered))
    except Exception as e:
        print(f"Warning: Could not render user template: {e}")

    # Template 2: System + User
    messages_system = [
        {"role": "system", "content": "%s"},
        {"role": "user", "content": "%s"}
    ]
    try:
        rendered = template.render(messages=messages_system, add_generation_prompt=True)
        templates_to_generate.append(('template_system.txt', rendered))
    except Exception as e:
        print(f"Warning: Could not render system template: {e}")

    # Check if model supports thinking mode (unlikely for base Llama)
    try:
        rendered = template.render(messages=messages_user, add_generation_prompt=True, enable_thinking=True)
        templates_to_generate.append(('template_user_thinking.txt', rendered))
        
        rendered = template.render(messages=messages_system, add_generation_prompt=True, enable_thinking=True)
        templates_to_generate.append(('template_system_thinking.txt', rendered))
    except Exception:
        pass  # Thinking mode not supported, skip silently

    # Write templates
    for filename, template_text in templates_to_generate:
        with open(os.path.join(output_dir, filename), 'w', encoding='utf-8', newline='') as f:
            f.write(template_text)

    print(f"[OK] Written {len(templates_to_generate)} prompt template(s) to '{output_dir}'")

def verify_gguf_model(model_path: str):
    """Verify that the GGUF model exists and print basic info."""
    gguf_path = Path(model_path)
    
    # Look for .gguf file
    gguf_files = list(gguf_path.glob("*.gguf"))
    
    if not gguf_files:
        print(f"Warning: No .gguf files found in {model_path}")
        return False
    
    print(f"\n[OK] Found GGUF model file(s):")
    for gguf_file in gguf_files:
        size_mb = gguf_file.stat().st_size / (1024 * 1024)
        print(f"  - {gguf_file.name} ({size_mb:.1f} MB)")
    
    return True

def load_tokenizer_and_config(model_path: str):
    """Loads tokenizer and config from Hugging Face model."""
    try:
        from transformers import AutoConfig, AutoTokenizer
        from types import SimpleNamespace
    except ImportError:
        print("Error: transformers package is required.")
        print("Please run: pip install transformers")
        return None

    print(f"\nLoading tokenizer from: {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    model_mock = SimpleNamespace()
    model_mock.tokenizer = tokenizer
    model_mock.bos_token_id = getattr(hf_config, "bos_token_id", 128000) 
    model_mock.eos_token_id = getattr(hf_config, "eos_token_id", 128009)  
    
    print(f"[OK] Loaded tokenizer and config")
    print(f"  Architecture: {getattr(hf_config, 'architectures', ['unknown'])[0]}")
    print(f"  Hidden size: {getattr(hf_config, 'hidden_size', 'unknown')}")
    print(f"  Num layers: {getattr(hf_config, 'num_hidden_layers', 'unknown')}")
    print(f"  Vocab size: {getattr(hf_config, 'vocab_size', 'unknown')}")
    
    return model_mock

def main():
    parser = argparse.ArgumentParser(
        description="Export LayerSkip-Llama3.2-1B tokenizer for C++ inference"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the Hugging Face model directory (for tokenizer) or GGUF model directory"
    )
    parser.add_argument(
        "--verify-gguf",
        action='store_true',
        help="Verify GGUF model file exists"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("LayerSkip-Llama3.2-1B Tokenizer Export")
    print("=" * 70)

    # Verify GGUF if requested
    if args.verify_gguf:
        verify_gguf_model(args.model_path)

    # Load tokenizer
    model_info = load_tokenizer_and_config(args.model_path)

    if model_info:
        print("\n" + "=" * 70)
        print("Exporting tokenizer and templates...")
        print("=" * 70)
        
        build_tokenizer(model_info, args.model_path)
        build_prompts(model_info, args.model_path)
        
        print("\n" + "=" * 70)
        print("[OK] Export complete!")
        print("=" * 70)
        print(f"\nGenerated files in: {args.model_path}")
        print("  - tokenizer.bin")
        print("  - template_*.txt")
        print("\nNote: GGUF weights are already quantized and don't need export.")
        print("Use the .gguf file directly with your C++ inference code.")
    else:
        print("\n[ERROR] Failed to load tokenizer")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())