#!/usr/bin/env python3
"""
Inspect GGUF file to see actual tensor names
"""

import sys

try:
    from gguf import GGUFReader 

except ImportError:
    print("Error: gguf library not installed")
    print("Install with: pip install gguf")
    sys.exit(1)

def read_gguf_tensors(filepath):
    reader = GGUFReader(filepath)
    
    print(f"Architecture: {reader.fields.get('general.architecture', 'unknown')}")
    print(f"Tensor count: {len(reader.tensors)}")
    
    # Categorize tensors
    layer_tensors = {}
    other_tensors = []
    
    for tensor in reader.tensors:
        name = tensor.name
        
        if 'blk.' in name:
            parts = name.split('.')
            layer_num = int(parts[1])
            tensor_type = '.'.join(parts[2:])
            
            if layer_num not in layer_tensors:
                layer_tensors[layer_num] = []
            layer_tensors[layer_num].append(tensor_type)
        else:
            other_tensors.append(name)
    
    # Print results
    print("\n" + "=" * 80)
    print("NON-LAYER TENSORS:")
    print("=" * 80)
    for name in sorted(other_tensors):
        print(f"  {name}")
    
    if 0 in layer_tensors:
        print("\n" + "=" * 80)
        print("LAYER 0 TENSORS (example):")
        print("=" * 80)
        for t in sorted(layer_tensors[0]):
            print(f"  blk.0.{t}")
    
    print("\n" + "=" * 80)
    print(f"Total layers found: {max(layer_tensors.keys()) + 1 if layer_tensors else 0}")
    print("=" * 80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_gguf.py <model.gguf>")
        sys.exit(1)
    
    read_gguf_tensors(sys.argv[1])