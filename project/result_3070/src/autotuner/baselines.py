"""Baselines fixieren (M0, Punkt 4)

Die Tuning-Ziele in M3/M4 werden relativ zu diesen Werten definiert.
"""

# GPU auf der die Baselines gemessen wurden (als Sanity-Check zur Laufzeit)
# abrufbar über: props = torch.cuda.get_device_properties(i)
# print(f"\nGPU {i}: {props.name}")
BASELINE_GPU = "NVIDIA GB10"

# TFLOPS, gemessen mit triton.testing.do_bench (warmup=200, rep=2000)
# Assignment 5: cmk,ckn->cmn mit C=4, M=N=K=4096, fp16
# Quelle: assignments/05_assignment/out/task4.log
A05_HAND_L2_TFLOPS = 66.10      # handoptimierte L2-Swizzle-Variante 
A05_BASELINE_TFLOPS = 38.60     # baseline: cuTile ohne L2-Swizzling (untere Grenze)

# Assignment 6: acspx,bspy->abcyx, fp16
# Quelle: assignments/06_assignment/out/task4.log
A06_HAND_CUTILE_TFLOPS = 49.84  # handoptimierte cuTile-Variante (Referenz für M4)
A06_TORCH_EINSUM_TFLOPS = 16.18 # torch.einsum

# geschätzte Erfolgsschwellen aus der Roadmap (projekt_b_cutile_autotuner.md)
M3_TARGET_FRACTION = 0.95       # >=95 % von A05_HAND_L2_TFLOPS
M4_TARGET_FRACTION = 0.90       # >=90 % von A06_HAND_CUTILE_TFLOPS

A05_TARGET_TFLOPS = A05_HAND_L2_TFLOPS * M3_TARGET_FRACTION
A06_TARGET_TFLOPS = A06_HAND_CUTILE_TFLOPS * M4_TARGET_FRACTION
