# CacheBlend Baseline Reproduction
- Date: 2026-06-18
- Hardware: A100 80GB PCIe
- Model: Mistral-7B-Instruct-v0.2
- Dataset: musique_s (150 records)
- Recompute ratio: 0.16 (hardcoded)

## Results
- TTFT with cache (CacheBlend): 0.1256s
- TTFT with full prefill: 0.5143s
- Speedup: ~4.1x
- F1 with cache: 0.2818
- F1 with full prefill: 0.2812
- Quality delta: negligible

## Gate
Week-2 de-risk gate: PASSED
