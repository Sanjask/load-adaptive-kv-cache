"""
Patches vllm_blend's xformers.py to support dynamic recompute ratio selection.

Run this ON THE POD after cloning CacheBlend:
    python3 patch_xformers.py /ephemeral/CacheBlend/vllm_blend/vllm/attention/backends/xformers.py

What it does:
  - Adds choose_dynamic_ratio() function near the top of xformers.py
  - Modifies the check-layer logic (status==1) to optionally use dynamic ratio
  - Controlled by cache_fuse_metadata["dynamic_ratio"]: if True, uses coverage-based
    selection; if False (default), uses the fixed recomp_ratio as before

This is a PATCH, not a replacement. It finds the exact lines and swaps them.
"""
import sys

if len(sys.argv) != 2:
    print("Usage: python3 patch_xformers.py <path_to_xformers.py>")
    sys.exit(1)

filepath = sys.argv[1]
with open(filepath, "r") as f:
    content = f.read()

# ---- PATCH 1: Add the dynamic ratio function after the imports ----
dynamic_fn = '''
def choose_dynamic_ratio(temp_diff,
                         min_ratio=0.05,
                         max_ratio=0.60,
                         coverage_target=0.90,
                         allowed_ratios=(0.05, 0.10, 0.16, 0.25, 0.40, 0.60)):
    """
    Select the smallest recomputation budget whose largest KV deviations
    explain `coverage_target` of total mismatch energy.
    """
    n = temp_diff.numel()
    if n == 0:
        return min_ratio
    sorted_diff = torch.sort(temp_diff.float(), descending=True).values
    total = sorted_diff.sum()
    if total <= 1e-12:
        return min_ratio
    cumulative_mass = torch.cumsum(sorted_diff, dim=0) / total
    k = int(torch.searchsorted(
        cumulative_mass,
        torch.tensor(coverage_target, device=temp_diff.device)
    ).item()) + 1
    raw_ratio = k / n
    raw_ratio = max(min_ratio, min(max_ratio, raw_ratio))
    return min(allowed_ratios, key=lambda r: abs(r - raw_ratio))

'''

# Insert after the last top-level import line
# Find a safe anchor: the class definition
anchor = "class XFormersImpl(AttentionImpl):"
if anchor not in content:
    print(f"ERROR: Could not find '{anchor}' in {filepath}")
    sys.exit(1)
content = content.replace(anchor, dynamic_fn + anchor)
print("PATCH 1: Added choose_dynamic_ratio() function")

# ---- PATCH 2: Modify the check-layer logic to support dynamic ratio ----
old_block = """            topk_num = int((total_len-last_len)*cache_fuse_metadata["recomp_ratio"])
            temp_diff = torch.sum((value[:-last_len,:,:]-value_old[:-last_len,:,:])**2, dim=[1,2])
            top_indices = torch.topk(temp_diff, k=topk_num).indices"""

new_block = """            prefix_len = total_len - last_len
            temp_diff = torch.sum((value[:-last_len,:,:]-value_old[:-last_len,:,:])**2, dim=[1,2])
            if cache_fuse_metadata.get("dynamic_ratio", False):
                coverage_target = cache_fuse_metadata.get("coverage_target", 0.90)
                dynamic_ratio = choose_dynamic_ratio(temp_diff, coverage_target=coverage_target)
                cache_fuse_metadata["chosen_ratio"] = dynamic_ratio
                topk_num = max(1, int(prefix_len * dynamic_ratio))
            else:
                topk_num = int(prefix_len * cache_fuse_metadata["recomp_ratio"])
            top_indices = torch.topk(temp_diff, k=topk_num).indices"""

if old_block not in content:
    print(f"ERROR: Could not find the target code block in {filepath}")
    print("The file may have already been patched or the code has changed.")
    sys.exit(1)

content = content.replace(old_block, new_block)
print("PATCH 2: Modified check-layer logic for dynamic ratio support")

with open(filepath, "w") as f:
    f.write(content)
print(f"Successfully patched {filepath}")