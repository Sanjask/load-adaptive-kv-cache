"""
Dynamic recompute-ratio sweep: calibrate coverage_target offline.

Instead of a fixed ratio, CacheBlend's KV-divergence signal decides how many
tokens to recompute per example. coverage_target controls how much of the total
mismatch energy must be covered. This sweep finds the best coverage_target.

Requires: patch_xformers.py applied first.
"""
from vllm import LLM, SamplingParams
import torch
import json
import time
import random
import numpy as np
from transformers import AutoTokenizer
from utils import load_dataset, build_qa_prompt, compute_f1

# ---------------- CONFIG ----------------
COVERAGE_TARGETS = [0.70, 0.80, 0.85, 0.90, 0.95, 0.98]
N_REPS = 3
OUTPUT_JSON = "dynamic_sweep_results.json"
SEED = 0
# ----------------------------------------

random.seed(SEED)
np.random.seed(SEED)

eval_dataset = load_dataset("inputs/musique_s.json")

llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2", gpu_memory_utilization=0.5)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
llm.set_tokenizer(tokenizer)

prefix_prompt = "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n"
query_prompt = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"

cache_fuse_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata

records = []
full_baseline = {}


def run_generate_ttft(prompt, max_tokens=32):
    sp = SamplingParams(temperature=0, max_tokens=max_tokens)
    out = llm.generate([prompt], sp)
    res = out[0].outputs[0].text
    ttft = out[0].metrics.first_token_time - out[0].metrics.first_scheduled_time
    return res, ttft


for ex_idx, ex in enumerate(eval_dataset):
    answers = ex["answers"]
    doc_prompts, q_prompt = build_qa_prompt(ex, query_prompt)
    doc_chunk_ids = [tokenizer.encode(doc)[1:] for doc in doc_prompts]
    q_ids = tokenizer.encode(q_prompt)[1:]

    sampling_params = SamplingParams(temperature=0, max_tokens=1)
    cache_fuse_metadata['collect'] = False
    cache_fuse_metadata['check'] = False

    s_start_full = [733, 16289, 28793] + tokenizer.encode(prefix_prompt)[1:]
    s_start_len = len(s_start_full) + 1
    s_start = []
    s_start_1_len = len(s_start) + 1
    s_end = [733, 28748, 16289, 28793]

    doc_chunk_ids = [s_start + chunk_ids for chunk_ids in doc_chunk_ids]
    doc_chunk_ids = [s_start_full] + doc_chunk_ids
    doc_chunk_ids = doc_chunk_ids + [s_start + q_ids + s_end]

    # BUG FIX
    last_len = len(q_ids + s_end)

    # ---------- PHASE 1: collect KVs ONCE ----------
    cache_fuse_metadata['collect'] = True
    cache_fuse_metadata["check"] = False
    num_layer = 32
    chunk_past_key_values = []

    for i in range(len(doc_chunk_ids)):
        prompts = [tokenizer.decode(doc_chunk_ids[i])]
        llm.generate(prompts, sampling_params)
        llm_layers = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
        for j in range(num_layer):
            past_key_values = llm_layers[j].self_attn.hack_kv
            if i == 0:
                temp_k = past_key_values[0][:s_start_len].clone()
                temp_v = past_key_values[1][:s_start_len].clone()
            else:
                temp_k = past_key_values[0][s_start_1_len:len(doc_chunk_ids[i]) + 1].clone()
                temp_v = past_key_values[1][s_start_1_len:len(doc_chunk_ids[i]) + 1].clone()
            if i == 0:
                chunk_past_key_values.append([temp_k, temp_v])
            else:
                chunk_past_key_values[j][0] = torch.cat((chunk_past_key_values[j][0], temp_k), dim=0)
                chunk_past_key_values[j][1] = torch.cat((chunk_past_key_values[j][1], temp_v), dim=0)

    input_ids = []
    for i in range(len(doc_chunk_ids)):
        temp_ids = doc_chunk_ids[i] if i == 0 else doc_chunk_ids[i][s_start_1_len - 1:]
        input_ids += temp_ids
    input_prompt = tokenizer.decode(input_ids)

    # ---------- FULL PREFILL BASELINE ----------
    full_ttfts = []
    full_res = None
    for _ in range(N_REPS):
        cache_fuse_metadata["check"] = False
        cache_fuse_metadata['collect'] = False
        cache_fuse_metadata['dynamic_ratio'] = False
        res, ttft = run_generate_ttft(input_prompt)
        full_ttfts.append(ttft)
        full_res = res
    full_f1 = max([compute_f1(full_res, a, tokenizer) for a in answers])
    full_baseline[ex_idx] = {"ttft_reps": full_ttfts, "f1": full_f1, "prediction": full_res}

    # ---------- DYNAMIC RATIO at each coverage target ----------
    for rep in range(N_REPS):
        target_order = COVERAGE_TARGETS[:]
        random.shuffle(target_order)
        for tau in target_order:
            fresh_kvs = [[k.clone(), v.clone()] for (k, v) in chunk_past_key_values]
            llm.llm_engine.model_executor.driver_worker.model_runner.model.model.old_kvs = fresh_kvs

            cache_fuse_metadata["check"] = True
            cache_fuse_metadata['collect'] = False
            cache_fuse_metadata['suffix_len'] = last_len
            cache_fuse_metadata['dynamic_ratio'] = True
            cache_fuse_metadata['coverage_target'] = tau

            res, ttft = run_generate_ttft(input_prompt)
            chosen_ratio = cache_fuse_metadata.get("chosen_ratio", -1)
            f1 = max([compute_f1(res, a, tokenizer) for a in answers])

            records.append({
                "idx": ex_idx,
                "rep": rep,
                "coverage_target": tau,
                "chosen_ratio": chosen_ratio,
                "ttft": ttft,
                "f1": f1,
                "prediction": res,
                "full_prediction_match": (res.strip() == full_baseline[ex_idx]["prediction"].strip()),
                "f1_minus_full": f1 - full_baseline[ex_idx]["f1"],
                "input_len": len(input_ids),
            })

    print(f"[{ex_idx+1}/{len(eval_dataset)}] done")

# ---------------- AGGREGATE ----------------
def pct(arr, p):
    return float(np.percentile(arr, p)) if len(arr) else float("nan")

summary = {"config": {"coverage_targets": COVERAGE_TARGETS, "n_reps": N_REPS,
                       "n_examples": len(eval_dataset), "seed": SEED},
           "targets": {}, "full_prefill": {}}

all_full_ttft = [t for d in full_baseline.values() for t in d["ttft_reps"]]
all_full_f1 = [d["f1"] for d in full_baseline.values()]
summary["full_prefill"] = {
    "mean_ttft": float(np.mean(all_full_ttft)),
    "p50_ttft": pct(all_full_ttft, 50),
    "p95_ttft": pct(all_full_ttft, 95),
    "mean_f1": float(np.mean(all_full_f1)),
}

print("\n---- Dynamic Ratio Sweep Summary ----")
print(f"{'target':>8} {'mean_ratio':>11} {'mean_ttft':>10} {'p50':>8} {'p95':>8} {'mean_f1':>9} {'mean_dF1':>9} {'exact%':>8}")
for tau in COVERAGE_TARGETS:
    rows = [r for r in records if r["coverage_target"] == tau]
    ratios = [r["chosen_ratio"] for r in rows]
    ttfts = [r["ttft"] for r in rows]
    f1s = [r["f1"] for r in rows]
    dF1s = [r["f1_minus_full"] for r in rows]
    exact = [r["full_prediction_match"] for r in rows]
    summary["targets"][tau] = {
        "mean_chosen_ratio": float(np.mean(ratios)),
        "mean_ttft": float(np.mean(ttfts)),
        "p50_ttft": pct(ttfts, 50),
        "p95_ttft": pct(ttfts, 95),
        "mean_f1": float(np.mean(f1s)),
        "mean_f1_minus_full": float(np.mean(dF1s)),
        "exact_match_rate": float(np.mean(exact)),
        "ratio_distribution": {
            r: float(np.mean([1 for x in ratios if x == r])) for r in [0.05, 0.10, 0.16, 0.25, 0.40, 0.60]
        },
    }
    s = summary["targets"][tau]
    print(f"{tau:>8} {s['mean_chosen_ratio']:>11.3f} {s['mean_ttft']:>10.4f} {s['p50_ttft']:>8.4f} "
          f"{s['p95_ttft']:>8.4f} {s['mean_f1']:>9.4f} {s['mean_f1_minus_full']:>9.4f} "
          f"{s['exact_match_rate']*100:>7.1f}%")

fp = summary["full_prefill"]
print(f"{'full':>8} {'--':>11} {fp['mean_ttft']:>10.4f} {fp['p50_ttft']:>8.4f} {fp['p95_ttft']:>8.4f} {fp['mean_f1']:>9.4f}")

with open(OUTPUT_JSON, "w") as f:
    json.dump({"summary": summary, "records": records,
               "full_baseline": {str(k): v for k, v in full_baseline.items()}}, f, indent=2)
print(f"\nSaved {len(records)} records to {OUTPUT_JSON}")