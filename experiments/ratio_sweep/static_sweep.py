"""
Proper static recompute-ratio sweep for CacheBlend on musique.

Fixes the last_len bug and adds rigorous measurement per advisor's spec:
  - per-example result logging (idx, ratio, ttft, f1, prediction, lengths, doc counts)
  - exact match against full-prefill output
  - paired F1 difference vs full prefill per sample
  - p50 / p95 TTFT (not just mean)
  - randomized ratio order per example (avoid warm-GPU / allocator bias)
  - N repetitions for TTFT
  - KV collection cost measured and reported SEPARATELY from online (cache-hit) TTFT

Bootstrap CIs are computed in a separate post-processing script (no GPU needed).
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
RATIOS = [0.05, 0.10, 0.16, 0.25, 0.40, 0.60, 1.0]
N_REPS = 3                     # repetitions for TTFT stability
OUTPUT_JSON = "static_sweep_results.json"
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

# flat list of per-(example,ratio,rep) records
records = []
# full-prefill baseline per example (ratio-independent): idx -> {ttft_reps, f1, prediction}
full_baseline = {}
# one-time KV collection cost per example (the cost the paper amortizes across requests)
collection_cost = {}


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

    # ---- BUG FIX: was len([q_ids + s_end]) which is always 1 ----
    last_len = len(q_ids + s_end)

    # ---------- PHASE 1: collect KVs ONCE, and TIME it separately ----------
    cache_fuse_metadata['collect'] = True
    cache_fuse_metadata["check"] = False
    num_layer = 32
    chunk_past_key_values = []

    torch.cuda.synchronize()
    t_collect_start = time.perf_counter()
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
    torch.cuda.synchronize()
    collection_cost[ex_idx] = time.perf_counter() - t_collect_start

    input_ids = []
    for i in range(len(doc_chunk_ids)):
        temp_ids = doc_chunk_ids[i] if i == 0 else doc_chunk_ids[i][s_start_1_len - 1:]
        input_ids += temp_ids
    input_prompt = tokenizer.decode(input_ids)

    # ---------- FULL PREFILL BASELINE (once per example, N reps for TTFT) ----------
    full_ttfts = []
    full_res = None
    for _ in range(N_REPS):
        cache_fuse_metadata["check"] = False
        cache_fuse_metadata['collect'] = False
        res, ttft = run_generate_ttft(input_prompt)
        full_ttfts.append(ttft)
        full_res = res
    full_f1 = max([compute_f1(full_res, a, tokenizer) for a in answers])
    full_baseline[ex_idx] = {"ttft_reps": full_ttfts, "f1": full_f1, "prediction": full_res}

    # ---------- BLEND at each ratio, RANDOMIZED order, N reps ----------
    for rep in range(N_REPS):
        ratio_order = RATIOS[:]
        random.shuffle(ratio_order)              # randomized per example & per rep
        for ratio in ratio_order:
            # old_kvs is mutated in place by the splice, so re-seed a fresh clone each run
            fresh_kvs = [[k.clone(), v.clone()] for (k, v) in chunk_past_key_values]
            llm.llm_engine.model_executor.driver_worker.model_runner.model.model.old_kvs = fresh_kvs

            cache_fuse_metadata["check"] = True
            cache_fuse_metadata['collect'] = False
            cache_fuse_metadata['suffix_len'] = last_len
            cache_fuse_metadata['recomp_ratio'] = ratio

            res, ttft = run_generate_ttft(input_prompt)
            f1 = max([compute_f1(res, a, tokenizer) for a in answers])

            records.append({
                "idx": ex_idx,
                "rep": rep,
                "ratio": ratio,
                "ttft": ttft,
                "f1": f1,
                "prediction": res,
                "full_prediction_match": (res.strip() == full_baseline[ex_idx]["prediction"].strip()),
                "f1_minus_full": f1 - full_baseline[ex_idx]["f1"],
                "input_len": len(input_ids),
                "query_len": len(q_ids),
                "num_docs": len(doc_prompts),
                "doc_lens": [len(x) for x in doc_chunk_ids[:-1]],
            })

    print(f"[{ex_idx+1}/{len(eval_dataset)}] collect={collection_cost[ex_idx]*1000:.1f}ms  full_ttft~{np.median(full_ttfts):.4f}")

# ---------------- AGGREGATE ----------------
def pct(arr, p):
    return float(np.percentile(arr, p)) if len(arr) else float("nan")

summary = {"config": {"ratios": RATIOS, "n_reps": N_REPS, "n_examples": len(eval_dataset), "seed": SEED},
           "ratios": {}, "full_prefill": {}, "collection_cost": {}}

# full prefill aggregate
all_full_ttft = [t for d in full_baseline.values() for t in d["ttft_reps"]]
all_full_f1 = [d["f1"] for d in full_baseline.values()]
summary["full_prefill"] = {
    "mean_ttft": float(np.mean(all_full_ttft)),
    "p50_ttft": pct(all_full_ttft, 50),
    "p95_ttft": pct(all_full_ttft, 95),
    "mean_f1": float(np.mean(all_full_f1)),
}

# collection cost aggregate (reported separately, NOT folded into online TTFT)
all_collect = list(collection_cost.values())
summary["collection_cost"] = {
    "mean_s": float(np.mean(all_collect)),
    "p50_s": pct(all_collect, 50),
    "p95_s": pct(all_collect, 95),
    "note": "one-time per-context KV build cost; amortized across reuse in the paper's premise",
}

print("\n---- Static Sweep Summary (online / cache-hit TTFT) ----")
print(f"{'ratio':>7} {'mean_ttft':>10} {'p50':>8} {'p95':>8} {'mean_f1':>9} {'mean_dF1':>9} {'exact%':>8}")
for ratio in RATIOS:
    rows = [r for r in records if r["ratio"] == ratio]
    ttfts = [r["ttft"] for r in rows]
    f1s = [r["f1"] for r in rows]
    dF1s = [r["f1_minus_full"] for r in rows]
    exact = [r["full_prediction_match"] for r in rows]
    summary["ratios"][ratio] = {
        "mean_ttft": float(np.mean(ttfts)),
        "p50_ttft": pct(ttfts, 50),
        "p95_ttft": pct(ttfts, 95),
        "mean_f1": float(np.mean(f1s)),
        "mean_f1_minus_full": float(np.mean(dF1s)),
        "exact_match_rate": float(np.mean(exact)),
    }
    s = summary["ratios"][ratio]
    print(f"{ratio:>7} {s['mean_ttft']:>10.4f} {s['p50_ttft']:>8.4f} {s['p95_ttft']:>8.4f} "
          f"{s['mean_f1']:>9.4f} {s['mean_f1_minus_full']:>9.4f} {s['exact_match_rate']*100:>7.1f}%")

fp = summary["full_prefill"]
print(f"{'full':>7} {fp['mean_ttft']:>10.4f} {fp['p50_ttft']:>8.4f} {fp['p95_ttft']:>8.4f} {fp['mean_f1']:>9.4f}")
print(f"\nMean one-time KV collection cost: {summary['collection_cost']['mean_s']*1000:.1f} ms "
      f"(p95 {summary['collection_cost']['p95_s']*1000:.1f} ms) -- reported separately")

# save EVERYTHING (per-example records too, for bootstrap CIs later)
with open(OUTPUT_JSON, "w") as f:
    json.dump({"summary": summary, "records": records,
               "full_baseline": full_baseline,
               "collection_cost": collection_cost}, f, indent=2)
print(f"\nSaved {len(records)} records to {OUTPUT_JSON}")