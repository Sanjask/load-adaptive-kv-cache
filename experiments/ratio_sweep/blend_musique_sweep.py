from vllm import LLM, SamplingParams
import torch
import json
import numpy as np
from transformers import AutoTokenizer
from utils import load_dataset, normalize_question, build_qa_prompt, compute_f1
from pathlib import Path

# ---- SWEEP CONFIG ----
RATIOS = [0.05, 0.10, 0.16, 0.25, 0.40, 0.60, 1.0]
OUTPUT_JSON = "ratio_sweep_results.json"
# ----------------------

eval_dataset = load_dataset("inputs/musique_s.json")

llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2", gpu_memory_utilization=0.5)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
llm.set_tokenizer(tokenizer)

prefix_prompt = "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n"
query_prompt = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"

# results[ratio] = {"ttft": [...], "f1": [...]}
results = {r: {"ttft": [], "f1": []} for r in RATIOS}
# full prefill baseline (ratio-independent)
ttft_full = []
f1_full = []

cache_fuse_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata

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
    s_end_len = len(s_end)

    doc_chunk_ids = [s_start + chunk_ids for chunk_ids in doc_chunk_ids]
    doc_chunk_ids = [s_start_full] + doc_chunk_ids
    doc_chunk_ids = doc_chunk_ids + [s_start + q_ids + s_end]

    last_len = len([q_ids + s_end])

    # ---------- PHASE 1: collect KVs ONCE per example (ratio-independent) ----------
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
        if i == 0:
            temp_ids = doc_chunk_ids[i]
        else:
            temp_ids = doc_chunk_ids[i][s_start_1_len - 1:]
        input_ids += temp_ids
    input_prompt = tokenizer.decode(input_ids)

    # ---------- PHASE 2a: blend at EACH ratio (re-uses the same collected KVs) ----------
    for ratio in RATIOS:
        # old_kvs is mutated in-place by the splice (status=2), so re-seed a fresh copy each ratio
        fresh_kvs = [[k.clone(), v.clone()] for (k, v) in chunk_past_key_values]
        llm.llm_engine.model_executor.driver_worker.model_runner.model.model.old_kvs = fresh_kvs

        sampling_params = SamplingParams(temperature=0, max_tokens=32)
        cache_fuse_metadata["check"] = True
        cache_fuse_metadata['collect'] = False
        cache_fuse_metadata['suffix_len'] = last_len
        cache_fuse_metadata['recomp_ratio'] = ratio   # <-- the swept knob

        output = llm.generate([input_prompt], sampling_params)
        res = output[0].outputs[0].text
        ttft = output[0].metrics.first_token_time - output[0].metrics.first_scheduled_time
        f1 = max([compute_f1(res, answer, tokenizer) for answer in answers])
        results[ratio]["ttft"].append(ttft)
        results[ratio]["f1"].append(f1)

    # ---------- PHASE 2b: full prefill baseline (once per example) ----------
    sampling_params = SamplingParams(temperature=0, max_tokens=32)
    cache_fuse_metadata["check"] = False
    cache_fuse_metadata['collect'] = False
    output = llm.generate([input_prompt], sampling_params)
    res = output[0].outputs[0].text
    ttft = output[0].metrics.first_token_time - output[0].metrics.first_scheduled_time
    f1 = max([compute_f1(res, answer, tokenizer) for answer in answers])
    ttft_full.append(ttft)
    f1_full.append(f1)

    print(f"[{ex_idx+1}/{len(eval_dataset)}] done")

# ---------- AGGREGATE + SAVE ----------
summary = {"ratios": {}, "full_prefill": {}}
print("\n--------- Ratio Sweep Summary ---------")
print(f"{'ratio':>8} {'mean_TTFT':>12} {'mean_F1':>10}")
for ratio in RATIOS:
    mttft = float(np.mean(results[ratio]["ttft"]))
    mf1 = float(np.mean(results[ratio]["f1"]))
    summary["ratios"][ratio] = {"mean_ttft": mttft, "mean_f1": mf1}
    print(f"{ratio:>8} {mttft:>12.4f} {mf1:>10.4f}")

mttft_full = float(np.mean(ttft_full))
mf1_full = float(np.mean(f1_full))
summary["full_prefill"] = {"mean_ttft": mttft_full, "mean_f1": mf1_full}
print(f"{'full':>8} {mttft_full:>12.4f} {mf1_full:>10.4f}")

with open(OUTPUT_JSON, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved to {OUTPUT_JSON}")
