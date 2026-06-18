# Reproducing the CacheBlend Baseline

## Requirements
- NVIDIA GPU >= 40GB (tested on A100 80GB)
- CUDA >= 12.1
- Python >= 3.9

## Steps
1. Clone CacheBlend: `git clone https://github.com/YaoJiayi/CacheBlend.git`
2. Install vLLM fork: `cd CacheBlend/vllm_blend && pip install -e .`
3. Install deps: `pip install -r requirements.txt && pip install "numpy<2" "transformers<4.45"`
4. Set HF token: `export HF_TOKEN=<your_token>`
5. Download model: `python3 -c "from huggingface_hub import snapshot_download; snapshot_download('mistralai/Mistral-7B-Instruct-v0.2', token='$HF_TOKEN')"`
6. Run: `cd CacheBlend && python3 example/blend_musique.py`
