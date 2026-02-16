# Augment_tableQA
This is the implementation for the paper: [Augment before You Try: Knowledge-Enhanced Table Question Answering via Table Expansion](https://arxiv.org/abs/2401.15555).

## Requirements
### Environment
Install conda environment by running
```
conda env create -f environment.yml
conda activate augment
```

## Usage
### 1) vLLM endpoint (open-source models)
Inference uses a local vLLM server. Configure it once:

1. Copy `vllm_config.example.json` to `vllm_config.json` and set `base_url` and `model` to your vLLM server (e.g. `http://localhost:8000/v1`, `google/gemma-2b-it`).
2. Create `key.txt` in the project root with any single line (e.g. `dummy`); it is not used for auth when using vLLM.

See [VLLM_SETUP.md](VLLM_SETUP.md) for starting the vLLM server (including T4 GPU settings) and for remote server + SSH tunnel. Verify with `python test_vllm_model.py --list-models`.

### 2) Run scripts
The running scripts are provided in `runscripts/`. To run our method, please use `run_augment_finqa.py`, `run_augment_tatqa.py`, and `run_augment_wikitq.py`. The output will be stored in `results/` and the performance will be printed.
Note: We observe that there might be about 1% random performance variation even if we use greedy decoding. You might try to run the code again if you can't get the number reported in the paper.

## References
If you find our work useful for your research, please consider citing our paper:
```
@misc{liu2024augment,
      title={Augment before You Try: Knowledge-Enhanced Table Question Answering via Table Expansion}, 
      author={Yujian Liu and Jiabao Ji and Tong Yu and Ryan Rossi and Sungchul Kim and Handong Zhao and Ritwik Sinha and Yang Zhang and Shiyu Chang},
      year={2024},
      eprint={2401.15555},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

Our implementation is based on the following repos:
* https://github.com/xlang-ai/Binder
* https://github.com/wenhuchen/Program-of-Thoughts
