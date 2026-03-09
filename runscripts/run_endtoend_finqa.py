import os
import json

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")
TOKENIZER_FALSE = "export TOKENIZERS_PARALLELISM=false\n"
DATASET = 'finqa'
DATASPLIT = "test"
N_SHOTS = 8
TEMPERATURE = 0.0
with open(os.path.join(ROOT_DIR, "vllm_config.json")) as f:
    ENGINE = json.load(f)["model"]
API_FILE = "key.txt"

os.system(fr"""{TOKENIZER_FALSE}python {ROOT_DIR}/scripts/execute_endtoend.py --dataset {DATASET} \
--dataset_split {DATASPLIT} \
--prompt_file templates/prompts/finqa_endtoend_ic.txt \
--output_file endtoend_{DATASET}_{DATASPLIT}_{N_SHOTS}_{ENGINE}_{TEMPERATURE}.json \
--n_processes 4 \
--n_shots {N_SHOTS} \
--max_generation_tokens 128 \
--max_api_total_tokens 16001 \
--temperature {TEMPERATURE} \
--engine {ENGINE} \
--api_config_file {API_FILE}""")
