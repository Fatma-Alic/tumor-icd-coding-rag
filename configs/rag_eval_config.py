import torch
from transformers import BitsAndBytesConfig

EIGHT_BIT = BitsAndBytesConfig(load_in_8bit=True)
FOUR_BIT = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

BATCH_SIZE = 16
DATASET_USAGE = 100

MODEL_CONFIGS = {
    "Qwen2.5-7B": {
        "base_model": "Qwen/Qwen2.5-7B-Instruct-1M",
    },
    "Qwen3-8B": {
        "base_model": "Qwen/Qwen3-8B",
    },
    "Qwen3-14B": {
        "base_model": "Qwen/Qwen3-14B",
        "quantization_config": EIGHT_BIT,
    },
    "Mistral-Nemo": {
        "base_model": "mistralai/Mistral-Nemo-Instruct-2407",
        "quantization_config": EIGHT_BIT,
    },
    "Llama-3.1-8B": {
        "base_model": "meta-llama/Llama-3.1-8B-Instruct",
    },
    "Llama-3.3-70B": {
        "base_model": "meta-llama/Llama-3.3-70B-Instruct",
        "quantization_config": FOUR_BIT,
        "num_gpus": 2,
    },
    "Qwen3-32B" : {
        "base_model": "Qwen/Qwen3-32B",
        "quantization_config": EIGHT_BIT,
    },
    "Qwen3-30B" : {
        "base_model": "Qwen/Qwen3-30B-A3B",
        "quantization_config": BitsAndBytesConfig(
            load_in_8bit=True,
        ),
    },
    "Mixtral-8x7B" : {
        "base_model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "quantization_config":FOUR_BIT,
        "num_gpus": 2,
        #"batch_size" : 10
    },
}

EVAL_CONFIG = {
    "model_dir": "/srv/llms/llit/final_finetune/",
    "output_dir": "/srv/llms/llit/final_eval/",

    "max_retries": 2,

    "datasets": [
        "/srv/llms/llit/GTDS-Dateien/evaluation_data/eval_icd_o.json",
        "/srv/llms/llit/GTDS-Dateien/evaluation_data/eval_bool_icd10.json",
        "/srv/llms/llit/GTDS-Dateien/evaluation_data/eval_icd_10.json",
        "./create_train_dataset/icd_o_dataset.json",
        "./create_train_dataset/bool_icd_10.json",
        "./create_train_dataset/icd_10_dataset_tumor.json"
    ],

    "dataset_names": [
        "icd_o",
        "bool_icd_10",
        "icd_10",
        "icd_o_train",
        "icd_bool_train",
        "icd_10_train"
    ],

    "dataset_usage": DATASET_USAGE,

    "batch_size": BATCH_SIZE

}
