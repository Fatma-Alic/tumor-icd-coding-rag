import gc
import os
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from configs.rag_eval_config import FOUR_BIT
from generate_responses import (
    generate_and_save_responses_base_model,
    generate_and_save_responses_peft_model,
    load_eval_dataset,
    save_total_time,
)
from rag.analyze_multiple_responses_with_F1 import (
    analyze_multiple_responses_on_model,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def cuda_soft_cleanup() -> None:
    """
    Parameters:
        None

    Returns:
        None
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=False,
)


def load_tokenizer(model_name: str):
    """
    Load a tokenizer and set padding settings.

    Parameters:
        model_name (str): Name of the tokenizer model.

    Returns:
        AutoTokenizer: Loaded tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def rag_evaluation_base_model(eval_data_path: str) -> None:
    """
    Evaluate generated questions from the classifier model
    with the untrained Llama model.

    Parameters:
        eval_data_path (str): Path to the evaluation dataset.

    Returns:
        None
    """
    # Tokenization and data preparation
    tokenizer = load_tokenizer("meta-llama/Llama-3.3-70B-Instruct")
    # tokenizer = load_tokenizer("meta-llama/Llama-3.1-8B-Instruct")

    eval_data = {
        "tokenizer": tokenizer,
        "data": load_eval_dataset(
            eval_data_path,
            tokenizer,
            dataset_usage_percentage=100,
        ),
    }

    generation_config = {
        "max_new_tokens": 20,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
        "temperature": None,
        "top_p": None,
        "top_k": None,
    }

    """
    max_length: defines the maximum length of the generated text
    temperature: controls the randomness of text generation
    top_p / nucleus sampling: controls token selection during text generation
    """

    file_stem = Path(eval_data_path).stem
    output_dir = (
        "./ICDO/Classifier/Tumordiagnose/Extended_Label/"
        f"Llama-3.3-70B/LLM_base/Filtered/{file_stem}"
    )
    # output_dir = (
    #     f"./Zero-shot-prompting/Llama-3.1-8B/LLM_base/Filtered/{file_stem}"
    # )
    os.makedirs(output_dir, exist_ok=True)

    result = generate_and_save_responses_base_model(
        # model_key="Llama-3.1-8B",
        model_key="Llama-3.3-70B",
        # base_model_name="meta-llama/Llama-3.1-8B-Instruct",
        base_model_name="meta-llama/Llama-3.3-70B-Instruct",
        eval_data=eval_data,
        generation_config=generation_config,
        output_dir=output_dir,
        result_file_name="responses_epoch_0.json",
        eval_data_name=file_stem,
        batch_size=64,
        rewrite_prompts=True,
        quantization_config=FOUR_BIT,
    )

    save_total_time(
        output_dir,
        model_key="Llama-3.3-70B",
        dataset_name=file_stem,
        total_seconds=result,
    )
    print(f"Generation completed for {eval_data_path}")


def load_peft_model(base_model_name: str, model_dir: str):
    """
    Load a PEFT model based on a base model.

    Parameters:
        base_model_name (str): Name of the base model.
        model_dir (str): Path to the PEFT checkpoint folder.

    Returns:
        Model or None: Loaded model if successful, otherwise None.
    """
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            # device_map="balanced_low_0",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            quantization_config=None,
        )
        model = PeftModel.from_pretrained(
            base_model,
            model_dir,
        )  # uses only the adapter weights to save space
        model.eval()
        print(base_model.hf_device_map)
        return base_model
    except Exception as error:
        print(f"Error while loading the PEFT model: {error}")
        return None


def rag_evaluation_peft_model(eval_data_path: str) -> None:
    """
    Evaluate generated questions from the classifier model
    with the trained Llama model.

    Parameters:
        eval_data_path (str): Path to the evaluation dataset.

    Returns:
        None
    """
    tokenizer = load_tokenizer("meta-llama/Llama-3.3-70B-Instruct")
    # tokenizer = load_tokenizer("meta-llama/Llama-3.1-8B-Instruct")

    eval_data = {
        "tokenizer": tokenizer,
        "data": load_eval_dataset(
            eval_data_path,
            tokenizer,
            dataset_usage_percentage=100,
        ),
    }
    print(
        "[tok] in generate_epoch_responses in main:",
        id(tokenizer),
        tokenizer.padding_side,
    )

    generation_config = {
        "max_new_tokens": 20,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
        "temperature": None,
        "top_p": None,
        "top_k": None,
    }

    file_stem = Path(eval_data_path).stem
    # output_dir = f"./Word2Vec_output/Llama-3.3-70B/LLM_peft/{file_stem}"
    output_dir = (
        "./ICDO/Classifier/Tumordiagnose/Extended_Label/"
        f"Llama-3.3-70B/LLM_peft/Filtered/{file_stem}"
    )
    os.makedirs(output_dir, exist_ok=True)

    result = generate_and_save_responses_peft_model(
        # model_key="Llama-3.1-8B",
        model_key="Llama-3.3-70B",
        base_model_name="meta-llama/Llama-3.3-70B-Instruct",
        # base_model_name="meta-llama/Llama-3.1-8B-Instruct",
        # model_dir="/srv/llms/llit/250301_finetune_llama3.1-8B/checkpoint-172706/",
        model_dir="/srv/llms/llit/final_finetune/Llama-3.3-70B/checkpoint-129529/",
        eval_data=eval_data,
        generation_config=generation_config,
        output_dir=output_dir,
        result_file_name="responses_epoch_0.json",
        eval_data_name=file_stem,
        batch_size=64,
        quantization_config=FOUR_BIT,
    )

    save_total_time(
        output_dir,
        model_key="Llama-3.3-70B",
        dataset_name=file_stem,
        total_seconds=result,
    )
    print(
        "[tok] in generate_epoch_responses in main:",
        id(tokenizer),
        tokenizer.padding_side,
    )


if __name__ == "__main__":
    cuda_soft_cleanup()
    QUESTIONS_DIR= "/home/alic/RAG/prompts"
    BASE_DIR="/home/alic/RAG/results"

    # For generating responses
    for root, dirs, files in os.walk(QUESTIONS_DIR):
        for file in files:
            if file.endswith("_questions.json"):
                path = os.path.join(root, file)
                print(f"-> Evaluating: {path}")
                try:
                    print(path)
                    rag_evaluation_base_model(path)
                    rag_evaluation_peft_model(path)
                except Exception as error:
                    print(f"Error in {file}: {error}")

    # For evaluating responses
    for modelname in os.listdir(BASE_DIR):
        model_path = os.path.join(BASE_DIR, modelname)
        llm_path = os.path.join(model_path, "Llama-3.3-70B")

        if os.path.isdir(llm_path):
            print(f"LLM folder found: {llm_path}")
            try:
                result_path = (
                    "/home/alic/LLIT-RAG/ICDO/Random/Tumordiagnose/Prompt_check/"
                    "Llama-3.3-70B/LLM_peft/Filtered/results"
                )
                analyze_multiple_responses_on_model(llm_path, result_path)
                print("Saved")
            except Exception as error:
                print(f"Error in {llm_path}: {error}")
