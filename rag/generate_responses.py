import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import Dataset
from typing import Dict, List, Any, Tuple, Optional
from collections import OrderedDict
import logging
import json
import os
import time
import random
from tqdm import tqdm
import gc
import datetime

def configure_logging(log_file: str):
    """Konfiguriert das Logging für eine bestimmte Datei."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode='w'
    )

def remove_system_header(text: str) -> str:
    system_header_indicator = "<|start_header_id|>system<|end_header_id|>"
    message_end_indicator = "<|eot_id|>"
    start_index = text.find(system_header_indicator)
    if start_index == -1:
        return text

    end_index = text.find(message_end_indicator, start_index + len(system_header_indicator))
    if end_index == -1:
        return text

    return text[:start_index] + text[(end_index + len(message_end_indicator)):]

def apply_chat_template(conversation: List[Dict[str, Any]], tokenizer: AutoTokenizer) -> List[Dict[str, Any]]:
    """
    Wandelt eine Liste von Frage-Antwort-Paaren in ein formatiertes Chat-Template um.

    Dabei wird jede Frage als Nachricht mit der Rolle "user" und jede zugehörige Antwort (außer der letzten) als Nachricht
    mit der Rolle "assistant" hinzugefügt. Anschließend wird das vom Tokenizer definierte Chat-Template angewendet.

    Args:
        conversation (List[Dict[str, Any]]): Liste von Dictionaries, die jeweils einen QA-Eintrag enthalten und die Keys "question" und "answer" besitzen.
        tokenizer (AutoTokenizer): Tokenizer, der über die Methode apply_chat_template das finale Format des Chats erstellt.

    Returns:
        List[Dict[str, Any]]: Eine Liste von Nachrichten im Chat-Format.
    """
    messages = []
    if isinstance(conversation, dict) and "questions" in conversation:
        conversation = conversation["questions"]

    elif (isinstance(conversation, list)
         and len(conversation) == 1
         and isinstance(conversation[0], dict)
         and "questions" in conversation[0]):
        conversation = conversation[0]["questions"]

    for i, entry in enumerate(conversation):
            messages.append({"role": "user", "content": entry["question"]})
            if i < len(conversation) - 1:
                messages.append({"role": "assistant", "content": entry["answer"]})
    return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

def load_eval_dataset(eval_data_path: str, tokenizer: AutoTokenizer, dataset_usage_percentage: float = 100.0) -> Dataset:
    """
    Lädt und verarbeitet einen Evaluationsdatensatz für die Modellbewertung.

    Args:
        eval_data_path (str): Pfad zur JSON-Datei des Evaluierungsdatensatzes.
        tokenizer (AutoTokenizer): Tokenizer zur Verarbeitung der Daten.
        dataset_usage_percentage (float, optional): Prozentualer Anteil der zu verwendenden Daten. Standard ist 100%.

    Returns:
        Dataset: Das verarbeitete Evaluierungsdatensatzobjekt.
    """
    # Setze padding_side auf 'left' für decoder-only Modelle
    tokenizer.padding_side = 'left'

    # Datensatz laden
    try:
        with open(eval_data_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception as e:
        logging.error(f"Fehler beim Laden des Datensatzes: {e}")
        print(f"Fehler beim Laden des Datensatzes: {e}")
        return Dataset.from_dict({})
    if not data:
        logging.warning("Der Datensatz ist leer")
        print("Der Datensatz ist leer")
        return Dataset.from_dict({})

    is_x_shot = isinstance(data[0], list)
    # Falls 1-shot: Jedes einzelne QA-Paar in eine Liste
    if not is_x_shot:
        data = [[entry] for entry in data]

    # Datensatzgröße anpassen
    print(f"Es werden {dataset_usage_percentage}% des Datensatzes verwendet")
    if dataset_usage_percentage < 100.0:
        sample_size = max(1, int(len(data) * dataset_usage_percentage / 100))
        data = random.sample(data, sample_size)
        logging.info(f"Reduzierte Datensatzgröße: {len(data)} Sätze.")
        print(f"Reduzierte Datensatzgröße: {len(data)} Sätze.")

    # "qa" : originale Question-Answer wie im JSON
    # "qa_chat": Question-Answer im Chat-Format ohne letzte Antort
    dataset = [{"qa" : conversation, "qa_chat": remove_system_header(apply_chat_template(conversation, tokenizer))} for conversation in data]

    # Erstellen des Datensatzobjekts
    eval_dataset = Dataset.from_list(dataset)
    logging.info(f"Evaluationsdatensatz erstellt: {len(eval_dataset)} Einträge.")
    print(f"Evaluationsdatensatz erstellt: {len(eval_dataset)} Einträge.")
    return eval_dataset

def prepare_batch_data(batch: Dict[str, Any], tokenizer: AutoTokenizer, eval_data_name: str, rewrite_prompts: bool = True) -> Tuple[List[str], List[Any]]:
    """
    Bereitet Batch-Daten für die Evaluation vor.

    Args:
        batch (Dict[str, Any]): Batch-Daten im Format {'qa_chat': [chat_prompts], 'qa': [Konversationen]}.
        tokenizer (AutoTokenizer): Tokenizer für die Textformatierung.
        eval_data_name (str): Name des Evaluationsdatensatzes, ggf. um spezifische Instruktionen hinzuzufügen.
        rewrite_prompts (bool, optional): Flag, ob die Chat-Prompts (und die zugehörigen QA-Einträge) modifiziert werden sollen. Standard ist False.

    Returns:
        Tuple[List[str], List[Any]]: Ein Tuple, wobei das erste Element die (ggf. modifizierten) Chat-Prompts (qa_chat) und das zweite Element die zugehörigen QA-Konversationen (qa) enthält.
    """
    qa_chats = batch["qa_chat"]
    qa = batch["qa"]

    if not rewrite_prompts:
        return qa_chats, qa

    else:
        instruction = ""
        """
        elif eval_data_name in ['icd_o', 'icd_o_train']:
            instruction = " Antworte nur kurz mit dem ICD-O-Topographie-Code."
        elif eval_data_name in ['bool_icd_10', 'icd_bool_train']:
            instruction = " Antworte nur kurz mit Ja oder Nein."
        else:
        """


        formatted_qa_chat = []
        for i, conversation in enumerate(qa):
            # conversation in eine flache Liste von {question, answer} bringen
            if isinstance(conversation, list) and len(conversation) == 1 and isinstance(conversation[0], dict) and "questions" in conversation[0]:
                conv_list = conversation[0]["questions"]
            elif isinstance(conversation, dict) and "questions" in conversation:
                conv_list = conversation["questions"]
            else:
                conv_list = conversation  # bereits Liste von QA-Dicts

            if not conv_list:
                # Leerer Fall absichern
                formatted_qa_chat.append(tokenizer.apply_chat_template(
                    [{"role":"user","content":"?"}], tokenize=False, add_generation_prompt=True
                ))
                continue

            # 1) letzte Frage mit Instruktion erweitern
            conv_list = [dict(x) for x in conv_list]  # defensive copy
            conv_list[-1]["question"] = conv_list[-1]["question"] + instruction

            # 2) Chat-Nachrichten aufbauen: alle QAs bis auf die letzte Antwort,
            #    danach die letzte Frage (ohne Antwort) – genau wie in apply_chat_template()
            messages = []
            for j, entry in enumerate(conv_list):
                messages.append({"role": "user", "content": entry["question"]})
                if j < len(conv_list) - 1:
                    messages.append({"role": "assistant", "content": entry["answer"]})

            # 3) Template anwenden (mit Generation Prompt), optional ohne remove_system_header
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            formatted_qa_chat.append(prompt)

            # 4) QA-Objekt in derselben Struktur updaten (damit "qa" konsistent bleibt)
            if isinstance(qa[i], list) and len(qa[i]) == 1 and isinstance(qa[i][0], dict) and "questions" in qa[i][0]:
                qa[i][0]["questions"] = conv_list
            elif isinstance(qa[i], dict) and "questions" in qa[i]:
                qa[i]["questions"] = conv_list
            else:
                qa[i] = conv_list

        qa_chats = formatted_qa_chat

    return qa_chats, qa

"""
def prepare_model_inputs(prompts: List[str], tokenizer: AutoTokenizer, device: torch.device) -> Dict[str, torch.Tensor]:

    Bereitet die Eingaben für das Modell vor.

    Args:
        prompts (List[str]): Liste von formatierten Chat-Prompts.
        tokenizer (AutoTokenizer): Der Tokenizer für die Textverarbeitung.
        device (torch.device): Das Ziel-Device auf welches man die Tensoren schiebt.

    Returns:
        Dict[str, torch.Tensor]: Ein Dictionary aus Input Tensoren (tokenisierten Eingaben) für das Modell.

    assert tokenizer.padding_side == "left", f"padding_side ist {tokenizer.padding_side}"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Fallback für leere Prompts (wie schon bei dir)
    fallback = tokenizer.apply_chat_template([{"role":"user","content":"?"}],
                                             tokenize=False, add_generation_prompt=True)

    bos = getattr(tokenizer, "bos_token", None) or " "
    clean_prompts = []
    for p in prompts:
        s = p if isinstance(p, str) else str(p)
        if not s.strip():
            s = bos
        clean_prompts.append(s)

    max_len = getattr(tokenizer, "model_max_length", 8192)

    inputs = tokenizer(
        clean_prompts,
        return_tensors="pt",
        padding='longest', #True
        truncation=False, #Wenn auf False, dann wird es nicht mehr abgeschnitten
        max_length=max_len,
    )

    overlong_count = 0
    for i, p in enumerate(clean_prompts):
        tok_len = len(tokenizer.encode(p, add_special_tokens=False))
        if tok_len > max_len:
            overlong_count += 1
            logging.warning(
                f"Prompt {i} ist länger ({tok_len} Tokens) als max_length={max_len} und wurde gekürzt."
            )
            print(
                f"⚠️ Warnung: Prompt {i} ist länger ({tok_len} Tokens) als max_length={max_len} und wurde gekürzt."
            )

    if overlong_count > 0:
        logging.warning(f"Insgesamt {overlong_count} Prompt(s) wurden wegen Überschreitung von max_length gekürzt.")
        print(f"⚠️ Gesamtwarnung: {overlong_count} Prompt(s) wurden wegen Überschreitung von max_length gekürzt.")

    # *** Kern-Fix: letzter Token darf NICHT pad_token_id (== eos) sein ***
    ii = inputs["input_ids"]
    am = inputs["attention_mask"]

    # Wähle einen harmlosen Nicht-PAD-Token als "Füller" (z. B. ein Space-Token)
    filler_ids = tokenizer.encode(" ", add_special_tokens=False)
    filler_id = filler_ids[0] if len(filler_ids) > 0 else (tokenizer.bos_token_id or 0)

    mask_last_is_pad = (ii[:, -1] == tokenizer.pad_token_id)
    if mask_last_is_pad.any():
        ii[mask_last_is_pad, -1] = filler_id  # ersetze nur den letzten EOS==PAD-Token
        # attention_mask bleibt 1 am Ende (rechts), d. h. KEIN Right-Pad

    # Sanity-Check: rechte Spalte darf nicht 0 sein (keine Right-Pad-Maske)
    assert (am[:, -1] == 1).all(), "Right padding oder leeres Prompt im Batch!"


    return {k: v.to(device) for k, v in inputs.items()}
"""

def prepare_model_inputs(prompts: List[str], tokenizer: AutoTokenizer, device: torch.device, max_ctx: Optional[int] = None) -> Dict[str, torch.Tensor]:
    assert tokenizer.padding_side == "left", f"padding_side ist {tokenizer.padding_side}"
    #if tokenizer.pad_token is None:
    #tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if max_ctx is None:
        max_ctx = getattr(getattr(tokenizer, "model", None), "config", None)
        max_ctx = getattr(max_ctx, "max_position_embeddings", None) or getattr(tokenizer, "model_max_length", 131072)

    bos = getattr(tokenizer, "bos_token", None) or " "
    clean_prompts = [(p if isinstance(p, str) else str(p)).strip() or bos for p in prompts]

    # Vorab-Längencheck (ohne Trunkierung)
    too_long = []
    for i, s in enumerate(clean_prompts):
        ids = tokenizer(s, add_special_tokens=False).input_ids
        if ids and len(ids) > max_ctx:
            too_long.append((i, len(ids)))

    if too_long:
        for i, L in too_long:
            msg = f"⚠️ Prompt {i} ist zu lang: {L} Tokens > Kontextfenster {max_ctx}. " \
                  f"Beispiele/Chunks reduzieren. (Keine Trunkierung durchgeführt.)"
            logging.warning(msg); print(msg)
        # harte Variante (empfohlen, wenn *alle* Beispiele drin bleiben müssen):
        raise ValueError(f"{len(too_long)} Prompt(s) überschreiten das Kontextfenster {max_ctx}.")

    # Tokenisieren ohne Trunkierung
    inputs = tokenizer(
        clean_prompts,
        return_tensors="pt",
        padding="longest",
        truncation=False
    )

    return {k: v.to(device) for k, v in inputs.items()}


def load_model(base_model_name: str, epoch_dir: str, quantization_config=None) -> PeftModel:
    """
    Lädt ein Basismodell und zugehörigen PEFT-Adapter aus angegebener Epoche.

    Args:
        base_model_name (str): Name des vortrainierten Basismodells.
        epoch_dir (str): Pfad zum Verzeichnis der Epoche mit dem Adapter.

    Returns:
        PeftModel: Das geladene PEFT-Modell.

    """
    gc.collect()
    torch.cuda.empty_cache()
    try:
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=quantization_config, device_map="auto")
        model = PeftModel.from_pretrained(base_model, epoch_dir)
        return model
    except Exception as e:
        logging.error(f"Fehler beim Laden des Modells aus {epoch_dir}: {e}")
        print(f"Fehler beim Laden des Modells aus {epoch_dir}: {e}")
        raise RuntimeError(f"Modell konnte nicht geladen werden: {e}")

def generate_model_responses(model: PeftModel, inputs: Dict[str, torch.Tensor], tokenizer: AutoTokenizer, generation_config: Dict[str, Any]) -> List[str]:
    """
    Generiert Modellantworten vom Modell anhand der Input Tensoren.

    Args:
        model (PeftModel): Das Modell mit dem Antworten generiert werden sollen.
        inputs (Dict[str, torch.Tensor]): Input Tensoren für das Modell.
        tokenizer (AutoTokenizer): Tokenizer um Antworten zu decoden.
        generation_config (Dict[str, Any]): Konfigurationen für Generierungs Parameter.

    Returns:
        List[str]: Liste der generierten Antworten.

    """
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **generation_config
        )

    answers = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return answers

def generate_epoch_responses(model: PeftModel, tokenizer: AutoTokenizer,
                             eval_data: Dataset, eval_data_name: str,
                             generation_config: Dict[str, Any], batch_size: int,
                             rewrite_prompts: bool = True) -> List[Dict[str, str]]:
    """
    Generiert Antworten für eine gesamte Modell-Epoche.

    Args:
        model (PeftModel): Das Modell, mit dem die Antworten generiert werden sollen.
        tokenizer (AutoTokenizer): Tokenizer für die Textformatierung.
        eval_data (Dataset): Der Evaluationsdatensatz.
        eval_data_name (str): Name des Evaluationsdatensatzes.
        generation_config (Dict[str, Any]): Konfigurationsparameter für die Textgenerierung.
        batch_size (int): Anzahl der Beispiele pro Batch.
        rewrite_prompts (bool, optional): Flag, ob die Chat-Prompts modifiziert werden sollen. Standard ist False.

    Returns:
        List[Dict[str, str]]: Eine Liste von Dictionaries, die die Keys 'qa', 'true_answer' und 'generated_answer' enthalten.
    """

    responses = []
    # Texte tokenisieren und Längen vorab berechnen
    text_lengths = [(i, len(tokenizer.encode(item['qa_chat'])))
                   for i, item in enumerate(eval_data)]

    # Nach Länge sortieren, aber Index behalten
    sorted_indices = [idx for idx, _ in sorted(text_lengths, key=lambda x: x[1], reverse=True)]

    for i in tqdm(range(0, len(sorted_indices), batch_size), desc="Generierung", unit="batches"):
        batch = [eval_data[idx] for idx in sorted_indices[i:i+batch_size]]
        qa_chats, conversation = prepare_batch_data({"qa_chat": [b["qa_chat"] for b in batch], "qa": [b["qa"] for b in batch]}, tokenizer, eval_data_name, rewrite_prompts)
        # Kontextgrenze vom Modell ziehen
        max_ctx = getattr(getattr(model, "config", None), "max_position_embeddings", None)
        #print("max_position_embeddings:", max_ctx)
        #print("tokenizer.model_max_length:", tokenizer.model_max_length)
        #safety_margin = 16
        #max_new_tokens = 20
        #max_input_ctx = max_ctx - max_new_tokens - safety_margin


        inputs = prepare_model_inputs(qa_chats, tokenizer, model.device, max_ctx=max_ctx)

        answers = generate_model_responses(model, inputs, tokenizer, generation_config)

        for chat, conv, generated in zip(qa_chats, conversation, answers):
            if isinstance(conv, dict) and "questions" in conv:
                qa_list = conv["questions"]
            elif isinstance(conv, list) and len(conv) == 1 and isinstance(conv[0], dict) and "questions" in conv[0]:
                qa_list = conv[0]["questions"]
            elif isinstance(conv, dict):
                qa_list = [conv]
            else:
                qa_list = conv
            responses.append({
                "qa": [OrderedDict([("question", qa["question"]), ("answer", qa["answer"])]) for qa in qa_list],
                "true_answer": qa_list[-1]['answer'],
                "generated_answer": generated
            })
    print("[tok] in generate_epoch_responses in generate epoch:", id(tokenizer), tokenizer.padding_side)

    true_answer = str(qa_list[-1]["answer"]).strip()
    if true_answer and true_answer in chat:
        print("❌ LEAK: true_answer ist im Prompt enthalten!")
        print("TRUE_ANSWER:", true_answer)
        print("PROMPT (letzte 600 Zeichen):", chat[-600:])
        #raise RuntimeError("Prompt enthält die Ground-Truth-Antwort → Evaluation ungültig!")
    return responses

def save_responses(responses: List[Dict[str, str]], output_dir: str, model_key: str, dataset_name: str, result_file_name: str) -> None:
    """
    Speichert die generierten Antworten einer Epoche in einem spezifischen Modellverzeichnis.

    Args:
        responses (List[Dict[str, str]]): Liste mit Antwort-Dictionaries.
        output_dir (str): Basisverzeichnis für die Ausgabedateien.
        model_key (str): Name des Modells (wird für den Verzeichnisnamen verwendet).
        dataset_name (str): Name des Datensatzes, für den die Antworten generiert werden.
        result_file_name (str): Name der Ergebnisdatei, die in der Verzeichnisstruktur erzeugt wird.

    Returns:
        None
    """
    out_path = os.path.join(output_dir, model_key, dataset_name)
    os.makedirs(out_path, exist_ok=True)
    with open(os.path.join(out_path, result_file_name), 'w', encoding='utf-8') as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)

def response_file_exists(output_dir: str, model_key: str, base_model_name: str, eval_data_name: str, result_file_name:str) -> bool:
    """
    Überprüft, ob bereits eine Antwortdatei für eine bestimmte Epoche existiert.

    Args:
        output_dir (str): Basisverzeichnis für die Speicherung.
        base_model_name (str): Name des Modells.
        eval_data_name (str): Name des Evaluationsdatensatzes.
        result_file_name (str): Name der Antwortdatei im Ordner.

    Returns:
        bool: True, wenn die Datei existiert, sonst False.
    """
    path = os.path.join(output_dir, model_key, eval_data_name, result_file_name)
    exists = os.path.exists(path)
    if exists:
        logging.info(f"Result file {result_file_name} already exists. Skipping.")
        print(f"Result file {result_file_name} already exists. Skipping.")
    return exists

def generate_and_save_responses_base_model(model_key: str, base_model_name: str, eval_data: Dict[str, Any],
                                           generation_config: Dict[str, Any],
                                           output_dir: str, result_file_name:str,
                                           eval_data_name: str, batch_size:int,
                                           rewrite_prompts:bool = True, quantization_config=None) -> float:
    """
    Generiert und speichert Antworten für ein Basismodell.

    Args:
        base_model_name (str): Name des Basis-Modells.
        eval_data (Dict[str, Any]): Tokenizer und Evaluationsdaten.
        generation_config (Dict[str, Any]): Konfiguration für die Textgenerierung.
        output_dir (str): Pfad zum Speichern der generierten Antworten.
        result_file_name (str): Name der Ergebnisdatei.
        eval_data_name (str): Name des Evaluationsdatensatzes.
        batch_size (int): Größe der Batches für die Inferenz.
        rewrite_prompts (bool, optional): Flag, ob die Chat-Prompts modifiziert werden sollen. Standard ist False.

    Returns:
        float: Laufzeit in Sekunden .
    """
    assert result_file_name.startswith("responses_epoch_"), f"Ungültiger Dateiname: {result_file_name}." + "Sollte heißen: 'responses_epoch_{i}'"
    if response_file_exists(output_dir, model_key, base_model_name, eval_data_name, result_file_name):
        return 0

    start_time = time.time()
    torch.cuda.empty_cache()
    logging.info(f"Evaluiere Basismodell [{base_model_name}]")
    print(f"Evaluiere Basismodell [{base_model_name}]")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=quantization_config, device_map="auto")
    except Exception as e:
        logging.info(f"Fehler beim Laden des Basis-Modells: {e}")
        raise RuntimeError(f"Modell konnte nicht geladen werden: {e}")

    responses_epoch_0 = generate_epoch_responses(base_model, eval_data['tokenizer'], eval_data['data'],
                                                 eval_data_name, generation_config, batch_size,
                                                 rewrite_prompts=rewrite_prompts)
    save_responses(responses_epoch_0, output_dir, model_key, eval_data_name, result_file_name)

    del base_model
    torch.cuda.empty_cache()

    return time.time() - start_time


def generate_and_save_responses_peft_model(model_key: str, base_model_name: str, model_dir: str, eval_data: Dict[str, Any], generation_config: Dict[str, Any], output_dir: str, result_file_name:str, eval_data_name: str, batch_size: int, quantization_config=None) -> float:
    """
    Generiert und speichert Antworten für ein PEFT-trainiertes Modell für ein gegebenes Evaluierungs-Dataset.

    Args:
        base_model_name (str): Name des Basismodells.
        model_dir (str): Pfad zum Verzeichnis des PEFT-Adapters (Checkpoint).
        eval_data (Dict[str, Any]): Dictionary mit 'tokenizer' und 'data' (Evaluationsdaten).
        generation_config (Dict[str, Any]): Konfiguration für die Textgenerierung.
        output_dir (str): Verzeichnis, in dem die generierten Antworten gespeichert werden.
        result_file_name (str): Name der Ergebnisdatei.
        eval_data_name (str): Name des Evaluationsdatensatzes.
        batch_size (int): Batchgröße für die Inferenz.

    Returns:
        float: Laufzeit in Sekunden.
    """

    assert result_file_name.startswith("responses_epoch_"), f"Ungültiger Dateiname: {result_file_name}." + "Sollte heißen: 'responses_epoch_{i}'"
    if response_file_exists(output_dir, model_key, base_model_name, eval_data_name, result_file_name):
        return 0

    start_time = time.time()
    try:
        gc.collect()
        torch.cuda.empty_cache()
        model = load_model(base_model_name, model_dir, quantization_config)
    except Exception as e:
        logging.error(f"Fehler beim Laden des Modells aus Verzeichnis {model_dir}")

    responses = generate_epoch_responses(
    model, eval_data['tokenizer'], eval_data['data'], eval_data_name,
    generation_config, batch_size, rewrite_prompts=True
    )

    save_responses(responses, output_dir, model_key, eval_data_name, result_file_name)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return time.time() - start_time

def generate_and_save_responses_peft_epochs(model_key:str, base_model_name: str, model_dir: str, eval_data: Dict[str, Any], generation_config: Dict[str, Any], output_dir: str, eval_data_name: str, batch_size: int, qunatization_config=None) -> Dict[int, float]:
    """
    Generiert und speichert Antworten für alle PEFT-angepassten Epochen eines Modells.

    Args:
        base_model_name (str): Name des Basismodells.
        model_dir (str): Verzeichnis mit den gespeicherten PEFT-Checkpoints.
        eval_data (Dict[str, Any]): Tokenizer und Evaluationsdaten.
        generation_config (Dict[str, Any]): Generierungskonfiguration.
        output_dir (str): Speicherort für die Ergebnisse.
        eval_data_name (str): Name des Evaluationsdatensatzes.
        batch_size (int): Batchgröße.

    Returns:
        Dict[int, float]: Laufzeit in Sekunden.
    """
    epoch_times = {}
    checkpoints = sorted([os.path.join(model_dir, d) for d in os.listdir(model_dir) if d.startswith("checkpoint-")], key=lambda x: int(x.split("-")[-1]))

    for i, checkpoint in enumerate(checkpoints, 1):
        logging.info(f"Evaluiere Modell aus Epoche {i} [{checkpoint}]")
        print(f"Evaluiere Modell aus Epoche {i} [{checkpoint}]")
        result_file_name = f"responses_epoch_{i}.json"
        epoch_time = generate_and_save_responses_peft_model(
                model_key, base_model_name, checkpoint, eval_data, generation_config, output_dir, result_file_name, eval_data_name, batch_size, qunatization_config)

        # Logge die Zeit für die aktuelle Epoche
        logging.info(f"Epoche {i} abgeschlossen. Benötigte Zeit: {epoch_time:.2f} Sekunden")
        print(f"Epoche {i} abgeschlossen. Benötigte Zeit: {epoch_time:.2f} Sekunden")

        epoch_times[i] = epoch_time

    return epoch_times


def save_total_time(output_dir: str, model_key: str, dataset_name: str, total_seconds: float) -> str:
    """
    Speichert NUR die Gesamtzeit als einzelne Datei:
    <output_dir>/<model_key>/<eval_data_name>/timings/total_time.json
    """
    timings_dir = os.path.join(output_dir, model_key, dataset_name, "timings")
    os.makedirs(timings_dir, exist_ok=True)

    payload = {
        "total_time_s": round(float(total_seconds), 6),
        "total_time_min": round(float(total_seconds) / 60.0, 6),
        "total_time_h": round(float(total_seconds) / 3600.0, 6),
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds")
    }
    out_path = os.path.join(timings_dir, "total_time.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[TIMING] Gesamtzeit gespeichert: {out_path}")
    logging.info(f"[TIMING] Gesamtzeit gespeichert: {out_path}")
    return out_path


def generate_and_save_responses(model_key: str, base_model_name: str, model_dir: str, eval_data_path: str, eval_data_name: str, output_dir: str, dataset_usage_percentage: float = 100.0, batch_size: int = 16, quantization_config=None) -> None:
    """
    Generiert und speichert Antworten auf Basis eines Modells für ein gegebenes Evaluierungs-Dataset.

    Args:
        base_model_name (str): Der Name des Basismodells.
        model_dir (str): Verzeichnis, das die Modell-Checkpoints enthält.
        eval_data_path (str): Pfad zum Evaluierungs-Dataset.
        eval_data_name (str): Name des Evaluierungs-Datasets.
        output_dir (str): Verzeichnis, in dem die generierten Antworten gespeichert werden.
        dataset_usage_percentage (float, optional): Der Prozentsatz des Datasets, der verwendet werden soll. Standardwert ist 100.0.
        batch_size (int, optional): Die Batchgröße für die Antwortgenerierung. Standardwert ist 16.

    Returns:
        None
    """
    torch.cuda.empty_cache()

    # Checke zuerst, ob bereits für alle Epochen ein response file existiert, wenn ja überspringe kompletten Datenstaz
    epoch_dirs = [os.path.join(model_dir, d) for d in os.listdir(model_dir) if d.startswith("checkpoint-")]
    epoch_dirs = sorted(epoch_dirs, key=lambda x: int(x.split("-")[-1]))

    # Tokenizer laden & konfigurieren
    #tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side ='left')
    #tokenizer.pad_token = tokenizer.eos_token
    #tokenizer.padding_side = 'left' # Setze padding_side auf 'left' für decoder-only Modelle
    #tokenizer.truncation_side = 'left'# damit bleiben die aktuelle Frage + Beispiele am Ende erhalten!
    #Das EOS-PAD muss gesetzt werden, BEVOR tokenisiert wird. Sonst werden Prompts falsch gepadded oder Attention Masks falsch
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    # 🔒 EOS = PAD (kritisch!)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # Decoder-only korrekt
    tokenizer.padding_side = "left"

    # Datensatz laden
    eval_data = {'tokenizer': tokenizer, 'data': load_eval_dataset(eval_data_path, tokenizer, dataset_usage_percentage)}

    generation_config = {
        "max_new_tokens": 30,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
        "temperature": None,
        "top_p": None,
        "top_k": None,
    }

    # Basismodell
    result_file_name_epoch_0 = "responses_epoch_0.json"
    epoch_0_time = generate_and_save_responses_base_model(model_key, base_model_name, eval_data, generation_config,
                                                          output_dir, result_file_name_epoch_0,
                                                          eval_data_name, batch_size, rewrite_prompts=True, quantization_config=quantization_config)

    # Logge die Zeit für die aktuelle Epoche
    logging.info(f"Epoche 0 abgeschlossen. Benötigte Zeit: {epoch_0_time:.2f} Sekunden")
    print(f"Epoche 0 abgeschlossen. Benötigte Zeit: {epoch_0_time:.2f} Sekunden")

    # Trainierte Peft-Modelle
    epoch_times = {0: epoch_0_time}
    epoch_times.update(generate_and_save_responses_peft_epochs(model_key, base_model_name, model_dir, eval_data, generation_config, output_dir, eval_data_name, batch_size, quantization_config))

    total_time = sum(epoch_times.values())
    logging.info("### Gesamtzeit für alle Epochen ###")
    print("\n### Gesamtzeit für alle Epochen ###")
    for epoch, epoch_time in sorted(epoch_times.items()):
        logging.info(f"Epoche {epoch}: {epoch_time:.2f} Sekunden")
        print(f"Epoche {epoch}: {epoch_time:.2f} Sekunden")
    logging.info(f"Total: {total_time:.2f} Sekunden = {total_time/60:.2f} Minuten = {total_time/60/60:.2f} Stunden")

    save_total_time(output_dir, model_key, eval_data_name, total_time)

def eval_model_on_multiple_datasets(model_key:str, base_model: str, model_dir: str, eval_datasets: List[str], eval_data_names: List[str], output_dir: str, dataset_usage_percentage: float = 100.0, batch_size: int = 16, quantization_config: Optional[BitsAndBytesConfig] = None) -> None:
    """
    Evaluiert ein Modell auf mehreren Datasets und generiert Antworten für jedes Dataset.

    Args:
        base_model (str): Der Name des Basismodells.
        model_dir (str): Verzeichnis, das die Modell-Checkpoints enthält.
        eval_datasets (List[str]): Liste der Pfade zu den Evaluierungs-Datasets.
        eval_data_names (List[str]): Liste der Namen der Evaluierungs-Datasets.
        output_dir (str): Verzeichnis, in dem die generierten Antworten gespeichert werden.
        dataset_usage_percentage (float, optional): Der Prozentsatz des Datasets, der verwendet werden soll. Standardwert ist 100.0.
        batch_size (int, optional): Die Batchgröße für die Antwortgenerierung. Standardwert ist 16.

    Returns:
        None
    """
    for dataset, dataset_name in zip(eval_datasets, eval_data_names):
        # Logging für den aktuellen Datensatz konfigurieren
        log_file = os.path.join(output_dir, model_key, dataset_name, f"evaluation.log")
        configure_logging(log_file)

        logging.info("\nStarte Antwortgenerierung mit folgenden Parametern:")
        logging.info(f"Basismodell: {base_model}")
        logging.info(f"Modell-Verzeichnis: {model_dir}")
        logging.info(f"Evaluierungsdatensatz: {dataset}")
        logging.info(f"Ausgabeverzeichnis: {output_dir}")
        logging.info(f"Datensatz-Nutzung: {dataset_usage_percentage}%")
        logging.info(f"Batch-Größe: {batch_size}")
        print("\nStarte Antwortgenerierung mit folgenden Parametern:")
        print(f"Basismodell: {base_model}")
        print(f"Modell-Verzeichnis: {model_dir}")
        print(f"Evaluierungsdatensatz: {dataset}")
        print(f"Ausgabeverzeichnis: {output_dir}")
        print(f"Datensatz-Nutzung: {dataset_usage_percentage}%")
        print(f"Batch-Größe: {batch_size}")
        print()
        generate_and_save_responses(model_key, base_model, model_dir, dataset, dataset_name, output_dir, dataset_usage_percentage, batch_size, quantization_config)