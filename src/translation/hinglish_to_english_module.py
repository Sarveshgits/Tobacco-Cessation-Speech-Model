import os
import time
import sys
import torch
import argparse
import re
# Allow unpickling of argparse.Namespace (DANGER: Only do this if you trust the source)
torch.serialization.add_safe_globals([argparse.Namespace])

# Add IndicTrans2 to Python path for local development

sys.path.append("/home/sarveshkumar/STT")


from ai4bharat.transliteration import XlitEngine
from langdetect import detect
import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoTokenizer
from IndicTransToolkit.IndicTransToolkit.processor import IndicProcessor

# Set Hugging Face cache path for faster reuse across reboots/runs
os.environ["HF_HOME"] = "../../cache/model_cache"

# Initialize transliteration engine for Hinglish -> Hindi
xlit_engine = XlitEngine("hi", beam_width=4, rescore=True, src_script_type="en")

# Translation setup
BATCH_SIZE = 4

# Check and use GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# If you want 4-bit or 8-bit quantization on GPU to save VRAM, set below
quantization = None  # Options: None, "4-bit", "8-bit"

# Using smaller distil version for faster GPU loading
indic_en_ckpt_dir = "ai4bharat/indictrans2-indic-en-1B"

def initialize_model_and_tokenizer(ckpt_dir, quantization):
    if quantization == "4-bit":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8-bit":
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        qconfig = None

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=qconfig,
        device_map={"": 0}  #  ADDED: automatically map to GPU if available for speed
    )

    # If no quantization, move explicitly to GPU and half precision for efficiency
    if qconfig is None and DEVICE == "cuda":
        model = model.to(DEVICE)
        model.half()  #  ADDED: ensures half precision on GPU for faster inference

    model.eval()
    return tokenizer, model

def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    translations = []
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i : i + BATCH_SIZE]
        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            generate_kwargs = {
                "use_cache": True,
                "min_length": 0,
                "max_length": 128,
                "num_beams": 8,
                "num_return_sequences": 1,
            }
            generate_kwargs.update(inputs)  # merge tensor inputs with generation kwargs

            generated_tokens = model.generate(**generate_kwargs)


        generated_tokens = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)
        

    return translations

# Load models and tokenizer for Indic -> English
tokenizer, model = initialize_model_and_tokenizer(indic_en_ckpt_dir, quantization)
ip = IndicProcessor(inference=True)

def is_devanagari(text):
    return bool(re.search("[\u0900-\u097F]", text))

def translate_to_english(text):
    if is_devanagari(text):
        src = "hin_Deva"
    else:
        # Assume Hinglish if not Devanagari
        text = xlit_engine.translit_sentence(text)["hi"]
        src = "hin_Deva"

    translated = batch_translate([text], src, "eng_Latn", model, tokenizer, ip)
    return translated[0]
