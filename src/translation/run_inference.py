# run_inference.py
import time
from hinglish_to_english_module import translate_to_english

while True:
    inp = input("\nEnter sentence (or type 'exit'): ").strip()
    if inp.lower() in ["exit", "quit"]:
        break
    start=time.time()
    result=translate_to_english(inp)
    end=time.time()
    print(result)
    print(f"Inference Time: {end-start:.2f} seconds")
