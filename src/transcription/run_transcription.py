from STT import transcribe_audio

# Transcribe Hindi audio
hindi_result = transcribe_audio("../../data/audio/hindi_input.mp4", language="hi", task="transcribe")
print("[Hindi Transcript]:", hindi_result["text"])
print(f"Inference time: {hindi_result['inference_time']:.2f} sec\n")

# Translate English audio
english_result = transcribe_audio("../../data/audio/english_input.mp4", language="en", task="translate")
print("[Translated Text]:", english_result["text"])
print(f"Inference time: {english_result['inference_time']:.2f} sec")
