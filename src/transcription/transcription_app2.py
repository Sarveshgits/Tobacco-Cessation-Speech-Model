import os
import gradio as gr
import whisper
import subprocess
import uuid
import soundfile as sf
from datetime import datetime

# ===== Logging utility =====
def log(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

# ===== Device Setup =====
device = "cuda:0"  # Change if needed, or "cpu"
fp16 = (device != "cpu")

log(f"Loading Whisper model (large-v2) on {device}...")
model = whisper.load_model("large-v2", device=device)
log("Model loaded successfully.")

# ===== Save NumPy audio to file =====
def save_numpy_audio(audio_tuple):
    if not audio_tuple:
        raise ValueError("No audio data received.")
    data, samplerate = audio_tuple
    temp_wav = f"temp_{uuid.uuid4().hex}.wav"
    sf.write(temp_wav, data, 16000)  # Save as 16kHz mono WAV
    return temp_wav

# ===== Translate or Transcribe to English =====
def translate_audio(audio_input):
    try:
        log(f"Received audio data: {type(audio_input)}")

        if audio_input is None:
            return "No audio file received."

        # If audio_input is a tuple: (numpy_array, sample_rate)
        if isinstance(audio_input, tuple):
            audio_data, sr = audio_input
            if not isinstance(audio_data, np.ndarray):
                return "Invalid audio data."
            temp_path = f"temp_{uuid.uuid4().hex}.wav"
            sf.write(temp_path, audio_data, sr)
            clean_audio = preprocess_audio(temp_path)

        # If audio_input is a file path (string)
        elif isinstance(audio_input, str) and os.path.exists(audio_input):
            clean_audio = preprocess_audio(audio_input)

        else:
            return "Unsupported audio input format."

        # Whisper transcription + translation
        log("Starting transcription + translation...")
        result = model.transcribe(
            clean_audio,
            task="translate",
            beam_size=5,
            best_of=5,
            temperature=0.0,
            patience=1.0,
            fp16=fp16,
            language=None,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            without_timestamps=True
        )

        english_text = result["text"].strip()
        detected_lang = result.get("language", "unknown")
        log(f"Detected language: {detected_lang}")
        log(f"Output text: {english_text}")

        # Cleanup
        if os.path.exists(clean_audio):
            os.remove(clean_audio)
        return english_text

    except Exception as e:
        log(f"Error: {str(e)}")
        return f"Error occurred: {str(e)}"

# ===== Gradio Interface =====
log("Creating Gradio interface...")
interface = gr.Interface(
    fn=translate_audio,
    inputs=gr.Audio(sources=["microphone", "upload"], type="numpy"),
    outputs="text",
    title="Hindi/English to English Transcription",
    description="Upload or record Hindi or English audio. Output will always be in English."
)

log("Launching Gradio app...")
interface.launch(share=True)
