import os
import gradio as gr
import whisper
import subprocess
import uuid
from datetime import datetime

# ===== Logging utility =====
def log(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

# ===== Device Setup =====
device = "cuda:1"  # Change if needed, or "cpu"
fp16 = (device != "cpu")

log(f"Loading Whisper model (large-v2) on {device}...")
model = whisper.load_model("large-v2", device=device)
log("Model loaded successfully.")

# ===== Preprocess Audio =====
def preprocess_audio(input_path):
    """
    Converts input audio to 16kHz mono PCM WAV using ffmpeg.
    """
    if not input_path or not os.path.exists(input_path):
        raise ValueError("Audio file path is invalid or missing.")

    output_path = f"temp_{uuid.uuid4().hex}.wav"
    log(f"Preprocessing audio: {input_path} -> {output_path}")

    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        output_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path

# ===== Translate or Transcribe to English =====
def translate_audio(audio_file_path):
    try:
        log(f"Received file: {audio_file_path}")
        if not audio_file_path:
            return "No audio file received."

        # Step 1: Preprocess
        clean_audio = preprocess_audio(audio_file_path)

        # Step 2: Run Whisper Translation
        log("Starting transcription + translation...")
        result = model.transcribe(
            clean_audio,
            task="translate",  # Always translate to English
            beam_size=5,
            best_of=5,
            temperature=0.0,
            patience=1.0,
            fp16=fp16,
            language=None,  # Auto-detect
            condition_on_previous_text=False,   # Prevent context repetition
                   
            compression_ratio_threshold=2.4,     # Remove weird over-compressed outputs
            logprob_threshold=-1.0,              # Filter low confidence segments
            without_timestamps=True              # Cleaner text output
        )


        english_text = result["text"].strip()
        detected_lang = result.get("language", "unknown")
        log(f"Detected language: {detected_lang}")
        log(f"Output text: {english_text}")

        # Step 3: Clean up temp file
        if os.path.exists(clean_audio):
            os.remove(clean_audio)
            log("Temporary file removed.")

        return english_text

    except Exception as e:
        log(f"Error: {str(e)}")
        return f"Error occurred: {str(e)}"

# ===== Gradio Interface =====
log("Creating Gradio interface...")
interface = gr.Interface(
    fn=translate_audio,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath"),
    outputs="text",
    title="Hindi/English to English Transcription",
    description="Upload or record Hindi or English audio. Output will always be in English."
)

log("Launching Gradio app...")
interface.launch(share=True)
