import whisper
import os
import time

# Optional: Set a custom model cache path
os.environ["WHISPER_CACHE_DIR"] = "E:/TTS_STT/model_cache"

# Load the Whisper model (medium or large for better accuracy)
model = whisper.load_model("large")

def transcribe_audio(filepath: str, language: str = None, task: str = "transcribe"):
    """
    Transcribe or translate audio using Whisper.

    Args:
        filepath (str): Path to the audio/video file.
        language (str): Force a specific language (e.g., "en", "hi"), or None to auto-detect.
        task (str): "transcribe" (default) or "translate".

    Returns:
        dict: {"text": str, "inference_time": float}
    """
    start = time.time()
    result = model.transcribe(filepath, language=language, task=task)
    duration = time.time() - start
    return {
        "text": result["text"],
        "inference_time": duration
    }