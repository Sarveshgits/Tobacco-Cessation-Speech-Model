import whisper
import gradio as gr

model = whisper.load_model("large")  # or "large" for better accuracy

def translate_audio(audio_file_path):
    result = model.transcribe(audio_file_path, task="translate", language="hi")
    return result["text"]

demo = gr.Interface(
    fn=translate_audio,
    inputs=gr.Audio(type="filepath"),
    outputs="text"
)

demo.launch(share=True)
