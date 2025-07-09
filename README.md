# 🗣️ Transcription and Translation System

This repository provides a pipeline to **transcribe** speech from audio/video files and **translate** spoken **Hinglish** or **Hindi** sentences into **English** using powerful AI models like [Whisper](https://github.com/openai/whisper) and [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2).

---

## 🎯 Objective

- Transcribe spoken content from audio using OpenAI Whisper.
- Translate spoken **Hinglish** (Hindi written in Latin script) or **Hindi** into **English** text using transliteration and translation modules.

---

## 🧩 Code Modules

### 1. **Speech-to-Text (STT) Module**

- Uses OpenAI’s Whisper model (`large`) to transcribe audio or translate speech.
- Handles both English and Hindi audio input.

> **File**: `STT.py`

#### 🔁 Flow


### 2. **Hinglish to English Translation Module**

- Detects language using `langdetect`.
- If Hinglish, transliterates to Hindi (`Devanagari`) using `ai4bharat.transliteration`.
- Translates Hindi to English using AI4Bharat's `indictrans2-indic-en` model.

> **File**: `hinglish_to_english.py`

#### 🔁 Flow


---

## 📦 Dependencies

Install the following Python packages:

```bash
pip install torch transformers langdetect ai4bharat-transliteration IndicTransToolkit openai-whisper

# For Ubuntu/Debian
sudo apt install ffmpeg

# For Windows (Chocolatey)
choco install ffmpeg

#clone the repo
git clone https://github.com/Sarveshgits/Tobacco-Cessation-Speech-Model.git
cd Tobacco-Cessation-Speech-Model

```
---

## Requirements.txt

```

torch>=2.0.0
transformers>=4.39.0
langdetect>=1.0.9
ai4bharat-transliteration==0.1.5
IndicTransToolkit==1.2.0
openai-whisper>=20231117
ffmpeg-python>=0.2.0

```
