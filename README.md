# 🎙️ Speech to Text Using Whisper Models

Automatic audio/video transcription using OpenAI's Whisper models — fully offline, with no cloud or API costs. This app provides both a Streamlit-based interactive interface and a Jupyter Notebook for exploration and experimentation.

---

## 🧭 Overview

`automation-risalah-rapat-with-ai` is a powerful local transcription tool designed to:

- Transcribe audio/video files using **Whisper**, a state-of-the-art speech recognition model from OpenAI.
- Run **entirely offline** — no internet connection or cloud service required.
- Support common media formats such as `.mp3`, `.wav`, `.mp4`, and more.
- Enable **pause and resume** functionality during transcription.
- Provide both **Streamlit app** and **Jupyter Notebook** interfaces for flexibility in usage and development.

---

## 🧠 About Whisper

[Whisper](https://github.com/openai/whisper) is a general-purpose speech recognition model developed by OpenAI. Trained on hundreds of thousands of hours of multilingual and multitask supervised data, Whisper achieves robust accuracy and language coverage, making it ideal for meetings, interviews, podcasts, and more.

---

## 🧪 Whisper Model Variants

There are six Whisper model sizes, offering a trade-off between speed, memory usage, and transcription accuracy.

|  Size  | Parameters | English-only model | Multilingual model | VRAM Required | Relative Speed |
|:------:|:----------:|:------------------:|:------------------:|:-------------:|:--------------:|
|  tiny  |    39 M    |     `tiny.en`      |       `tiny`       |     ~1 GB     |      ~10×      |
|  base  |    74 M    |     `base.en`      |       `base`       |     ~1 GB     |      ~7×       |
| small  |   244 M    |     `small.en`     |      `small`       |     ~2 GB     |      ~4×       |
| medium |   769 M    |    `medium.en`     |      `medium`      |     ~5 GB     |      ~2×       |
| large  |   1550 M   |        N/A         |      `large`       |    ~10 GB     |       1×       |
| turbo  |   809 M    |        N/A         |      `turbo`       |     ~6 GB     |      ~8×       |

> **Note:** The `.en` versions are optimized for English-only transcription and perform better for those cases. The `turbo` model is a faster variant of `large-v3` with near-equivalent accuracy.

📊 Below is a performance breakdown of Whisper models by language using WER (Word Error Rate) or CER (Character Error Rate):

![WER breakdown by language](https://github.com/openai/whisper/assets/266841/f4619d66-1058-4005-8f67-a9d811b77c62)

For full benchmarking details, refer to the [Whisper paper](https://arxiv.org/abs/2212.04356).

---

## 🚀 Key Features

- ✅ Automatic transcription with Whisper
- 🔒 Fully offline — no need for cloud APIs
- 🛑 Pause and ▶️ Resume button during transcription
- 🔬 Jupyter Notebook version available for technical exploration
- 🎧 Audio and video file support: `.mp3`, `.wav`, `.mp4`, etc.
- 📜 Real-time logging and status updates

---

## 🔧 Requirements

- Python: `3.11.x`
- OS: macOS, Linux, or Windows

Install required system dependencies:

```bash
# You must have ffmpeg installed
brew install ffmpeg     # for macOS
sudo apt install ffmpeg # for Linux
```

Python dependencies (already listed in pyproject.toml):

```bash
streamlit
openai-whisper
torch
numpy
pandas
soundfile
pydub
ffmpeg-python
python-multipart
scipy
```
You can install all dependencies using:
```bash
pip install -r requirements.txt
```