import streamlit as st
from pydub import AudioSegment
import tempfile
import os
import json
import math
import datetime
from whisper import load_model
import textwrap
import hashlib

st.set_page_config(page_title="Whisper Turbo Transcriber", layout="wide")
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "base"
if "model_confirmed" not in st.session_state:
    st.session_state.model_confirmed = False


st.sidebar.image("assets/image/self-daily-persona.jpeg")

OUTPUT_FOLDER = "output_streamlit"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Session State Initialization ---
if "log" not in st.session_state:
    st.session_state.log = []
if "transcripts" not in st.session_state:
    st.session_state.transcripts = {}
if "stop" not in st.session_state:
    st.session_state.stop = False
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "turbo"

def write_log(msg):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    log_msg = f"{timestamp} - {msg}"
    st.session_state.log.append(log_msg)

def get_output_json_path(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    return os.path.join(OUTPUT_FOLDER, base + "_state.json")

def load_state(json_path):
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return {}

def save_state(json_path, state):
    with open(json_path, "w") as f:
        json.dump(state, f, indent=2)

def compute_audio_hash(audio):
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    audio.export(temp_path, format="wav")
    with open(temp_path, "rb") as f:
        audio_bytes = f.read()
    return hashlib.sha256(audio_bytes).hexdigest()

@st.cache_resource(show_spinner=True)
def load_whisper_model_cached(model_name: str):
    model = load_model(model_name)
    return model

# --- Sidebar ---
with st.sidebar:
    with st.expander("⚙️ Settings", expanded=True):
        selected_option = st.selectbox(
            "Pilih Whisper Model",
            options=["tiny", "base", "small", "medium", "large", "turbo"],
            index=["tiny", "base", "small", "medium", "large", "turbo"].index(st.session_state.selected_model),
        )

        if st.button("✅ Konfirmasi model"):
            st.session_state.selected_model = selected_option
            st.session_state.model_confirmed = True
            st.success(f"Model '{selected_option}' telah dikonfirmasi.")
        
        chunk_duration = st.slider("Chunk Duration (seconds)", 10, 60, 30, 1)

        uploaded_file = st.session_state.get("uploaded_file")
        if uploaded_file:
            audio = AudioSegment.from_file(uploaded_file)
            duration_sec = math.ceil(audio.duration_seconds)
            st.markdown(f"**Audio Duration:** {duration_sec} sec")

            start_trim = st.number_input("Trim Start (sec)", 0, duration_sec, 0, key="trim_start")
            end_trim = st.number_input("Trim End (sec)", 0, duration_sec, duration_sec, key="trim_end")
        else:
            start_trim, end_trim, duration_sec = 0, 0, 0

    with st.expander("🪵 Log", expanded=True):
        log_placeholder = st.empty()  # tempat log

    def update_log_display():
        log_placeholder.text_area("Logs", "\n".join(st.session_state.log), height=300, disabled=True)


    if st.session_state.get("model_confirmed", False):
        write_log(f"Loading model: {st.session_state.selected_model}...")
        update_log_display()
        model = load_whisper_model_cached(st.session_state.selected_model)
        write_log(f"Model '{st.session_state.selected_model}' loaded successfully.")
        update_log_display()

# --- Main ---
st.title("🎧 Whisper Turbo Audio Transcriber")

uploaded_file = st.file_uploader("Upload audio file", type=["mp3", "wav", "m4a"], key="uploaded_file")

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    audio = AudioSegment.from_file(uploaded_file)
    duration_sec = math.ceil(audio.duration_seconds)
    st.markdown(f"**Audio Duration:** {duration_sec} seconds")

    start_trim = st.session_state.get("trim_start", 0)
    end_trim = st.session_state.get("trim_end", duration_sec)
    trimmed_audio = audio[start_trim * 1000:end_trim * 1000]

    col1, col2 = st.columns([1, 1])
    with col1:
        transcribe_button = st.button("🚀 Transcribe Audio")
    with col2:
        button_label = "🛑 Stop" if not st.session_state.stop else "▶️ Lanjutkan"
        if st.button(button_label):
            st.session_state.stop = not st.session_state.stop
            write_log("User stopped transcription." if st.session_state.stop else "User resumed transcription.")
            update_log_display()

    audio_hash = compute_audio_hash(trimmed_audio)
    json_path = get_output_json_path(uploaded_file.name)
    state = load_state(json_path)

    if state.get("_hash", "") != audio_hash:
        write_log("New file detected. Resetting cache.")
        update_log_display()
        st.session_state.transcripts = {}
        state = {"_hash": audio_hash}
        save_state(json_path, state)

    total_chunks = math.ceil((end_trim - start_trim) / chunk_duration)

    progress_bar = st.empty()
    chunk_scroll_expander = st.expander("📦 Output per Chunk (Live)", expanded=True)
    full_output_container = st.expander("📄 Full Transcript", expanded=False)

    with chunk_scroll_expander:
        for key in sorted(st.session_state.transcripts.keys(), key=lambda x: int(x)):
            st.markdown(f"##### 🎯 Chunk {int(key)+1}")
            st.code(st.session_state.transcripts[key])

    if transcribe_button:
        st.session_state.stop = False
        write_log("Starting transcription process...")
        update_log_display()
        full_transcript = ""

        with chunk_scroll_expander:
            for i in range(total_chunks):
                if st.session_state.stop:
                    write_log("Transcription manually stopped.")
                    update_log_display()
                    break

                if state.get(str(i), {}).get("done", False):
                    chunk_text = st.session_state.transcripts.get(str(i), "")
                    full_transcript += chunk_text + "\n\n"
                    continue

                start = i * chunk_duration * 1000
                end = min((i + 1) * chunk_duration * 1000, len(trimmed_audio))
                chunk = trimmed_audio[start:end]

                temp_chunk = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                chunk.export(temp_chunk.name, format="wav")

                write_log(f"Processing chunk {i+1}/{total_chunks}...")
                update_log_display()
                result = model.transcribe(temp_chunk.name)
                text = result["text"].strip()
                wrapped_text = textwrap.fill(text, 100)

                start_vtt = str(datetime.timedelta(seconds=(start / 1000)))
                end_vtt = str(datetime.timedelta(seconds=(end / 1000)))
                chunk_text = f"{i+1}\n{start_vtt} --> {end_vtt}\n{wrapped_text}\n"

                st.session_state.transcripts[str(i)] = chunk_text
                st.markdown(f"##### 🎯 Chunk {i+1}")
                st.code(chunk_text)

                full_transcript += chunk_text + "\n\n"
                state[str(i)] = {"done": True}
                save_state(json_path, state)

                progress = int((i + 1) / total_chunks * 100)
                progress_bar.progress(progress / 100.0, f"Chunk {i+1}/{total_chunks}")

        base_name = os.path.splitext(uploaded_file.name)[0]
        txt_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.txt")
        vtt_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.vtt")

        with open(txt_path, "w") as f:
            f.write(full_transcript)

        with open(vtt_path, "w") as f:
            for block in full_transcript.strip().split("\n\n"):
                f.write(block + "\n\n")

        with full_output_container:
            st.text_area("📄 Transcript", full_transcript.strip(), height=300)

        st.download_button("⬇️ Download TXT", full_transcript, file_name=f"{base_name}.txt")
        st.download_button("⬇️ Download VTT", full_transcript, file_name=f"{base_name}.vtt")