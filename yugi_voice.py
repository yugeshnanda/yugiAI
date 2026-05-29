import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import os
import time

# ── Load Whisper model ────────────────────────────────────────────
print("Loading Yugi's ears (Whisper model)...")
model = whisper.load_model("base")
print("Whisper ready!")

# ── Audio settings ────────────────────────────────────────────────
SAMPLE_RATE   = 16000  # Whisper expects 16kHz audio
CHUNK_SECONDS = 5      # Record in 5 second chunks
WAKE_WORD     = "hey yugi"

# ── Record audio from microphone ──────────────────────────────────
def record_chunk(seconds=CHUNK_SECONDS):
    print(f"🎙 Listening for {seconds} seconds...")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()  # Wait until recording is finished
    return audio.flatten()

# ── Transcribe audio using Whisper ────────────────────────────────
def transcribe(audio):
    # Save to a temp file because Whisper reads from files
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
        wav.write(tmp_path, SAMPLE_RATE, audio)

    result = model.transcribe(tmp_path, language="en")
    os.unlink(tmp_path)  # Delete temp file after transcription
    return result["text"].strip().lower()

# ── Main loop ─────────────────────────────────────────────────────
def main():
    print("\n🤖 Yugi Week 3 — Voice Module")
    print(f"Say '{WAKE_WORD}' to wake Yugi up")
    print("Press Ctrl+C to quit\n")

    listening_for_command = False

    while True:
        try:
            audio = record_chunk()
            text  = transcribe(audio)

            if not text:
                continue

            print(f"Heard: '{text}'")

            # ── Wake word detection ───────────────────────────────
            if WAKE_WORD in text:
                print("\n✅ Wake word detected!")
                print("🎙 Yugi is listening — speak your command...\n")
                listening_for_command = True
                continue

            # ── Command processing ────────────────────────────────
            if listening_for_command:
                print(f"\n💬 Command received: '{text}'")
                print("🧠 (Yugi brain coming in Week 4...)\n")
                listening_for_command = False

        except KeyboardInterrupt:
            print("\n👋 Yugi shutting down.")
            break

if __name__ == "__main__":
    main()
