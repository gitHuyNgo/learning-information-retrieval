import os
import subprocess
from faster_whisper import WhisperModel


def init_model():
    return WhisperModel(
        "large-v3",
        device="cpu",
        compute_type="int8"
    )


def extract_audio(video_path, audio_path):
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            audio_path
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=True
    )


def extract_keyframes(video_path, out_dir="keyframes"):
    os.makedirs(out_dir, exist_ok=True)

    subprocess.run(
        [
            "ffmpeg", "-skip_frame", "nokey",
            "-i", video_path,
            "-vsync", "vfr",
            f"{out_dir}/%04d.jpg"
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=True
    )


def transcription(model, audio_path):
    segments, info = model.transcribe(audio_path, beam_size=5)

    print(f"Language: {info.language} ({info.language_probability:.2f})")

    for s in segments:
        print(f"[{s.start:.2f}s → {s.end:.2f}s] {s.text}")


def main():
    model = init_model()
    extract_audio("input.mp4", "output.wav")
    extract_keyframes("input.mp4")
    transcription(model, "output.wav")


if __name__ == "__main__":
    main()