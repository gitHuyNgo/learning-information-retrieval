import os
import subprocess
from faster_whisper import WhisperModel
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector




VIDEO_FILE = "input.mp4"
OUTPUT_FOLDER = "output_data"




def extract_audio(video_path, audio_output_path):
    print(f"[Stream A] Extracting Audio to {audio_output_path}...")

    subprocess.run(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            audio_output_path
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=True
    )
    print(f"[Stream A] Audio extraction complete!")


def get_timestamps(video_path, threshold=10.0, min_interval=15.0):
    print(f"[Stream B] Detecting scenes in video...")

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video, show_progress=True)

    scene_list = scene_manager.get_scene_list()
    smart_times = [scene[0].get_seconds() for scene in scene_list]

    duration = video.duration.get_seconds()
    forced_times = [t for t in range(0, int(duration), int(min_interval))]
    
    all_times = sorted(list(set(smart_times + forced_times)))
    
    clean_times = []
    if all_times:
        last_t = all_times[0]
        clean_times.append(last_t)
        for t in all_times[1:]:
            if t - last_t > 2.0: 
                clean_times.append(t)
                last_t = t

    print(f"[Stream B] Found {len(clean_times)} timestamps")
    return clean_times


def extract_frames(video_path, timestamps, output_folder):
    print(f"[Stream B] Extracting frames to {output_folder}")
    
    os.makedirs(output_folder, exist_ok=True)
    
    for i, t in enumerate(timestamps):
        filename = f"frame_{int(t):05d}.jpg"
        output_path = os.path.join(output_folder, filename)
        
        subprocess.run(
            [
                "ffmpeg", "-ss", str(t), "-i", video_path,
                "-frames:v", "1", "-q:v", "2", "-y", "-loglevel", "error",
                output_path
            ],
            check=True
        )
        print(f"  Extracted frame {i+1}/{len(timestamps)}: {filename}")
    
    print(f"[Stream B] Frame extraction complete!")


def align_audio_keyframes_content():
    '''
    Docstring for align_audio_keyframes_content:
        - Use faster_whisper to collect audio from .wav file then align it with keyframes to enhance retrieved data
    '''
    pass


if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    audio_output = os.path.join(OUTPUT_FOLDER, "audio.wav")
    frames_folder = os.path.join(OUTPUT_FOLDER, "frames")

    extract_audio(VIDEO_FILE, audio_output)
    timestamps = get_timestamps(VIDEO_FILE, threshold=10.0, min_interval=15.0)

    if timestamps:
        extract_frames(VIDEO_FILE, timestamps, frames_folder)
    else:
        print("[Stream B] No timestamps found to extract frames")