"""음성 질문을 글로 바꾸고 글을 음성으로 읽는 모듈"""

import os
import subprocess
import sys
import time

# 강의실에서 소리가 나지 않도록 기본으로 끔(2026-11-07 사용자 지시)
PLAY_SOUND = False
MAC = sys.platform == "darwin"
MIC_DEVICE = "plughw:3,0"
RECORD_SECONDS = 5
AUDIO_DIR = "speech"
INPUT_FILE = AUDIO_DIR + "/input.wav"
WHISPER_PATH = "whisper-cli" if MAC else "../whisper.cpp/build-cpu/bin/whisper-cli"
WHISPER_MODEL = "../src/models/Whisper/ggml-base.bin" if MAC else "../whisper.cpp/models/ggml-base.bin"
PIPER_PYTHON = "../.venv-piper/bin/python" if MAC else "../.piper_venv/bin/python"
PIPER_MODEL = "../src/models/Piper/ko_KR-kss-medium.onnx"
if MAC:
    RECORD_COMMAND = ["ffmpeg", "-y", "-loglevel", "error", "-f", "avfoundation", "-i", ":0", "-t", str(RECORD_SECONDS), "-ar", "16000", "-ac", "1", INPUT_FILE]
    PLAY_COMMAND = "afplay"
else:
    RECORD_COMMAND = ["pasuspender", "--", "arecord", "-D", MIC_DEVICE, "-f", "S16_LE", "-r", "16000", "-c", "1", "-d", str(RECORD_SECONDS), INPUT_FILE]
    PLAY_COMMAND = "aplay"
os.makedirs(AUDIO_DIR, exist_ok=True)
# 같은 지시문을 다시 합성하지 않도록 만든 음성 파일을 재사용함
spoken = {}


def speech_to_text():
    """마이크로 녹음한 질문을 Whisper로 글로 바꾸는 함수"""
    subprocess.run(RECORD_COMMAND, check=True)
    result = subprocess.run([WHISPER_PATH, "-m", WHISPER_MODEL, "-f", INPUT_FILE, "-l", "ko", "--no-timestamps"], text=True, capture_output=True, check=True)
    return result.stdout.strip()


def text_to_speech(text):
    """글을 Piper로 음성 파일로 만들어 재생하는 함수"""
    if text not in spoken:
        path = f"{AUDIO_DIR}/tts-{len(spoken)}.wav"
        subprocess.run([PIPER_PYTHON, "-m", "piper", "-m", PIPER_MODEL, "-f", path, "--", text], capture_output=True, check=True)
        spoken[text] = path
    subprocess.run([PLAY_COMMAND, spoken[text]], capture_output=True, check=True)


def speak_loop(shared):
    """가장 최근 문장만 하나씩 읽는 함수"""
    while True:
        # 녹음에 안내 음성이 섞이지 않도록 듣는 동안에는 읽지 않음
        if shared["speech"] is None or shared["listening"]:
            time.sleep(0.05)
            continue
        text = shared["speech"]
        shared["speech"] = None
        if not PLAY_SOUND:
            continue
        shared["speaking"] = True
        # 음성 도구나 스피커가 없어도 시연은 화면과 키보드로 이어 가야 함
        try:
            text_to_speech(text)
        except (OSError, subprocess.CalledProcessError):
            shared["voice_error"] = "음성 출력을 쓸 수 없습니다"
        shared["speaking"] = False


def listen(shared):
    """읽기가 끝난 뒤 음성 질문을 받아 질문 칸에 넣는 함수"""
    shared["listening"] = True
    while shared["speaking"]:
        time.sleep(0.05)
    # 마이크가 없어도 터미널 질문은 계속 받아야 함
    try:
        question = speech_to_text()
        if question:
            shared["question"] = question
    except (OSError, subprocess.CalledProcessError):
        shared["voice_error"] = "마이크를 쓸 수 없습니다"
    shared["listening"] = False
