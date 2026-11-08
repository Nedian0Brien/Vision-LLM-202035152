"""음성 질문을 글로 바꾸고 글을 음성으로 읽는 모듈"""

import os
import signal
import subprocess
import sys
import time

# 새 지시문과 답을 Piper로 읽음
PLAY_SOUND = True
MAC = sys.platform == "darwin"

# 카드 번호는 꽂는 순서와 재부팅에 따라 바뀌어 장치 이름으로 지정함(2026-11-07 Jetson에서 USB Microphone이 카드 2번)
MIC_DEVICE = "plughw:CARD=Microphone,DEV=0"

# 다시 눌러 멈출 수 있어 넉넉히 두고, 누르는 것을 잊어도 이 시간에 끝남
RECORD_SECONDS = 10
AUDIO_DIR = "output/speech"
INPUT_FILE = AUDIO_DIR + "/input.wav"
WHISPER_PATH = "whisper-cli" if MAC else "../whisper.cpp/build-cpu/bin/whisper-cli"
WHISPER_MODEL = (
    "../src/models/Whisper/ggml-base.bin" if MAC else "../whisper.cpp/models/ggml-base.bin"
)
PIPER_PYTHON = "../.venv-piper/bin/python" if MAC else "../.piper_venv/bin/python"
PIPER_MODEL = "../src/models/Piper/ko_KR-kss-medium.onnx"
# Whisper가 16kHz 모노 음성을 받아 두 장비 모두 그 형식으로 녹음함
# Mac은 ffmpeg로 녹음하고 afplay로 재생하며, Jetson은 arecord로 녹음하고 aplay로 재생함
if MAC:
    RECORD_COMMAND = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "avfoundation",
        "-i",
        ":0",
        "-t",
        str(RECORD_SECONDS),
        "-ar",
        "16000",
        "-ac",
        "1",
        INPUT_FILE,
    ]
    PLAY_COMMAND = "afplay"
else:
    RECORD_COMMAND = [
        # pasuspender는 녹음하는 동안 PulseAudio가 잡고 있는 마이크를 풀어 arecord가 쓰게 함
        "pasuspender",
        "--",
        "arecord",
        "-D",
        MIC_DEVICE,
        "-f",
        "S16_LE",
        "-r",
        "16000",
        "-c",
        "1",
        "-d",
        str(RECORD_SECONDS),
        INPUT_FILE,
    ]
    PLAY_COMMAND = "aplay"

# 음성 파일을 쓸 폴더가 없으면 만듦
os.makedirs(AUDIO_DIR, exist_ok=True)

# 같은 지시문을 다시 합성하지 않도록 만든 음성 파일을 재사용함
spoken = {}


def speech_to_text():
    """녹음한 질문을 Whisper로 글로 바꾸는 함수"""
    # -l ko는 한국어로 인식하고 --no-timestamps는 시간 표시 없이 글만 받음
    result = subprocess.run(
        [WHISPER_PATH, "-m", WHISPER_MODEL, "-f", INPUT_FILE, "-l", "ko", "--no-timestamps"],
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def text_to_speech(text):
    """글을 Piper로 음성 파일로 만들어 재생하는 함수"""
    if text not in spoken:

        # 지금까지 만든 파일 수를 번호로 붙여 문장마다 다른 파일에 저장함
        path = f"{AUDIO_DIR}/tts-{len(spoken)}.wav"
        subprocess.run(
            [PIPER_PYTHON, "-m", "piper", "-m", PIPER_MODEL, "-f", path, "--", text],
            capture_output=True,
            check=True,
        )
        spoken[text] = path

    subprocess.run([PLAY_COMMAND, spoken[text]], capture_output=True, check=True)


def speak_loop(shared):
    """가장 최근 문장만 하나씩 읽는 함수"""
    while True:
        # 녹음에 안내 음성이 섞이지 않도록 듣는 동안에는 읽지 않음
        if shared["speech"] is None or shared["listening"]:
            time.sleep(0.05)
            continue

        # 가져가며 비워 두면 읽는 동안 들어온 문장 중 가장 최근 것만 다음에 읽음
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
    """읽기가 끝난 뒤 녹음하고 음성 질문을 질문 칸에 넣는 함수"""
    shared["listening"] = True
    shared["stop_requested"] = False

    # 읽는 소리가 끝나야 녹음에 섞이지 않음
    while shared["speaking"]:
        time.sleep(0.05)

    # 마이크가 없어도 터미널 질문은 계속 받아야 함
    try:
        shared["listen_start"] = time.perf_counter()
        shared["recorder"] = subprocess.Popen(RECORD_COMMAND)

        # 다시 눌러 멈춘 녹음은 0이 아닌 값으로 끝나도 그때까지 녹음한 파일을 씀
        if shared["recorder"].wait() != 0 and not shared["stop_requested"]:
            raise OSError("녹음 실패")

        # 녹음이 끝나면 Whisper로 글로 바꾸고, 글이 비어 있으면 질문으로 쓰지 않음
        shared["recorder"] = None
        shared["transcribing"] = True
        question = speech_to_text()
        if question:
            shared["question"] = question
    except (OSError, subprocess.CalledProcessError):
        shared["voice_error"] = "마이크를 쓸 수 없습니다"

    shared["recorder"] = None
    shared["transcribing"] = False
    shared["listening"] = False


def stop_listening(shared):
    """녹음 중이면 녹음을 멈추는 함수"""
    if shared["recorder"] is not None:
        # 녹음 프로세스를 끝내면 listen 안에서 기다리던 wait가 풀려 그때까지의 파일로 이어 감
        # SIGTERM을 받은 pasuspender는 PulseAudio를 정지 상태로 두어 다음 aplay가 끝나지 않으므로 SIGINT로 멈춤(2026-11-08 Jetson 재현)
        shared["stop_requested"] = True
        shared["recorder"].send_signal(signal.SIGINT)
