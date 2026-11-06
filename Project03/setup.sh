#!/bin/bash
# Project03을 실행할 장비를 맞추는 스크립트
#   bash setup.sh                이 장비를 셋업함
#   bash setup.sh 사용자@주소    코드와 영상을 그 장비로 보낸 뒤 그 장비를 셋업함

set -u
cd "$(dirname "$0")" || exit 1

REMOTE_DIR=vision-llm
MODELS=../src/models
GEMMA_DIR=$MODELS/Gemma4
YOLO_DIR=$MODELS/YOLO
LLM_FILE=google_gemma-4-E2B-it-Q4_K_M.gguf
VISION_FILE=mmproj-google_gemma-4-E2B-it-f16.gguf
GEMMA_URL=https://huggingface.co/bartowski/google_gemma-4-E2B-it-GGUF/resolve/main
YOLO_URL=https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
ENGINE_FILE=yolo11n_736x1280_fp16.engine
VIDEOS=(tunnel-korea.mp4 golden-gate.mp4 sf-demo.mp4 school-zone-stills.mp4)
EXIT_NO_DOWNLOAD=3
# Jetson 수업 환경과 같은 버전을 받는 torch 저장소(03 노트북)
JETSON_INDEX=https://pypi.jetson-ai-lab.io/jp6/cu126
NVCC=/usr/local/cuda/bin/nvcc

send_and_setup() {
    target=$1
    cd .. || exit 1
    includes=()
    for video in "${VIDEOS[@]}"; do
        includes+=(--include "Project03/videos/$video")
    done
    rsync -avRP "${includes[@]}" --include 'Project03/videos/*-meta.txt' --exclude 'Project03/videos/*' \
        --exclude 'Project03/data' --exclude '__pycache__' --exclude '.ruff_cache' \
        --exclude 'Project03/setup-report.txt' --exclude 'Project03/jetson-*' --exclude 'Project03/vla-output.jsonl' \
        Project03 src/models/YOLO/yolo11n.pt "$target:$REMOTE_DIR/" || exit 1
    ssh -t "$target" "cd $REMOTE_DIR/Project03 && bash setup.sh"
    status=$?
    # 장비가 Hugging Face에 접속하지 못할 때만 이 컴퓨터의 Gemma 파일을 보냄
    if [ "$status" -eq "$EXIT_NO_DOWNLOAD" ]; then
        rsync -avRP "src/models/Gemma4/$LLM_FILE" "src/models/Gemma4/$VISION_FILE" "$target:$REMOTE_DIR/" || exit 1
        ssh -t "$target" "cd $REMOTE_DIR/Project03 && bash setup.sh"
        status=$?
    fi
    exit "$status"
}

file_size() {
    if [ -e "$1" ]; then
        wc -c < "$1" | tr -d ' '
    else
        echo 0
    fi
}

apt_install() {
    # 패키지 목록이 오래되면 설치할 판을 찾지 못함
    sudo apt-get update
    sudo apt-get install -y "$1"
}

is_usable_python() {
    "$1" -c "import $2" 2> /dev/null
}

use_venv() {
    folder=$1
    shift
    PY=$folder/bin/python
    if [ ! -x "$PY" ]; then
        base=$(command -v python3.10 || command -v python3)
        if ! "$base" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 10))'; then
            echo "실패: Python 3.10이 필요합니다. numpy 1.23을 설치할 수 있는 판이기 때문입니다."
            exit 1
        fi
        # Ubuntu의 python3에는 venv가 없어 00 노트북처럼 먼저 설치함
        if [ "$(uname)" = Linux ] && ! "$base" -c 'import ensurepip' 2> /dev/null; then
            apt_install python3-venv
        fi
        if ! "$base" -m venv "$@" "$folder"; then
            # 만들다 멈춘 폴더가 남으면 다음 실행이 pip 없는 환경을 그대로 씀
            rm -rf "$folder"
            exit 1
        fi
    fi
}

choose_python() {
    if [ -n "${VIRTUAL_ENV:-}" ]; then
        PY=$VIRTUAL_ENV/bin/python
    elif [ "$DEVICE" != jetson ]; then
        use_venv ../.venv-project03
    else
        PY=
        candidates=$(find "$HOME" -maxdepth 6 -path '*/.venv/bin/python' 2> /dev/null)
        # 수업을 어디까지 진행했는지에 따라 있는 패키지가 달라 가장 나중 단계의 패키지부터 찾음
        for module in llama_cpp ultralytics torch; do
            while IFS= read -r candidate; do
                if [ -z "$PY" ] && [ -n "$candidate" ] && is_usable_python "$candidate" "$module"; then
                    PY=$candidate
                fi
            done <<< "$candidates"
        done
        if [ -z "$PY" ]; then
            # OpenCV와 tensorrt가 JetPack 시스템 패키지라 00 노트북처럼 함께 보이게 만듦
            use_venv ../.venv --system-site-packages
        fi
    fi
    PY=$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")
    echo "Python: $PY"
}

need() {
    module=$1
    shift
    if ! is_usable_python "$PY" "$module"; then
        "$PY" -m pip --version > /dev/null 2>&1 || "$PY" -m ensurepip
        echo "설치: pip install $*"
        "$PY" -m pip install "$@"
    fi
}

install_llama() {
    if is_usable_python "$PY" llama_cpp; then
        return
    fi
    # Linux 빌드에 C++ 컴파일러가 필요함
    if [ "$DEVICE" != mac ] && ! command -v g++ > /dev/null; then
        apt_install build-essential
    fi
    if [ -x "$NVCC" ] || command -v nvcc > /dev/null; then
        cuda_compiler=$(command -v nvcc || echo "$NVCC")
        echo "CUDA로 llama-cpp-python을 빌드합니다. 수십 분 걸릴 수 있습니다."
        PATH=$(dirname "$cuda_compiler"):$PATH CUDACXX=$cuda_compiler \
            CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_COMPILER=$cuda_compiler" need llama_cpp --no-cache-dir llama-cpp-python==0.3.36
    else
        need llama_cpp llama-cpp-python==0.3.36
    fi
}

install_packages() {
    if [ "$DEVICE" = jetson ]; then
        # 03, 04, 05 노트북의 명령과 버전을 그대로 씀
        need numpy numpy==1.23.0
        need torch torch==2.8.0 --index-url "$JETSON_INDEX" --no-deps
        need torchvision torchvision==0.23.0 --index-url "$JETSON_INDEX" --no-deps
        need sympy --no-deps "sympy>=1.13.3"
        need PIL --no-deps Pillow==10.4.0
        need typing_extensions --no-deps "typing_extensions>=4.10.0,<5"
        need filelock --no-deps "filelock>=3.16.1,<4"
        # 05 노트북은 판을 적지 않았으나 개발에서 측정한 판으로 고정함
        need ultralytics --no-deps ultralytics==8.4.194
        need onnx --no-deps onnx==1.16.0
        need onnxruntime --no-deps onnxruntime==1.23.2
        need onnxslim --no-deps onnxslim==0.1.95
        need ml_dtypes --no-deps ml_dtypes==0.5.4
        need coloredlogs --no-deps coloredlogs==15.0.1
        need humanfriendly --no-deps humanfriendly==10.0
        need google.protobuf --no-deps protobuf==3.20.3
    else
        # Mac 개발 환경 .venv-project03과 같은 버전임
        need numpy numpy==1.23.5
        need cv2 opencv-python==4.10.0.84
        need PIL Pillow==10.4.0
        need torch torch==2.14.1
        need torchvision torchvision==0.29.2
        need ultralytics ultralytics==8.4.194
    fi
    install_llama
}

prepare_file() {
    folder=$1
    name=$2
    size=$3
    url=$4
    path=$folder/$name
    mkdir -p "$folder"
    if [ "$(file_size "$path")" = "$size" ]; then
        echo "있음: $name"
        return 0
    fi
    while IFS= read -r candidate; do
        if [ -n "$candidate" ] && [ "$(file_size "$candidate")" = "$size" ]; then
            rm -f "$path"
            ln -s "$candidate" "$path"
            echo "연결: $path -> $candidate"
            return 0
        fi
    done <<< "$(find "$HOME" -name "$name" 2> /dev/null)"
    available=$(df -Pk "$folder" | awk 'NR==2 {print $4}')
    if [ "$((size / 1024))" -gt "$available" ]; then
        echo "실패: 디스크가 부족합니다. $name에 ${size} 바이트가 필요합니다."
        exit 1
    fi
    echo "내려받기: $name"
    curl -L -C - --fail --connect-timeout 20 --speed-limit 1024 --speed-time 60 --retry 2 -o "$path" "$url"
    if [ "$(file_size "$path")" != "$size" ]; then
        echo "실패: $name을 내려받지 못했습니다. 다른 컴퓨터에서 'bash setup.sh 사용자@주소'로 보내면 그 컴퓨터의 파일을 씁니다."
        exit "$EXIT_NO_DOWNLOAD"
    fi
}

install_font() {
    if [ "$DEVICE" != mac ] && [ ! -f /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc ] && [ ! -f /usr/share/fonts/truetype/nanum/NanumGothic.ttf ]; then
        apt_install fonts-noto-cjk
    fi
}

build_engine() {
    yolo_path=$(cd "$YOLO_DIR" && pwd)
    if [ -f "$yolo_path/$ENGINE_FILE" ]; then
        echo "있음: $ENGINE_FILE"
        return
    fi
    # 04 노트북의 내보내기와 trtexec 순서이고 입력 크기만 detect.IMAGE_SIZE에 맞춤
    # 내보내기가 실패하면 이전 ONNX로 엔진을 만들지 않고 멈춤
    if ! "$PY" -c "from ultralytics import YOLO; YOLO('$yolo_path/yolo11n.pt').export(format='onnx', imgsz=[736, 1280], batch=1, dynamic=False, simplify=True, opset=20)" ||
        ! /usr/src/tensorrt/bin/trtexec --onnx="$yolo_path/yolo11n.onnx" --saveEngine="$yolo_path/$ENGINE_FILE" --fp16; then
        rm -f "$yolo_path/$ENGINE_FILE"
    fi
}

check() {
    name=$1
    shift
    if "$@" > /dev/null 2>&1; then
        echo "통과: $name"
    else
        echo "실패: $name"
        FAILURES=$((FAILURES + 1))
    fi
}

warn_jetson_state() {
    # 05 노트북: R36.4.7은 메모리가 남아도 CUDA 할당 오류가 날 수 있고 R36.5에서 고쳐짐
    if head -n 1 /etc/nv_tegra_release | grep -q 'R36 (release), REVISION: 4.7'; then
        echo "경고: Jetson Linux R36.4.7입니다. Gemma 할당 오류가 나면 05 노트북의 R36.5 업그레이드 절차를 따릅니다. 재부팅이 필요해 자동으로 하지 않습니다."
    fi
    power=$(sudo -n nvpmodel -q 2> /dev/null)
    if [ -z "$power" ]; then
        echo "확인 못 함: 전력 모드. 'sudo nvpmodel -q'로 확인합니다."
    elif ! echo "$power" | grep -q MAXN; then
        echo "경고: 전력 모드가 MAXN이 아닙니다. 04 노트북 안내대로 바꿉니다."
    fi
    if [ ! -e /dev/video0 ]; then
        echo "경고: /dev/video0이 없습니다. 카메라 버튼을 쓰려면 USB 카메라를 연결합니다."
    fi
}

check_all() {
    FAILURES=0
    check "Python 3.10" "$PY" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 10))'
    check "numpy 2 미만" "$PY" -c 'import numpy, sys; sys.exit(int(numpy.__version__.split(".")[0]) >= 2)'
    check "OpenCV" "$PY" -c 'import cv2'
    check "ultralytics" "$PY" -c 'import ultralytics'
    check "Gemma4ChatHandler" "$PY" -c 'from llama_cpp.llama_chat_format import Gemma4ChatHandler'
    check "한글 글꼴" "$PY" -c 'import hud'
    check "$LLM_FILE" test "$(file_size "$GEMMA_DIR/$LLM_FILE")" = 3462680032
    check "$VISION_FILE" test "$(file_size "$GEMMA_DIR/$VISION_FILE")" = 985653760
    check "yolo11n.pt" test "$(file_size "$YOLO_DIR/yolo11n.pt")" = 5613764
    for video in "${VIDEOS[@]}"; do
        check "$video" "$PY" -c "import cv2, sys; c = cv2.VideoCapture('videos/$video'); sys.exit(not (c.isOpened() and c.get(cv2.CAP_PROP_FPS) > 0 and c.read()[0]))"
    done
    if [ "$DEVICE" = jetson ]; then
        check "torch CUDA" "$PY" -c 'import sys, torch; sys.exit(not torch.cuda.is_available())'
        check "tensorrt, onnx, onnxslim" "$PY" -c 'import tensorrt, onnx, onnxslim'
        check "llama-cpp-python GPU 빌드" "$PY" -c 'import sys, llama_cpp; sys.exit(not llama_cpp.llama_supports_gpu_offload())'
        check "$ENGINE_FILE" test -f "$YOLO_DIR/$ENGINE_FILE"
        check "OpenCV GStreamer" "$PY" -c "import cv2, sys; sys.exit('GStreamer:                   YES' not in cv2.getBuildInformation())"
        warn_jetson_state
    fi
}

try_frame() {
    if [ "$DEVICE" = jetson ]; then
        model=$YOLO_DIR/$ENGINE_FILE
    else
        model=$YOLO_DIR/yolo11n.pt
    fi
    # 화면을 쓰지 않아 SSH에서도 실제 판단 경로를 끝까지 지남
    "$PY" - "$model" << 'PYTHON'
import sys
import time

import cv2
from ultralytics import YOLO

import gemma
from detect import CLASSES, DETECT_CONF, IMAGE_SIZE

frame = cv2.resize(cv2.VideoCapture("videos/tunnel-korea.mp4").read()[1], (1280, 720))
start = time.perf_counter()
result = YOLO(sys.argv[1]).predict(source=frame, conf=DETECT_CONF, iou=0.5, verbose=False, classes=CLASSES, imgsz=IMAGE_SIZE)[0]
print("YOLO 탐지", len(result.boxes), "개,", round(time.perf_counter() - start, 2), "초")
llm = gemma.load_gemma()
start = time.perf_counter()
found = gemma.recognize(llm, frame, result)
print("Gemma 판단", found, round(time.perf_counter() - start, 2), "초")
law = gemma.load_law("current")
situation = found[0] if found and found[0] in law else next(iter(law))
print("지시문", gemma.read_provision(llm, situation, law[situation]))
PYTHON
}

setup_device() {
    exec > >(tee setup-report.txt) 2>&1
    if [ -f /etc/nv_tegra_release ]; then
        DEVICE=jetson
    elif [ "$(uname)" = Darwin ]; then
        DEVICE=mac
    else
        DEVICE=linux
    fi
    echo "장비: $DEVICE"
    choose_python
    install_packages
    prepare_file "$GEMMA_DIR" "$LLM_FILE" 3462680032 "$GEMMA_URL/$LLM_FILE"
    prepare_file "$GEMMA_DIR" "$VISION_FILE" 985653760 "$GEMMA_URL/$VISION_FILE"
    prepare_file "$YOLO_DIR" yolo11n.pt 5613764 "$YOLO_URL"
    install_font
    if [ "$DEVICE" = jetson ]; then
        build_engine
    fi
    check_all
    if [ "$FAILURES" -gt 0 ]; then
        echo "실패 $FAILURES개: 위 항목을 고친 뒤 다시 실행합니다. 결과는 setup-report.txt에 있습니다."
        exit 1
    fi
    if ! try_frame; then
        echo "실패: 시험 실행. 결과는 setup-report.txt에 있습니다."
        exit 1
    fi
    echo "모든 점검 통과"
    echo "실행 명령: cd $(pwd) && $PY main.py videos/tunnel-korea.mp4$([ "$DEVICE" = jetson ] && echo " --model $YOLO_DIR/$ENGINE_FILE")"
}

if [ $# -eq 1 ]; then
    send_and_setup "$1"
else
    setup_device
fi
