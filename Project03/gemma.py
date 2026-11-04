"""로컬 Gemma를 불러오는 모듈"""

import llama_cpp.mtmd_cpp as mtmd_cpp
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Gemma4ChatHandler

LLM_PATH = "../src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
VISION_PATH = "../src/models/Gemma4/mmproj-google_gemma-4-E2B-it-f16.gguf"
# 기본값인 144개에서는 신호등 색과 표지판 글자를 전혀 읽지 못했고 560개에서 표지판 글자 5개 중 4개를 읽었음(2026-11-02 키프레임 13장 측정)
IMAGE_TOKENS = 560
GEMMA_THREADS = 2
ACTIONS = ["정지", "양보", "서행", "진입·회전 금지", "앞지르기·차로 변경 금지", "주의"]

default_vision_params = mtmd_cpp.mtmd_context_params_default


def vision_params():
    """이미지 토큰 수를 고정한 비전 모듈 설정을 만드는 함수"""
    params = default_vision_params()
    params.image_min_tokens = IMAGE_TOKENS
    params.image_max_tokens = IMAGE_TOKENS
    return params


def load_gemma():
    """비전 모듈을 붙인 Gemma를 불러오는 함수"""
    # Gemma4ChatHandler가 이미지 토큰 수를 인자로 받지 않아 설정을 만드는 함수를 바꿔 끼움
    mtmd_cpp.mtmd_context_params_default = vision_params
    handler = Gemma4ChatHandler(clip_model_path=VISION_PATH, verbose=False)
    # 05 노트북 설정에서 이미지 토큰 560개가 들어가도록 n_ctx와 n_batch를 늘림
    # 기본값은 CPU 코어 수만큼 스레드를 써서 같은 CPU의 YOLO가 20ms에서 358ms로 느려졌고, 2개로 줄이면 110ms였음(2026-11-03 Mac 측정)
    # 비전 모듈도 n_threads 값을 그대로 씀
    return Llama(model_path=LLM_PATH, chat_handler=handler, n_gpu_layers=-1, n_ctx=4096, n_batch=512, n_ubatch=512,
                 n_threads=GEMMA_THREADS, n_threads_batch=GEMMA_THREADS, verbose=False)
