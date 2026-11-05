"""로컬 Gemma로 상황을 인식하고 조문을 읽어 지시문을 만드는 모듈"""

import base64
import json
import time

import cv2
import llama_cpp.mtmd_cpp as mtmd_cpp
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Gemma4ChatHandler

from detect import road_crop, sign_strip

LLM_PATH = "../src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
VISION_PATH = "../src/models/Gemma4/mmproj-google_gemma-4-E2B-it-f16.gguf"
# 이미지 3장의 크기를 따로 정하려고 토큰 수를 고정하지 않고 범위로 둠
MIN_IMAGE_TOKENS = 70
MAX_IMAGE_TOKENS = 560
# 비전 모듈은 48픽셀 칸 하나를 토큰 하나로 읽음
TOKEN_PIXELS = 48
GEMMA_THREADS = 2
# 개발용 세트의 정답 유형에서 정한 상황이고 이 목록 밖의 상황에는 지시문을 내지 않음
SITUATIONS = [
    "a red traffic light facing the car",
    "a yellow traffic light facing the car",
    "a red octagonal STOP sign facing the car",
    "a YIELD sign facing the car",
    "a sign that prohibits turning, such as NO TURN or a crossed-out arrow",
    "a speed limit sign showing a number",
    "a sign or marking that prohibits passing or overtaking",
    "a pedestrian walking on a crosswalk ahead",
    "a lane reserved for buses, shown by a sign or BUS painted on the road",
    "a lane or track reserved for trams or streetcars",
    "a road work zone with cones, barriers or workers",
    "the car is inside or entering a tunnel",
    "the car is inside a tunnel, where headlights must be on",
    "the car is on a bridge",
]
# 예시 없는 객관식은 22/53이고 글 예시 6개를 넣으면 26/53이었음(2026-11-05 개발용 측정)
RECOGNIZE_PROMPT = (
    "Possible situations:\n"
    + "".join([f"{number}. {situation}\n" for number, situation in enumerate(SITUATIONS, 1)])
    + "0. none of these; ordinary driving\n"
    "Examples:\n"
    "The light ahead is red and a person is crossing on the crosswalk. Answer: 1, 8\n"
    "A green light ahead, cars in front, shops on both sides. Answer: 0\n"
    "A sign with a crossed-out left arrow above the intersection. Answer: 5\n"
    "Dark walls and ceiling lamps surround the road. Answer: 12, 13\n"
    "Orange cones close off the right lane. Answer: 11\n"
    "Parked cars and a street name sign, no rule applies. Answer: 0\n"
    "Which situations are present right now? Reply with only the numbers, separated by commas."
)
# 행동 영어 표현마다 패널에 쓰는 한국어 이름과 지시문을 잇는 전치사를 둠
ACTIONS = {
    "Stop": ("정지", " for"),
    "Yield": ("양보", " to"),
    "Slow down": ("서행", " for"),
    "Keep out": ("진입·회전 금지", " of"),
    "Overtaking banned": ("앞지르기·차로 변경 금지", " at"),
    "Caution": ("주의", " for"),
}
READ_PROMPT = (
    "A camera on a car saw this situation. Some provisions below may not apply. "
    "Follow the provision that matches the situation and tell a driving model what to do now. "
    "Choose one action from: " + ", ".join(ACTIONS) + ". Then write an instruction of at most 8 words naming what it applies to.\n"
)
# 예시가 없으면 5/53이고 14상황 모두 Caution이었음(2026-11-05 개발용 측정)
# 예시 조문은 코퍼스와 개발용 상황에서 가져오지 않고 새로 씀
READ_EXAMPLES = (
    "Example:\nSituation: a lowered gate at a railway crossing\nProvisions:\n[1] A driver shall halt before a crossing gate that is down.\nAnswer: Stop for the lowered crossing gate\n"
    "Example:\nSituation: a merge lane joining a busy main road\nProvisions:\n[1] A driver entering a main road shall give way to vehicles on it.\nAnswer: Yield to vehicles on the main road\n"
    "Example:\nSituation: a school zone with children nearby\nProvisions:\n[1] A driver shall reduce speed inside a school zone.\nAnswer: Slow down for the school zone\n"
    "Example:\nSituation: a one-way road against the car's direction\nProvisions:\n[1] No vehicle shall enter a one-way road against its direction.\nAnswer: Keep out of the one-way road\n"
    "Example:\nSituation: a double solid center line on a curve\nProvisions:\n[1] A driver shall not pass another vehicle across a double solid line.\nAnswer: Overtaking banned at the double line\n"
    "Example:\nSituation: a flashing amber beacon over the road\nProvisions:\n[1] A driver shall proceed with care where an amber beacon flashes.\nAnswer: Caution for the flashing beacon\n"
)
# 인식이 완벽하다고 가정한 개발용 수치는 120자 44/53, 240자 46/53, 400자 39/53이었음(2026-11-05)
READ_CHARS = 240

default_vision_params = mtmd_cpp.mtmd_context_params_default


def vision_params():
    """이미지 토큰 수 범위를 정한 비전 모듈 설정을 만드는 함수"""
    params = default_vision_params()
    params.image_min_tokens = MIN_IMAGE_TOKENS
    params.image_max_tokens = MAX_IMAGE_TOKENS
    return params


def load_gemma():
    """비전 모듈을 붙인 Gemma를 불러오는 함수"""
    # Gemma4ChatHandler가 이미지 토큰 수를 인자로 받지 않아 설정을 만드는 함수를 바꿔 끼움
    mtmd_cpp.mtmd_context_params_default = vision_params
    handler = Gemma4ChatHandler(clip_model_path=VISION_PATH, verbose=False)
    # 기본값은 CPU 코어 수만큼 스레드를 써서 같은 CPU의 YOLO가 20ms에서 358ms로 느려졌고, 2개로 줄이면 110ms였음(2026-11-03 Mac 측정)
    # 비전 모듈도 n_threads 값을 그대로 씀
    return Llama(model_path=LLM_PATH, chat_handler=handler, n_gpu_layers=-1, n_ctx=4096, n_batch=512, n_ubatch=512,
                 n_threads=GEMMA_THREADS, n_threads_batch=GEMMA_THREADS, verbose=False)


def load_law(short):
    """법령의 상황별 조문 사전을 만드는 함수"""
    with open(f"corpus-{short}.json", encoding="utf-8") as file:
        articles = json.load(file)["articles"]
    with open(f"index-{short}.json", encoding="utf-8") as file:
        picks = json.load(file)["picks"]
    provisions = {}
    for situation, pick in picks.items():
        # Sonnet이 맞는 조문을 찾지 못한 상황은 null이고 조문 없이 지시문을 내지 않음
        if pick["provision"] is not None:
            provisions[situation] = articles[pick["provision"]]
    return provisions


def image_part(image, tokens):
    """이미지를 토큰 수에 맞게 줄여 Gemma 메시지 항목으로 만드는 함수"""
    height, width = image.shape[:2]
    scale = (tokens * TOKEN_PIXELS * TOKEN_PIXELS / (width * height)) ** 0.5
    if scale < 1:
        size = (max(TOKEN_PIXELS, int(width * scale)), max(TOKEN_PIXELS, int(height * scale)))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])[1]
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode()}}


def parse_situations(answer):
    """Gemma가 답한 번호 문자열을 상황 목록으로 바꾸는 함수"""
    found = []
    for part in answer.replace(".", ",").split(","):
        part = part.strip()
        if part.isdigit() and 1 <= int(part) <= len(SITUATIONS) and SITUATIONS[int(part) - 1] not in found:
            found.append(SITUATIONS[int(part) - 1])
    return found


def recognize(llm, frame, result):
    """프레임과 크롭 두 장을 보고 지금 있는 상황 목록을 만드는 함수"""
    # 고정 크롭을 YOLO 크롭으로 바꾸어 현행 개발용 수치가 22/53에서 31/53으로 오름(2026-11-05)
    content = [image_part(frame, 280), image_part(sign_strip(frame, result), 140), image_part(road_crop(frame), 140), {"type": "text", "text": RECOGNIZE_PROMPT}]
    response = llm.create_chat_completion(messages=[{"role": "user", "content": content}], max_tokens=20, temperature=0)
    return parse_situations(response["choices"][0]["message"]["content"])


def read_provision(llm, situation, article):
    """조문을 읽어 행동과 지시문을 만드는 함수"""
    text = (article["title"] + ": " if article["title"] else "") + article["text"]
    text = text[:READ_CHARS]
    # create_completion이 BOS를 스스로 붙이므로 프롬프트에 쓰지 않음
    prompt = f"<|turn>user\n{READ_PROMPT}{READ_EXAMPLES}Situation: {situation}\nProvisions:\n[1] {text}\nAnswer:<turn|>\n<|turn>model\n"
    # 행동이 정해진 6개 밖으로 나가지 않도록 첫 토큰을 6개로 제한함
    allowed = {}
    for action in ACTIONS:
        allowed[llm.tokenize(action.encode("utf-8"), add_bos=False)[0]] = 1000.0
    first = llm.create_completion(prompt, max_tokens=1, temperature=0, logit_bias=allowed)["choices"][0]["text"].strip()
    for action, (name, preposition) in ACTIONS.items():
        if action.startswith(first):
            start = action + preposition
            rest = llm.create_completion(prompt + start, max_tokens=12, temperature=0, stop=["\n", "<turn|>"])["choices"][0]["text"]
            return {"law": article["id"], "action": name, "instruction": (start + rest).strip(), "text": text}


def watch(llm, laws, shared):
    """최신 프레임으로 상황을 인식하고 조문을 읽어 규칙 목록을 계속 갱신하는 함수"""
    # 같은 법령과 상황은 입력이 같아 출력도 같으므로 처음 읽은 결과를 다시 씀(2026-11-05 27개 상황 일치 측정)
    readings = {}
    while True:
        if shared["input"] is None:
            time.sleep(0.05)
            continue
        frame, result = shared["input"]
        law = shared["law"]
        rules = []
        for situation in recognize(llm, frame, result):
            if situation in laws[law]:
                if (law, situation) not in readings:
                    readings[(law, situation)] = read_provision(llm, situation, laws[law][situation])
                rules.append(readings[(law, situation)])
        shared["rules"] = rules
