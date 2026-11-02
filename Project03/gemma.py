"""로컬 Gemma로 주행 지시문을 만들고 화면의 맥락을 판단하는 모듈"""

import base64
import time

import cv2
import llama_cpp.mtmd_cpp as mtmd_cpp
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Gemma4ChatHandler

LLM_PATH = "../src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
VISION_PATH = "../src/models/Gemma4/mmproj-google_gemma-4-E2B-it-f16.gguf"
# 기본값인 144개에서는 신호등 색과 표지판 글자를 전혀 읽지 못했고 560개에서 표지판 글자 5개 중 4개를 읽었음(2026-11-02 키프레임 13장 측정)
IMAGE_TOKENS = 560
# 같은 측정에서 도로 종류와 터널은 믿을 수 없어 교차로와 횡단보도만 물음
SCENE_ITEMS = ["교차로", "횡단보도"]

# "40자 이내 한 문장"으로 지시하면 조항의 첫 문장만 요약해 개정된 뒷부분이 빠졌음
INSTRUCTION_PROMPT = (
    "너는 자율주행 차량의 주행 규칙 안내기다. 주어진 법령 조항을 근거로, 지금 장면에서 운전자가 해야 할 행동을 "
    "한국어 명령문 두 문장 이내로 쓴다. 조항에 적힌 행동 조건을 모두 담고, 조항에 없는 내용은 쓰지 않는다."
)
SCENE_PROMPT = (
    "이 사진은 주행 중인 차량에서 앞을 찍은 화면이다. 아래 항목마다 사진에 보이면 예, 보이지 않으면 아니오로만 답하라. "
    "설명은 쓰지 마라.\n교차로:\n횡단보도:\n표지판 글자(없으면 없음):"
)
CHECK_PROMPT = (
    "너는 자율주행 차량의 법령 검색 검증기다. 운전자가 도로에서 본 표지판 글자와 번호가 붙은 후보 조항들이 주어진다. "
    "그 표지판을 본 운전자가 지켜야 할 내용을 직접 정한 조항이 있으면 그 번호 하나만 숫자로 쓰고, 없으면 없음이라고만 쓴다. "
    "지명, 도로 이름, 건물 이름은 운전 규칙과 관련 없다."
)

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
    return Llama(model_path=LLM_PATH, chat_handler=handler, n_gpu_layers=-1, n_ctx=4096, n_batch=512, n_ubatch=512, verbose=False)


def ask(llm, messages, max_tokens):
    """Gemma에게 메시지를 보내고 답 문자열을 받는 함수"""
    # temperature 0은 같은 입력에 늘 같은 답이 나오게 함
    response = llm.create_chat_completion(messages=messages, max_tokens=max_tokens, temperature=0)
    return response["choices"][0]["message"]["content"].strip()


def write_instruction(llm, state, article):
    """조항 하나를 근거로 주행 지시문을 만드는 함수"""
    messages = [
        {"role": "system", "content": INSTRUCTION_PROMPT},
        {"role": "user", "content": f"장면: {state}\n조항: {article['id']} {article['title']}\n{article['text']}"},
    ]
    return ask(llm, messages, 120)


def parse_answer(text):
    """"항목: 값" 줄로 된 답을 항목 이름별 값 사전으로 바꾸는 함수"""
    answer = {}
    for line in text.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            # "표지판 글자(없으면 없음)"처럼 질문의 괄호까지 따라 쓰거나 "**교차로**"처럼 굵게 쓰는 답이 있었음
            answer[key.split("(")[0].strip(" *")] = value.strip(" *")
    return answer


def read_scene(llm, frame):
    """프레임 한 장에서 보이는 맥락 항목과 표지판 글자를 읽는 함수"""
    ok, jpg = cv2.imencode(".jpg", frame)
    url = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()
    content = [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": SCENE_PROMPT}]
    answer = parse_answer(ask(llm, [{"role": "user", "content": content}], 80))
    facts = {item for item in SCENE_ITEMS if answer.get(item, "").startswith("예")}
    sign = answer.get("표지판 글자", "").strip(" .")
    # 표지판 글자 칸에도 예나 아니오로 답하는 경우가 있었고, 그 낱말로 검색하면 엉뚱한 조항이 후보가 됨
    if sign in ("없음", "예", "아니오"):
        sign = ""
    return facts, sign


def check_sign(llm, retriever, sign):
    """표지판 글자로 찾은 후보 조항 중 Gemma가 관련 있다고 고른 조항을 돌려주는 함수"""
    candidates = [article for score, article in retriever.search(sign, 3)]
    # 별표 2 항목은 id가 같아 조항 번호 대신 후보 순번으로 고르게 함
    listing = "\n".join(f"{i + 1}. {a['id']} {a['title']} {a['text'][:300]}" for i, a in enumerate(candidates))
    messages = [
        {"role": "system", "content": CHECK_PROMPT},
        {"role": "user", "content": f"표지판 글자: {sign}\n후보 조항:\n{listing}"},
    ]
    answer = ask(llm, messages, 10)
    for i in range(len(candidates)):
        if answer.startswith(str(i + 1)):
            return candidates[i]
    return None


def watch(llm, retriever, shared):
    """최신 프레임을 계속 판단해 채택한 맥락을 shared["scene"]에 넣는 스레드 함수"""
    last_facts, last_sign = set(), ""
    checked = {}
    while True:
        frame = shared["frame"]
        if frame is None:
            time.sleep(0.1)
            continue
        facts, sign = read_scene(llm, frame)
        # 한 번의 오답이 패널에 바로 나오지 않도록 두 번 연속 같은 판단만 채택함
        scene = {"facts": facts & last_facts, "sign": "", "article": None, "instruction": ""}
        if sign and sign == last_sign:
            if sign not in checked:
                article = check_sign(llm, retriever, sign)
                instruction = write_instruction(llm, "표지판 " + sign, article) if article else ""
                checked[sign] = (article, instruction)
            scene["sign"] = sign
            scene["article"], scene["instruction"] = checked[sign]
        last_facts, last_sign = facts, sign
        print("맥락", sorted(scene["facts"]), scene["sign"])
        # 새 사전을 한 번에 넣어 메인 루프가 반쯤 바뀐 결과를 읽지 않게 함
        shared["scene"] = scene
