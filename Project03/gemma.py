"""로컬 Gemma로 주행 지시문을 만들고 화면의 맥락을 판단하는 모듈"""

import base64
import time

import cv2
import llama_cpp.mtmd_cpp as mtmd_cpp
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Gemma4ChatHandler

from edge import ACTIONS, sentences

LLM_PATH = "../src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
VISION_PATH = "../src/models/Gemma4/mmproj-google_gemma-4-E2B-it-f16.gguf"
# 기본값인 144개에서는 신호등 색과 표지판 글자를 전혀 읽지 못했고 560개에서 표지판 글자 5개 중 4개를 읽었음(2026-11-02 키프레임 13장 측정)
IMAGE_TOKENS = 560
GEMMA_THREADS = 2
# 개발용 세트에서 두 번의 판단 중 한 번이라도 예이면 인정했을 때 정밀도가 80% 미만인 항목은 묻지 않음(2026-11-02)
# 제외: 횡단보도 5/7, 라바콘 2/3, 양보 표지 0/1, 철길 건널목 0/2, 회전교차로 0/4
# 버스 전용차로는 9개 중 2개만 맞혀 법 개념 대신 눈에 보이는 노면 글자로 물음
SCENE_ITEMS = ["버스 전용 글자가 쓰인 차로", "노면전차 선로", "회전 금지 표지", "터널", "통학버스", "긴급자동차", "어린이보호구역"]

# 의역한 지시문은 부정어와 조건이 빠지거나 뜻이 뒤집혀서 원문 문장의 기호만 고르게 함(2026-11-02)
# 숫자만 붙이면 조문의 항 번호(②, ⑥)나 조 번호(21950)를 답하는 경우가 있어 S1, S2처럼 기호를 붙임
SELECT_PROMPT = (
    "너는 자율주행 차량의 주행 규칙 안내기다. 기호가 붙은 법령 문장 중 주어진 장면에서 자동차 운전자가 지킬 의무를 정한 문장의 기호를 모두 쓰고, "
    "다음 줄에 그 의무가 요구하는 행동 종류를 " + ", ".join(ACTIONS) + " 중 하나 쓴다."
)
# 개발용 엣지 케이스에서 고른 예시이고, 장면 이름, 조항, 문장 목록, 정답으로 이루어짐
EXAMPLES = [
    ('통학버스', '도로교통법 제51조 ① 어린이통학버스의 특별보호', ['① 어린이통학버스가 도로에 정차하여 어린이나 영유아가 타고 내리는 중임을 표시하는 점멸등 등의 장치를 작동 중일 때에는 어린이통학버스가 정차한 차로와 그 차로의 바로 옆 차로로 통행하는 차의 운전자는 어린이통학버스에 이르기 전에 일시정지하여 안전을 확인한 후 서행하여야 한다.'], 'S1\n정지'),
    ('노면전차 선로', '도로교통법 제16조 ② 노면전차 전용로의 설치 등', ['② 노면전차의 운전자는 제1항에 따른 노면전차 전용도로 또는 전용차로로 통행하여야 하며, 차마의 운전자는 노면전차 전용도로 또는 전용차로를 다음 각 호의 경우를 제외하고는 통행하여서는 아니 된다.', '1. 좌회전, 우회전, 횡단 또는 회전하기 위하여 궤도부지를 가로지르는 경우 2. 도로, 교통안전시설, 도로의 부속물 등의 보수를 위하여 진입이 불가피한 경우 3. 노면전차 전용차로에서 긴급자동차가 그 본래의 긴급한 용도로 운행되고 있는 경우'], 'S1\n진입·회전 금지'),
    ('공사 구간', '도로교통법 제5조 ② 신호 또는 지시에 따를 의무', ['② 도로를 통행하는 보행자, 차마 또는 노면전차의 운전자는 제1항에 따른 교통안전시설이 표시하는 신호 또는 지시와 교통정리를 하는 경찰공무원 또는 경찰보조자(이하 "경찰공무원등"이라 한다)의 신호 또는 지시가 서로 다른 경우에는 경찰공무원등의 신호 또는 지시에 따라야 한다.'], 'S1\n주의'),
    ('정지 표지', 'CVC §22450(a)', ['(a) The driver of any vehicle approaching a stop sign at the entrance to, or within, an intersection shall stop at a limit line, if marked, otherwise before entering the crosswalk on the near side of the intersection.', 'If there is no limit line or crosswalk, the driver shall stop at the entrance to the intersecting roadway.'], 'S1, S2\n정지'),
    ('회전 금지 표지', 'CVC §21461(a)', ['(a) It is unlawful for a driver of a vehicle to fail to obey a sign or signal defined as regulatory in the federal Manual on Uniform Traffic Control Devices, or a Department of Transportation approved supplement to that manual of a regulatory nature erected or maintained to enhance traffic safety and operations or to indicate and carry out the provisions of this code or a local traffic ordinance or resolution adopted pursuant to a local traffic ordinance, or to fail to obey a device erected or maintained by lawful authority of a public body or official.'], 'S1\n진입·회전 금지'),
]
# 조항 전체를 주고 지시문을 쓰게 하면 다른 운전자의 의무나 다른 호를 옮겨서 고른 문장과 정해진 행동 종류만 줌
INSTRUCT_PROMPT = (
    "너는 자율주행 차량의 주행 규칙 안내기다. 주어진 법령 문장만 근거로, 주어진 장면에서 자동차가 지킬 의무를 한국어로 한 줄에 하나씩 쓴다. "
    "줄은 '-해야 함', '-해선 안 됨', '-할 수 있음' 중 하나로 끝맺는다. "
    "행동 종류는 이미 정해져 있다. 노면전차나 보행자 같은 다른 주체의 의무는 옮기지 않는다. "
    "문장에 있는 조건, 예외, 부정어를 빠뜨리지 않고, 문장에 없는 내용은 쓰지 않는다."
)
# 개발용 엣지 케이스에서 고른 예시이고, 장면 이름, 행동 종류, 법령 문장, 지시문으로 이루어짐
# 예시가 없을 때는 법령 문장을 그대로 옮기거나 노면전차 운전자의 의무를 자동차에 옮겼음
INSTRUCT_EXAMPLES = [
    ('노면전차 선로', '진입·회전 금지', '② 노면전차의 운전자는 제1항에 따른 노면전차 전용도로 또는 전용차로로 통행하여야 하며, 차마의 운전자는 노면전차 전용도로 또는 전용차로를 다음 각 호의 경우를 제외하고는 통행하여서는 아니 된다. 1. 좌회전, 우회전, 횡단 또는 회전하기 위하여 궤도부지를 가로지르는 경우 2. 도로, 교통안전시설, 도로의 부속물 등의 보수를 위하여 진입이 불가피한 경우 3. 노면전차 전용차로에서 긴급자동차가 그 본래의 긴급한 용도로 운행되고 있는 경우', '노면전차 전용도로나 전용차로로 들어가선 안 됨\n좌회전, 우회전, 횡단, 회전을 위해 궤도를 가로지를 때와 도로 보수로 진입이 불가피할 때는 들어갈 수 있음'),
    ('횡단보도 보행자', '정지', '① 모든 차 또는 노면전차의 운전자는 보행자(제13조의2제6항에 따라 자전거등에서 내려서 자전거등을 끌거나 들고 통행하는 자전거등의 운전자를 포함한다)가 횡단보도를 통행하고 있거나 통행하려고 하는 때에는 보행자의 횡단을 방해하거나 위험을 주지 아니하도록 그 횡단보도 앞(정지선이 설치되어 있는 곳에서는 그 정지선을 말한다)에서 일시정지하여야 한다.', '보행자가 횡단보도를 건너고 있거나 건너려고 하면 횡단보도 앞 정지선에서 일시정지해야 함'),
    ('정지 표지', '정지', '(a) The driver of any vehicle approaching a stop sign at the entrance to, or within, an intersection shall stop at a limit line, if marked, otherwise before entering the crosswalk on the near side of the intersection. If there is no limit line or crosswalk, the driver shall stop at the entrance to the intersecting roadway.', '정지선이 있으면 정지선에서 정지해야 함\n정지선이 없으면 교차로 앞 횡단보도에 들어가기 전에 정지해야 함\n정지선과 횡단보도가 모두 없으면 교차하는 도로의 입구에서 정지해야 함'),
]
SCENE_PROMPT = (
    "이 사진은 주행 중인 차량에서 앞을 찍은 화면이다. 아래 항목마다 사진에 보이면 예, 보이지 않으면 아니오로만 답하라. "
    "설명은 쓰지 마라.\n" + "\n".join(item + ":" for item in SCENE_ITEMS) + "\n표지판 글자(없으면 없음):"
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
    # 기본값은 CPU 코어 수만큼 스레드를 써서 같은 CPU의 YOLO가 20ms에서 358ms로 느려졌고, 2개로 줄이면 110ms였음(2026-11-03 Mac 측정)
    # 비전 모듈도 n_threads 값을 그대로 씀
    return Llama(model_path=LLM_PATH, chat_handler=handler, n_gpu_layers=-1, n_ctx=4096, n_batch=512, n_ubatch=512,
                 n_threads=GEMMA_THREADS, n_threads_batch=GEMMA_THREADS, verbose=False)


def ask(llm, messages, max_tokens):
    """Gemma에게 메시지를 보내고 답 문자열을 받는 함수"""
    # temperature 0은 같은 입력에 늘 같은 답이 나오게 함
    response = llm.create_chat_completion(messages=messages, max_tokens=max_tokens, temperature=0)
    return response["choices"][0]["message"]["content"].strip()


def sentence_message(name, header, lines):
    """장면 이름, 조항, 기호를 붙인 문장 목록으로 사용자 메시지를 만드는 함수"""
    listing = "\n".join(f"S{i + 1}. {line}" for i, line in enumerate(lines))
    return {"role": "user", "content": f"장면: {name}\n조항: {header}\n{listing}"}


def write_instruction(llm, name, article):
    """조항에서 장면에 해당하는 원문 문장과 행동 종류를 고르고 그 문장만으로 지시문을 만드는 함수"""
    lines = sentences(article["text"])
    messages = [{"role": "system", "content": SELECT_PROMPT}]
    for example_name, header, example_lines, example_answer in EXAMPLES:
        messages.append(sentence_message(example_name, header, example_lines))
        messages.append({"role": "assistant", "content": example_answer})
    messages.append(sentence_message(name, (article["id"] + " " + article["title"]).strip(), lines))
    answer = ask(llm, messages, 30).split("\n")
    chosen = []
    for mark in answer[0].split(","):
        mark = mark.strip(" .")
        if mark[1:].isdigit() and 1 <= int(mark[1:]) <= len(lines) and lines[int(mark[1:]) - 1] not in chosen:
            chosen.append(lines[int(mark[1:]) - 1])
    # 고른 기호가 없으면 조항 전체를 보여 줘 의무가 빠지지 않게 함
    if not chosen:
        chosen = lines
    # "주의 진행 중"처럼 낱말을 덧붙여 답하는 경우가 있어 포함된 행동 종류를 고름
    action = ""
    for kind in ACTIONS:
        if kind in answer[-1]:
            action = kind
    messages = [{"role": "system", "content": INSTRUCT_PROMPT}]
    for example_name, example_action, example_text, example_instruction in INSTRUCT_EXAMPLES:
        messages.append({"role": "user", "content": f"장면: {example_name}\n행동 종류: {example_action}\n법령 문장: {example_text}"})
        messages.append({"role": "assistant", "content": example_instruction})
    messages.append({"role": "user", "content": f"장면: {name}\n행동 종류: {action}\n법령 문장: {' '.join(chosen)}"})
    return {"지시": ask(llm, messages, 150), "문장": chosen, "행동 종류": action}


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
    answer = parse_answer(ask(llm, [{"role": "user", "content": content}], 160))
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


def watch(llm, retrievers, shared):
    """활성 엣지 케이스의 지시문을 만들고 최신 프레임을 판단해 shared["written"]과 shared["scene"]에 넣는 스레드 함수"""
    last_video, last_facts, last_sign = None, set(), ""
    written, sign_result = {}, None
    while True:
        if shared["frame"] is None:
            time.sleep(0.1)
            continue
        corpus, active = shared["corpus"], shared["active"]
        # 사라진 엣지 케이스의 지시문을 버려 다시 나타나면 그 입력으로 새로 만들게 함
        written = {key: value for key, value in written.items() if key[0] == corpus and key[1] in active}
        # 새 사전을 한 번에 넣어 메인 루프가 반쯤 바뀐 결과를 읽지 않게 함
        shared["written"] = dict(written)
        # 비교 모드에서는 법령이 None이고 지시문을 만들지 않음
        for name in active:
            if corpus is not None and (corpus, name) not in written:
                retriever = retrievers[corpus]
                article = retriever.search(retriever.queries[name], 1)[0][1]
                written[(corpus, name)] = (article, write_instruction(llm, name, article))
                shared["written"] = dict(written)
                print("지시문", name, written[(corpus, name)][1]["행동 종류"])
        video, frame = shared["frame"]
        facts, sign = read_scene(llm, frame)
        # 판단하는 동안 영상 버튼이 눌렸으면 이전 영상의 판단을 버림
        if shared["frame"][0] != video:
            continue
        if video != last_video:
            last_facts, last_sign = set(), ""
        corpus = shared["corpus"]
        # 개발용 세트에서 두 번 모두 예일 때만 인정하면 재현율이 46%, 한 번이라도 예이면 58%였음(2026-11-02)
        scene = {"facts": facts | last_facts, "corpus": corpus, "sign": "", "article": None, "instruction": None}
        # 표지판 지시문은 같은 법령에서 같은 글자가 이어서 확인되는 동안만 유지함
        if corpus is not None and sign and sign == last_sign:
            if sign_result is None or sign_result[:2] != (corpus, sign):
                article = check_sign(llm, retrievers[corpus], sign)
                instruction = write_instruction(llm, "표지판 " + sign, article) if article else None
                sign_result = (corpus, sign, article, instruction)
            scene["sign"] = sign
            scene["article"], scene["instruction"] = sign_result[2], sign_result[3]
        else:
            sign_result = None
        last_video, last_facts, last_sign = video, facts, sign
        print("맥락", sorted(scene["facts"]), scene["sign"])
        # 새 사전을 한 번에 넣어 메인 루프가 반쯤 바뀐 결과를 읽지 않게 함
        shared["scene"] = scene
        # Gemma가 쉬는 동안 YOLO가 GPU를 혼자 씀
        time.sleep(shared["interval"])
