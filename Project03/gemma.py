"""로컬 Gemma로 상황을 인식하고 조문을 읽어 지시문을 만드는 모듈"""

import base64
import json
import os
import time

import cv2
import llama_cpp
import numpy as np
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Gemma4ChatHandler

from detect import BUS_CLASSES, HYDRANT_CLASSES, LIGHT_CLASSES, PERSON_CLASSES, SIGN_CLASSES, class_strip, road_crop

LLM_PATH = "../src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
VISION_PATH = "../src/models/Gemma4/mmproj-google_gemma-4-E2B-it-f16.gguf"
EMBED_PATH = "../src/models/EmbeddingGemma/embeddinggemma-300M-Q8_0.gguf"
# 비전 모듈은 48픽셀 칸 하나를 토큰 하나로 읽음
TOKEN_PIXELS = 48
GEMMA_THREADS = 2
# 개발용 세트의 정답 유형에서 정한 상황과 시연용 어린이 보호구역이고 이 목록 밖의 상황에는 지시문을 내지 않음
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
    "the car is inside a tunnel, where overtaking is prohibited",
    "the car is inside a tunnel, where headlights must be on",
    "the car is on a bridge",
    "a child protection school zone, shown by red pavement, the words 어린이 보호구역 painted on the road or a school zone sign",
]
RED, YELLOW, PEDESTRIAN, SCHOOL_ZONE = SITUATIONS[0], SITUATIONS[1], SITUATIONS[7], SITUATIONS[14]
# 객관식 목록에 상황을 더하면 다른 상황의 답이 흔들려 크롭 질문으로 따로 물음(2026-11-06 개발용 측정, 현행 35/53에서 33/53)
RAIN = "it is raining on the road"
SCHOOL_BUS = "a children's school bus driving ahead of the car"
SCHOOL_CROSSWALK = "a crosswalk with no traffic light inside a child protection school zone"
HYDRANT = "a fire hydrant beside the road"
CROP_SITUATIONS = [RAIN, SCHOOL_BUS, SCHOOL_CROSSWALK, HYDRANT]
SITUATION_NAMES = ["적색 신호", "황색 신호", "정지 표지", "양보 표지", "회전 금지", "속도 제한", "앞지르기 금지", "횡단 보행자", "버스 전용차로", "노면전차", "공사 구간", "터널 앞지르기", "터널 전조등", "다리 위", "어린이 보호구역"]
# 상황마다 예시를 하나씩 둠. 적색과 보행자를 한 예시에 묶었을 때는 횡단보도 선택 21번이 모두 적색과 함께 나왔음(2026-11-05 개발용 측정)
RECOGNIZE_PROMPT = (
    "Possible situations:\n"
    + "".join([f"{number}. {situation}\n" for number, situation in enumerate(SITUATIONS, 1)])
    + "0. none of these; ordinary driving\n"
    "Examples:\n"
    "The traffic light ahead shows a red lamp. Answer: 1\n"
    "The traffic light ahead shows a yellow lamp. Answer: 2\n"
    "A red eight-sided sign reading STOP stands at the corner. Answer: 3\n"
    "A downward triangle sign reading YIELD stands at the junction. Answer: 4\n"
    "A sign with a crossed-out left arrow above the intersection. Answer: 5\n"
    "A sign by the road shows the number 40. Answer: 6\n"
    "A sign shows two cars with a line through them. Answer: 7\n"
    "A person is walking across the striped crosswalk ahead. Answer: 8\n"
    "BUS is painted on the lane ahead. Answer: 9\n"
    "Rails run along the lane ahead. Answer: 10\n"
    "Orange cones close off the right lane. Answer: 11\n"
    "Dark walls and ceiling lamps surround the road. Answer: 12, 13\n"
    "Railings and open water lie on both sides of the road. Answer: 14\n"
    "Red pavement with a speed number and the words 어린이 보호구역 painted on it. Answer: 15\n"
    "A green light ahead, cars in front, shops on both sides. Answer: 0\n"
    "Parked cars and a street name sign, no rule applies. Answer: 0\n"
    "Which situations are present right now? Reply with only the numbers, separated by commas."
)
LIGHT_PROMPT = "Which lamp color is lit on the traffic lights that face the camera? Answer with one word: red, yellow, green or none."
LIGHT_WORDS = ["red", "yellow", "green", "none"]
PEDESTRIAN_PROMPT = "Is a person walking on a striped crosswalk in this image? Answer with one word: yes or no."
CROSSWALK_PROMPT = "Is a striped crosswalk painted on the road in this image? Answer with one word: yes or no."
RAIN_PROMPT = "Is it raining in this image? Answer with one word: yes or no."
BUS_PROMPT = "Is this a yellow school bus? Answer with one word: yes or no."
ZEBRA_PROMPT = "Are zebra stripes painted across the road ahead in this image? Answer with one word: yes or no."
OBJECT_PROMPT = "What is the short object standing beside the road in this image? Answer with one word: hydrant, cone, bollard, sign or other."
OBJECT_WORDS = ["hydrant", "cone", "bollard", "sign", "other"]
YES_NO = ["yes", "no"]
# 영어 표현은 행동 고르기 프롬프트에만 쓰고, 코드는 지시문 언어와 무관하게 그대로 둠
ACTIONS = {
    "Stop": "STOP",
    "Yield": "YIELD",
    "Slow down": "SLOW_DOWN",
    "Keep out": "KEEP_OUT",
    "Overtaking banned": "NO_OVERTAKING",
    "Caution": "CAUTION",
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
# 예시와 행동 낱말 없이 조문을 읽게 해야 조문을 베끼지 않고 숫자와 조건까지 답함(2026-11-06 개발용 27개 상황, 형식 8개 비교)
INSTRUCTION_PROMPT = (
    "Traffic law text: {text}\nUnder this law, what must a driver do now when the camera sees {situation}? "
    "Answer in Korean as one short command of at most 10 words."
)
# 조문 없음 모드의 지시문이고 조문을 읽는 프롬프트와 다른 점은 법령 본문이 없다는 것뿐임
CLOSED_PROMPT = "What must a driver do now under traffic law when the camera sees {situation}? Answer in Korean as one short command of at most 10 words."
# 질문에 답할 때도 조문 없음 모드와 조문을 읽는 프롬프트의 차이는 법령 본문뿐임
ANSWER_PROMPT = (
    "Traffic law text:\n{text}\nA driver asks: {question}\n"
    "Answer in Korean in at most two short sentences using only this law text. If this text does not answer the question, say 해당 조문을 찾지 못했습니다."
)
CLOSED_ANSWER_PROMPT = "A driver asks: {question}\nAnswer in Korean in at most two short sentences."
# 질문 18개로 쟀을 때 상위 3개에 정답 조문이 든 비율이 키워드 검색보다 높았음(2026-11-07 개발용, 현행 14/17, 2006년 14/16, 캘리포니아 8/11)
SEARCH_TOP = 3
NO_LAW = "none"
NO_ARTICLE = {"id": "조문 없음", "title": "", "text": "", "references": []}
# 인식이 완벽하다고 가정한 개발용 수치는 120자 44/53, 240자 46/53, 400자 39/53이었음(2026-11-05)
READ_CHARS = 240
# Jetson 판단 약 5초와 처음 읽는 조문 약 5초 동안 지시가 사라지지 않도록 다음 판단까지 유지함(2026-11-07 Jetson 측정, 사용자 지시)
HOLD_SECONDS = 8.0

# 화면이 모듈별 처리 상태를 그리는 값이고 판단에는 쓰지 않음
STATUS = {"stage": "wait", "started": 0.0, "images": {}, "answers": {}, "odds": {}, "outputs": 0}


def load_gemma():
    """비전 모듈을 붙인 Gemma를 불러오는 함수"""
    # Jetson에서 CUDA 그래프 캡처가 YOLO의 torch.cuda.synchronize와 충돌해 프로세스가 멈춰 끔(2026-11-07 Jetson 측정)
    os.environ["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    handler = Gemma4ChatHandler(clip_model_path=VISION_PATH, verbose=False)
    # 기본값은 CPU 코어 수만큼 스레드를 써서 같은 CPU의 YOLO가 20ms에서 358ms로 느려졌고, 2개로 줄이면 110ms였음(2026-11-03 Mac 측정)
    # 비전 모듈도 n_threads 값을 그대로 씀
    return Llama(model_path=LLM_PATH, chat_handler=handler, n_gpu_layers=-1, n_ctx=4096, n_batch=512, n_ubatch=512,
                 n_threads=GEMMA_THREADS, n_threads_batch=GEMMA_THREADS, verbose=False)


def load_law(short):
    """법령의 상황별 조문 사전을 불러오는 함수"""
    if short == NO_LAW:
        return {situation: NO_ARTICLE for situation in SITUATIONS + CROP_SITUATIONS}
    with open(f"index-{short}.json", encoding="utf-8") as file:
        return json.load(file)["picks"]


def load_embedder():
    """질문을 벡터로 바꾸는 임베딩 모델을 불러오는 함수"""
    return Llama(model_path=EMBED_PATH, embedding=True, pooling_type=llama_cpp.LLAMA_POOLING_TYPE_MEAN, n_ctx=2048, n_batch=2048, n_ubatch=2048,
                 n_gpu_layers=0, n_threads=GEMMA_THREADS, verbose=False)


def load_corpus(short):
    """법령의 전체 항목과 미리 만든 벡터를 불러오는 함수"""
    if short == NO_LAW:
        return {"articles": [], "vectors": None}
    with open(f"corpus-{short}.json", encoding="utf-8") as file:
        return {"articles": json.load(file)["articles"], "vectors": np.load(f"vectors-{short}.npy")}


def search_law(embedder, question, corpus):
    """질문과 가장 가까운 법령 항목을 찾는 함수"""
    query = np.array(embedder.embed("task: search result | query: " + question, normalize=True), dtype=np.float32)
    order = np.argsort(-(corpus["vectors"] @ query))
    return [corpus["articles"][int(index)] for index in order[:SEARCH_TOP]]


def answer_question(llm, embedder, question, corpus):
    """질문에 맞는 조문을 찾아 읽고 답을 만드는 함수"""
    articles = search_law(embedder, question, corpus) if corpus["articles"] else []
    text = "\n".join([f"[{article['id']}] {article['title']} {article['text'][:READ_CHARS]}" for article in articles])
    prompt = ANSWER_PROMPT.format(text=text, question=question) if articles else CLOSED_ANSWER_PROMPT.format(question=question)
    ask = f"<|turn>user\n{prompt}<turn|>\n<|turn>model\n"
    answer = llm.create_completion(ask, max_tokens=80, temperature=0, stop=["<turn|>"])["choices"][0]["text"]
    return {"question": question, "answer": answer.strip().replace("**", ""), "articles": articles}


def start_stage(stage):
    """지금 도는 모델 호출을 화면 상태에 적는 함수"""
    STATUS["stage"] = stage
    STATUS["started"] = time.perf_counter()


def end_stage(stage, answer, odds, tokens):
    """모델 호출의 답, 걸린 시간, 답별 확률, 입력 토큰 수를 화면 상태에 적는 함수"""
    STATUS["answers"][stage] = (answer, time.perf_counter() - STATUS["started"], odds, tokens)


def answer_odds(llm, words):
    """방금 호출에서 모델이 낱말마다 매긴 점수를 낱말끼리의 확률로 바꾸는 함수"""
    # 생성이 끝난 뒤 모델의 마지막 점수를 읽기만 하므로 답에는 영향이 없음
    logits = np.ctypeslib.as_array(llama_cpp.llama_get_logits_ith(llm.ctx, -1), (llm.n_vocab(),))
    scores = np.array([logits[llm.tokenize(word.encode("utf-8"), add_bos=False)[0]] for word in words])
    probabilities = np.exp(scores - scores.max())
    return list(zip(words, (probabilities / probabilities.sum()).tolist()))


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


def first_token_bias(llm, words):
    """답의 첫 토큰을 낱말들의 첫 토큰으로 제한하는 logit_bias를 만드는 함수"""
    return {llm.tokenize(word.encode("utf-8"), add_bos=False)[0]: 1000.0 for word in words}


def ask_word(llm, stage, image, question, words):
    """크롭 한 장을 보고 정해진 낱말 중 하나로 답하게 하고 답을 화면 상태에 적는 함수"""
    start_stage(stage)
    content = [image_part(image, 140), {"type": "text", "text": question}]
    response = llm.create_chat_completion(messages=[{"role": "user", "content": content}], max_tokens=1, temperature=0, logit_bias=first_token_bias(llm, words))
    answer = response["choices"][0]["message"]["content"].strip().lower()
    end_stage(stage, answer, answer_odds(llm, words), llm.n_tokens)
    return answer


def recognize(llm, frame, result, in_school_zone):
    """프레임과 크롭을 보고 지금 있는 상황 목록을 만드는 함수"""
    signs = class_strip(frame, result, SIGN_CLASSES)
    # 신호등과 표지 박스가 없으면 프레임 전체를 대신 넣음
    if signs is None:
        signs = frame
    # 고정 크롭을 YOLO 크롭으로 바꾸어 현행 개발용 수치가 22/53에서 31/53으로 오름(2026-11-05)
    road = road_crop(frame)
    lights = class_strip(frame, result, LIGHT_CLASSES)
    people = class_strip(frame, result, PERSON_CLASSES)
    buses = class_strip(frame, result, BUS_CLASSES)
    hydrants = class_strip(frame, result, HYDRANT_CLASSES)
    # 이번 사이클에 부르지 않은 호출의 답이 화면에 남지 않도록 객관식 답만 두고 지움
    STATUS["answers"] = {"choice": STATUS["answers"].get("choice")}
    STATUS["images"] = {"frame": frame, "road": road, "lights": lights, "people": people, "buses": buses, "hydrants": hydrants}
    start_stage("choice")
    content = [image_part(frame, 280), image_part(signs, 140), image_part(road, 140), {"type": "text", "text": RECOGNIZE_PROMPT}]
    response = llm.create_chat_completion(messages=[{"role": "user", "content": content}], max_tokens=20, temperature=0)
    chosen = parse_situations(response["choices"][0]["message"]["content"])
    # 적색, 황색, 보행자는 크롭 질문의 답을 쓰므로 객관식의 답을 화면에도 내지 않음
    names = [SITUATION_NAMES[SITUATIONS.index(situation)] for situation in chosen if situation not in (RED, YELLOW, PEDESTRIAN)]
    end_stage("choice", ", ".join(names) if names else "해당 상황 없음", [], llm.n_tokens)
    found = []
    # 신호 색과 횡단보도는 객관식에서 오인식이 많아 대상만 자른 크롭으로 따로 물음(2026-11-05 개발용 측정, 현행 28/53에서 31/53)
    if lights is not None:
        color = ask_word(llm, "light", lights, LIGHT_PROMPT, LIGHT_WORDS)
        if color == "red":
            found.append(RED)
        if color == "yellow":
            found.append(YELLOW)
    for situation in chosen:
        if situation not in (RED, YELLOW, PEDESTRIAN):
            found.append(situation)
    # 사람 크롭에는 횡단보도가 보이지 않는 경우가 많아 걷는 사람이면 yes로 답하므로 도로 크롭에서 횡단보도를 확인함(2026-11-06 측정)
    if people is not None and ask_word(llm, "pedestrian", people, PEDESTRIAN_PROMPT, YES_NO) == "yes":
        if ask_word(llm, "crosswalk", road, CROSSWALK_PROMPT, YES_NO) == "yes":
            found.append(PEDESTRIAN)
    # 표본에서 비 프레임 27/80이 yes이고 맑음 0/100, 안개 0/24라 앞단 호출 없이 바로 물음(2026-11-07 71784 표본)
    if ask_word(llm, "rain", frame, RAIN_PROMPT, YES_NO) == "yes":
        found.append(RAIN)
    if buses is not None and ask_word(llm, "bus", buses, BUS_PROMPT, YES_NO) == "yes":
        found.append(SCHOOL_BUS)
    # YOLO가 한국의 라바콘과 볼라드도 소화전으로 잡아 물체 종류를 다시 물음(2026-11-07 71784 표본에서 66박스 중 13박스만 hydrant)
    # 답을 한 토큰만 받아 hydrant는 hyd로 돌아옴
    if hydrants is not None and ask_word(llm, "hydrant", hydrants, OBJECT_PROMPT, OBJECT_WORDS).startswith("hyd"):
        found.append(HYDRANT)
    # 제27조 ⑦은 보호구역 안의 신호 없는 횡단보도에만 적용되고, 노면 글자에서 벗어나도 보호구역은 이어지므로 유지 시간 안의 인식을 씀
    if (in_school_zone or SCHOOL_ZONE in found) and lights is None:
        if ask_word(llm, "zebra", road, ZEBRA_PROMPT, YES_NO) == "yes":
            found.append(SCHOOL_CROSSWALK)
    return found


def read_provision(llm, situation, article):
    """조문을 읽어 행동과 지시문을 만드는 함수"""
    # 참조한 항의 내용이 고른 항 뒤에서 잘리지 않도록 항마다 따로 자름
    parts = [(part["title"] + ": " if part["title"] else "") + part["text"] for part in [article] + article["references"]]
    text = " ".join([part[:READ_CHARS] for part in parts])
    # create_completion이 BOS를 스스로 붙이므로 프롬프트에 쓰지 않음
    prompt = f"<|turn>user\n{READ_PROMPT}{READ_EXAMPLES}Situation: {situation}\nProvisions:\n[1] {text or 'none'}\nAnswer:<turn|>\n<|turn>model\n"
    # 행동이 정해진 6개 밖으로 나가지 않도록 첫 토큰을 6개로 제한함
    start_stage("action")
    first = llm.create_completion(prompt, max_tokens=1, temperature=0, logit_bias=first_token_bias(llm, ACTIONS))["choices"][0]["text"].strip()
    code = [code for action, code in ACTIONS.items() if action.startswith(first)][0]
    odds = answer_odds(llm, ACTIONS)
    end_stage("action", code, odds, llm.n_tokens)
    start_stage("instruction")
    question = INSTRUCTION_PROMPT.format(text=text, situation=situation) if text else CLOSED_PROMPT.format(situation=situation)
    ask = f"<|turn>user\n{question}<turn|>\n<|turn>model\n"
    answer = llm.create_completion(ask, max_tokens=30, temperature=0, stop=["\n", "<turn|>"])["choices"][0]["text"]
    # Gemma가 문장을 마크다운 강조로 감싸거나 강조 뒤에 영어 번역을 붙이기도 함
    instruction = answer.strip().strip("*").split("**")[0]
    end_stage("instruction", instruction, [], llm.n_tokens)
    STATUS["odds"][instruction] = odds
    return {"law": article["id"], "action": code, "instruction": instruction, "text": text}


def watch(llm, laws, shared, embedder, corpora):
    """최신 프레임으로 상황을 인식하고 조문을 읽어 규칙 목록을 계속 갱신하는 함수"""
    # 같은 법령과 상황은 입력이 같아 출력도 같으므로 처음 읽은 결과를 다시 씀(2026-11-05 27개 상황 일치 측정)
    readings = {}
    while True:
        # Llama 객체는 동시에 부를 수 없어 판단 주기 사이에 질문을 처리함
        if shared["question"] is not None:
            question = shared["question"]
            shared["question"] = None
            shared["thinking"] = question
            shared["answer"] = answer_question(llm, embedder, question, corpora[shared["law"]])
            shared["answer_time"] = time.perf_counter()
            shared["thinking"] = None
            shared["speech"] = shared["answer"]["answer"]
            continue
        # 묶음을 입력보다 먼저 읽어야 새 묶음에 이전 영상 프레임의 판단이 들어가지 않음
        clip = shared["clip"]
        current = shared["input"]
        if current is None:
            time.sleep(0.05)
            continue
        frame, result = current
        law = shared["law"]
        in_school_zone = time.perf_counter() - clip["last_seen"].get(SCHOOL_ZONE, -HOLD_SECONDS) <= HOLD_SECONDS
        found = recognize(llm, frame, result, in_school_zone)
        now = time.perf_counter()
        for situation in found:
            clip["last_seen"][situation] = now
        active = [situation for situation, seen in clip["last_seen"].items() if now - seen <= HOLD_SECONDS and situation in laws[law]]
        # 조문을 읽기 전에 올려 처음 찾은 조문이 읽기 시간만큼 늦게 보이지 않게 함
        nodes = [{"situation": situation, "article": laws[law][situation], "fresh": situation in found,
                  "instruction": readings[(law, situation)]["instruction"] if (law, situation) in readings else None}
                 for situation in active if laws[law][situation]["text"]]
        clip["nodes"] = nodes
        rules = []
        for situation in active:
            if (law, situation) not in readings:
                readings[(law, situation)] = read_provision(llm, situation, laws[law][situation])
                for node in nodes:
                    if node["situation"] == situation:
                        node["instruction"] = readings[(law, situation)]["instruction"]
            rules.append(readings[(law, situation)])
        clip["rules"] = rules
        STATUS["stage"] = "wait"
