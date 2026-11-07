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
# 상황 문장으로 요약 지시문을 검색해 기본 규칙 밖의 관련 규칙 후보를 이만큼 가져옴(2026-11-07 상황 19개 확인)
RULE_TOP = 4
# 상황 문장만 보면 장면에 없는 조건의 규칙(보행자우선도로, 밤)도 골라 작은 이미지를 함께 보여 줌(2026-11-07 사용자 지시, Jetson 70토큰 이미지 호출 0.52초)
CHOOSE_TOKENS = 70
# 요약 지시문은 빌드 단계에서 Sonnet이 조문을 읽고 만든 것이고, Gemma는 영어 문장으로 고르고 화면에는 한국어 문장을 냄(2026-11-07 사용자 지시)
CHOOSE_PROMPT = (
    "A self-driving car sees this image: {situation}.\nTraffic rules found for this scene:\n{options}0. none of these rules applies now\n"
    "Which rules must the car follow right now? Reply with only the numbers of all rules that apply, separated by commas, or 0."
)
# 정해 둔 상황 목록 밖의 장면도 찾도록 객관식 답 뒤에 이어 물어 Gemma가 장면을 짧은 영어 구절로 적음(2026-11-07 사용자 지시, PC 실험 fD와 cont)
# 문구는 이미지 3장 호출 때와 같음. 목록 밖의 구체 대상만 묻는 문구 두 가지는 범주 이름이나 설명 문장을 내어 잘못된 지시가 늘었음(2026-11-07 PC 측정)
FREE_PROMPT = (
    "You are the eyes of a self-driving car. Name what traffic law cares about in this road scene: places such as tunnels, bridges, "
    "school zones, crossings or work zones, and signs, road markings, lights and road users ahead. "
    "Reply with at most three short English phrases separated by semicolons, or none if it is ordinary driving.\n"
    "Examples:\n"
    "Dark walls and ceiling lamps surround the road. Answer: inside a road tunnel\n"
    "Railings and open water lie on both sides of the road. Answer: on a bridge over a river\n"
    "Red pavement with 어린이 보호구역 painted on it. Answer: child protection school zone\n"
    "Barriers close a lane and a gate arm is down across rails. Answer: railroad crossing with a lowered gate\n"
    "A sign by the road shows the number 40. Answer: speed limit sign 40\n"
    "A green light ahead and cars in front. Answer: none\n"
    "Answer:"
)
# 앞 호출의 답 뒤에 모델 차례를 닫고 사용자 차례와 새 모델 차례를 여는 Gemma 4 대화 형식(llama-cpp-python 0.3.26 이상의 GGUF 템플릿)
NEXT_TURN = "<turn|>\n<|turn>user\n{prompt}<turn|>\n<|turn>model\n"
FREE_PHRASES = 3
# 장면에 있는 대상이 아니라 "특별한 것 없음"이나 행동을 적은 구절은 검색하면 장면과 관계없는 규칙을 끌어와 불용어처럼 뺌(2026-11-08 사용자 지시, 개발용 47장에서 나온 구절)
STOP_PHRASES = ["none", "ordinary driving", "headlights on"]
STOP_PREFIXES = ["no specific", "no special", "no traffic"]
# 한 프레임의 구절 묶음을 상황 하나로 다루고, 검색할 때 다시 구절로 나눔
PHRASE_JOIN = "; "
NO_LAW = "none"
NO_ARTICLE = {"id": "조문 없음", "title": "", "text": "", "references": []}
# 인식이 완벽하다고 가정한 개발용 수치는 120자 44/53, 240자 46/53, 400자 39/53이었음(2026-11-05)
READ_CHARS = 240
# Jetson 판단 약 5초와 처음 읽는 조문 약 5초 동안 지시가 사라지지 않도록 다음 판단까지 유지함(2026-11-07 Jetson 측정, 사용자 지시)
HOLD_SECONDS = 8.0

# watch가 판단 중인 묶음, 법령, 모델을 담아 publish가 쓰고, 영상이 바뀌어 버려진 묶음인지 가리는 데도 씀
WATCHING = {}
# 화면이 호출별 처리 상태를 그리는 값이고 판단에는 쓰지 않음
STATUS = {"stage": "wait", "started": 0.0, "images": {}, "answers": {}, "odds": {}, "outputs": 0}


def load_gemma():
    """비전 모듈을 붙인 Gemma를 불러오는 함수"""
    # Jetson에서 CUDA 그래프 캡처가 YOLO의 torch.cuda.synchronize와 충돌해 프로세스가 멈춰 끔(2026-11-07 Jetson 측정)
    os.environ["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    handler = Gemma4ChatHandler(clip_model_path=VISION_PATH, verbose=False)
    # 기본값은 CPU 코어 수만큼 스레드를 써서 같은 CPU의 YOLO가 20ms에서 358ms로 느려졌고, 2개로 줄이면 110ms였음(2026-11-03 Mac 측정)
    # 비전 모듈도 n_threads 값을 그대로 씀
    # flash attention을 켜면 Jetson에서 판단 한 번이 5.08초에서 3.65초로 줄고 같은 7프레임의 인식 결과는 같았음(2026-11-07 측정)
    return Llama(model_path=LLM_PATH, chat_handler=handler, n_gpu_layers=-1, n_ctx=4096, n_batch=512, n_ubatch=512,
                 n_threads=GEMMA_THREADS, n_threads_batch=GEMMA_THREADS, flash_attn=True, verbose=False)


def load_law(short):
    """법령의 상황별 조문 사전을 불러오는 함수"""
    if short == NO_LAW:
        return {situation: NO_ARTICLE for situation in SITUATIONS + CROP_SITUATIONS}
    with open(f"laws/index-{short}.json", encoding="utf-8") as file:
        return json.load(file)["picks"]


def load_embedder():
    """질문을 벡터로 바꾸는 임베딩 모델을 불러오는 함수"""
    return Llama(model_path=EMBED_PATH, embedding=True, pooling_type=llama_cpp.LLAMA_POOLING_TYPE_MEAN, n_ctx=2048, n_batch=2048, n_ubatch=2048,
                 n_gpu_layers=0, n_threads=GEMMA_THREADS, verbose=False)


def load_corpus(short):
    """법령의 전체 항목, 요약 지시문과 미리 만든 벡터를 불러오는 함수"""
    if short == NO_LAW:
        return {"articles": [], "vectors": None, "rules": [], "rule_vectors": None}
    with open(f"laws/corpus-{short}.json", encoding="utf-8") as file:
        articles = json.load(file)["articles"]
    with open(f"laws/rules-{short}.json", encoding="utf-8") as file:
        rules = json.load(file)
    return {"articles": articles, "vectors": np.load(f"laws/vectors-{short}.npy"), "rules": rules, "rule_vectors": np.load(f"laws/rule-vectors-{short}.npy")}


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
    # 비전 모듈은 48×48 칸 하나를 토큰 하나로 읽으므로 넓이를 토큰 수만큼의 칸으로 줄임
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


def judging_old_clip():
    """판단 중인 묶음이 영상 전환으로 버려졌는지 돌려주는 함수"""
    return "clip" in WATCHING and WATCHING["shared"]["clip"] is not WATCHING["clip"]


def ask_word(llm, stage, image, question, words):
    """크롭 한 장을 보고 정해진 낱말 중 하나로 답하게 하고 답을 화면 상태에 적는 함수"""
    # 영상이 바뀌면 결과가 버려지므로 남은 크롭 질문을 부르지 않아 새 영상 판단을 앞당김
    if judging_old_clip():
        return ""
    start_stage(stage)
    content = [image_part(image, 140), {"type": "text", "text": question}]
    response = llm.create_chat_completion(messages=[{"role": "user", "content": content}], max_tokens=1, temperature=0, logit_bias=first_token_bias(llm, words))
    answer = response["choices"][0]["message"]["content"].strip().lower()
    end_stage(stage, answer, answer_odds(llm, words), llm.n_tokens)
    return answer


def find_rules(llm, embedder, frame, situation, article, corpus, exclude=()):
    """빌드 때 고른 조문의 요약을 기본 규칙으로 두고, 검색한 관련 규칙 가운데 Gemma가 고른 것을 덧붙이는 함수"""
    start_stage("search")
    # 목록 상황은 문장 하나이고, 자유 구절 묶음은 구절마다 검색해 후보를 모음
    queries = [np.array(embedder.embed("task: search result | query: " + phrase, normalize=True), dtype=np.float32) for phrase in situation.split(PHRASE_JOIN)]
    allowed = np.array([(rule["id"], rule["instruction"]) not in exclude for rule in corpus["rules"]])
    # 한 조문에서 조건별 요약이 여러 개 나오므로 상황 문장과 가장 가까운 요약을 기본 규칙으로 씀
    own = [position for position, rule in enumerate(corpus["rules"]) if rule["id"] == article["id"]]
    found = [corpus["rules"][max(own, key=lambda position: corpus["rule_vectors"][position] @ queries[0])]] if own else []
    others = []
    for query in queries:
        scores = np.where(allowed, corpus["rule_vectors"] @ query, -np.inf)
        for position in np.argsort(-scores)[:RULE_TOP]:
            rule = corpus["rules"][int(position)]
            if allowed[int(position)] and rule not in found and rule not in others:
                others.append(rule)
    end_stage("search", f"후보 {len(others)}개", [], 0)
    if others:
        start_stage("choose")
        options = "".join([f"{number}. {rule['instruction_en']} (applies to: {rule['scene']})\n" for number, rule in enumerate(others, 1)])
        content = [image_part(frame, CHOOSE_TOKENS), {"type": "text", "text": CHOOSE_PROMPT.format(situation=situation, options=options)}]
        answer = llm.create_chat_completion(messages=[{"role": "user", "content": content}], max_tokens=12, temperature=0)["choices"][0]["message"]["content"]
        for part in answer.replace(" ", "").split(","):
            if part.isdigit() and 0 < int(part) <= len(others) and others[int(part) - 1] not in found:
                found.append(others[int(part) - 1])
        end_stage("choose", answer.strip(), [], llm.n_tokens)
    return [{"law": rule["id"], "title": rule["title"], "action": rule["action"], "instruction": rule["instruction"], "text": rule["text"][:READ_CHARS]} for rule in found]


def publish(found):
    """판단 중에 찾은 상황을 유지 목록에 넣고 규칙을 찾아 선택한 법령의 규칙 목록과 법령 비교 열을 바로 화면에 넘기는 함수"""
    # 예열과 평가에서는 판단 중인 묶음이 없고, 버려진 묶음은 화면에 나오지 않아 넘기지 않음
    if "clip" not in WATCHING or judging_old_clip():
        return
    # recent는 최근 HOLD_SECONDS 안에 본 상황이고, rules는 화면과 VLA 출력으로 나가는 목록임
    clip, law, laws, readings = WATCHING["clip"], WATCHING["law"], WATCHING["laws"], WATCHING["readings"]
    now = time.perf_counter()
    for situation in found:
        clip["last_seen"][situation] = now
    recent = recent_situations(clip)
    active = recent
    # 이미 찾은 규칙은 바로 올리고, 새 상황의 규칙은 찾을 때마다 뒤에 붙여 가장 새 규칙으로 보이게 함
    pending = [situation for situation in active if (law, situation) not in readings]
    pairs = [(situation, rule) for situation in active if situation not in pending for rule in readings[(law, situation)]]
    clip["rules"] = unique_rules(pairs)
    clip["compare"] = compare_rules(recent)
    for situation in pending:
        readings[(law, situation)] = find_law_rules(law, situation)
        pairs = pairs + [(situation, rule) for rule in readings[(law, situation)]]
        clip["rules"] = unique_rules(pairs)
        clip["compare"] = compare_rules(recent)


def find_law_rules(law, situation):
    """한 법령과 상황의 규칙을 찾는 함수"""
    if law == NO_LAW:
        return [read_provision(WATCHING["llm"], situation, NO_ARTICLE)]
    corpus = WATCHING["corpora"][law]
    if is_free(situation):
        # 어디서나 적용되는 일반 규칙을 빼야 장면에 맞지 않는 지시가 줄고, 목록 상황에서 이미 찾은 규칙은 다시 고르지 않음(2026-11-07 PC 실험, 현행 잘못된 지시 13에서 9)
        taken = {(rule["law"], rule["instruction"]) for situation_seen in recent_situations(WATCHING["clip"]) if not is_free(situation_seen)
                 for rule in WATCHING["readings"].get((law, situation_seen), [])}
        general = {(rule["id"], rule["instruction"]) for rule in corpus["rules"] if rule["general"]}
        return find_rules(WATCHING["llm"], WATCHING["embedder"], WATCHING["frame"], situation, {"id": None}, corpus, general | taken)
    # 빌드 때 기본 조문을 정하지 못한 상황도 관련 규칙은 검색해 Gemma가 고르게 함(2026-11-07 사용자 지시, 캘리포니아 통학버스 §22454)
    article = WATCHING["laws"][law].get(situation, {"id": None})
    return find_rules(WATCHING["llm"], WATCHING["embedder"], WATCHING["frame"], situation, article, corpus)


def is_free(situation):
    """상황이 Gemma가 적은 자유 구절 묶음인지 가리는 함수"""
    return situation not in SITUATIONS and situation not in CROP_SITUATIONS


def recent_situations(clip):
    """최근 HOLD_SECONDS 안에 본 상황 목록을 만드는 함수"""
    now = time.perf_counter()
    return [situation for situation, seen in clip["last_seen"].items() if now - seen <= HOLD_SECONDS]


def compare_rules(recent):
    """법령마다 최근 상황의 규칙과 아직 찾지 못한 상황이 있는지를 담은 비교 열을 만드는 함수"""
    laws, readings = WATCHING["laws"], WATCHING["readings"]
    return {short: {"rules": unique_rules([(situation, rule) for situation in recent for rule in readings.get((short, situation), [])]),
                    "pending": any((short, situation) not in readings for situation in recent)}
            for short in laws}


def fill_compare():
    """선택하지 않은 법령의 규칙도 찾아 법령 비교 열을 채우는 함수"""
    # 선택한 법령의 규칙을 늦추지 않도록 판단 주기 끝에 다른 법령을 찾음
    if "clip" not in WATCHING or judging_old_clip():
        return
    clip, laws, readings = WATCHING["clip"], WATCHING["laws"], WATCHING["readings"]
    recent = recent_situations(clip)
    for short in laws:
        for situation in recent:
            if (short, situation) not in readings:
                if judging_old_clip():
                    return
                readings[(short, situation)] = find_law_rules(short, situation)
                clip["compare"] = compare_rules(recent)


def unique_rules(pairs):
    """여러 상황이 같은 규칙을 찾아도 한 번만 남긴 규칙 목록을 만드는 함수"""
    rules = []
    for situation, rule in pairs:
        if rule not in rules:
            rules.append(rule)
    return rules


def recognize(llm, frame, result, in_school_zone):
    """프레임과 크롭을 보고 지금 있는 상황 목록을 만드는 함수"""
    # 객관식 한 번으로 장면 전체를 보고, 객관식이 약한 상황은 크롭 질문으로 따로 확인함
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
    # 다른 호출이 KV를 지우기 전에, 객관식 호출의 이미지 연산 결과를 다시 쓰도록 바로 이어 물음(2026-11-07 PC 실험, 구절 호출 0.28초에서 0.09초)
    phrases = describe_scene(llm)
    found = []
    # 신호 색과 횡단보도는 객관식에서 오인식이 많아 대상만 자른 크롭으로 따로 물음(2026-11-05 개발용 측정, 현행 28/53에서 31/53)
    if lights is not None:
        color = ask_word(llm, "light", lights, LIGHT_PROMPT, LIGHT_WORDS)
        if color == "red":
            found.append(RED)
        if color == "yellow":
            found.append(YELLOW)
        publish(found)
    for situation in chosen:
        if situation not in (RED, YELLOW, PEDESTRIAN):
            found.append(situation)
    # 객관식에서 찾은 상황을 크롭 질문을 기다리지 않고 바로 화면에 넘김(2026-11-07 사용자 지시, Jetson 객관식 3.4초와 크롭 질문 한 번 0.5~0.8초)
    publish(found)
    # 사람 크롭에는 횡단보도가 보이지 않는 경우가 많아 걷는 사람이면 yes로 답하므로 도로 크롭에서 횡단보도를 확인함(2026-11-06 측정)
    if people is not None and ask_word(llm, "pedestrian", people, PEDESTRIAN_PROMPT, YES_NO) == "yes":
        if ask_word(llm, "crosswalk", road, CROSSWALK_PROMPT, YES_NO) == "yes":
            found.append(PEDESTRIAN)
            publish(found)
    # 표본에서 비 프레임 27/80이 yes이고 맑음 0/100, 안개 0/24라 앞단 호출 없이 바로 물음(2026-11-07 71784 표본)
    if ask_word(llm, "rain", frame, RAIN_PROMPT, YES_NO) == "yes":
        found.append(RAIN)
        publish(found)
    if buses is not None and ask_word(llm, "bus", buses, BUS_PROMPT, YES_NO) == "yes":
        found.append(SCHOOL_BUS)
        publish(found)
    # YOLO가 한국의 라바콘과 볼라드도 소화전으로 잡아 물체 종류를 다시 물음(2026-11-07 71784 표본에서 66박스 중 13박스만 hydrant)
    # 답을 한 토큰만 받아 hydrant는 hyd로 돌아옴
    if hydrants is not None and ask_word(llm, "hydrant", hydrants, OBJECT_PROMPT, OBJECT_WORDS).startswith("hyd"):
        found.append(HYDRANT)
        publish(found)
    # 제27조 ⑦은 보호구역 안의 신호 없는 횡단보도에만 적용되고, 노면 글자에서 벗어나도 보호구역은 이어지므로 유지 시간 안의 인식을 씀
    if (in_school_zone or SCHOOL_ZONE in found) and lights is None:
        if ask_word(llm, "zebra", road, ZEBRA_PROMPT, YES_NO) == "yes":
            found.append(SCHOOL_CROSSWALK)
            publish(found)
    # 목록 상황의 규칙을 먼저 화면에 내고, 마지막에 자유 구절의 규칙을 찾음
    if phrases:
        found.append(PHRASE_JOIN.join(phrases))
        publish(found)
    return found


def describe_scene(llm):
    """직전 객관식 호출에 이어 장면에서 교통법규와 관련된 것을 짧은 영어 구절 목록으로 적게 하는 함수"""
    start_stage("free")
    # 채팅 호출은 매번 KV를 지우므로 KV에 남은 토큰 뒤에 질문을 붙인 토큰 목록을 넘겨 앞부분을 다시 쓰게 함
    # 이미지 자리의 토큰 번호는 0이지만 KV에는 이미지 임베딩이 남아 있음(2026-11-07 PC 확인, 처음부터 다시 넣은 토큰열과 8/8 일치)
    head = llm.input_ids[:llm.n_tokens].tolist()
    tokens = head + llm.tokenize(NEXT_TURN.format(prompt=FREE_PROMPT).encode("utf-8"), add_bos=False, special=True)
    answer = llm.create_completion(tokens, max_tokens=24, temperature=0, stop=["<turn|>"])["choices"][0]["text"]
    phrases = []
    for part in answer.split(";"):
        phrase = part.strip(" .\n").lower()
        if phrase and phrase not in STOP_PHRASES and not phrase.startswith(tuple(STOP_PREFIXES)) and phrase not in phrases:
            phrases.append(phrase)
    phrases = phrases[:FREE_PHRASES]
    end_stage("free", ", ".join(phrases) if phrases else "없음", [], llm.n_tokens)
    return phrases


def read_provision(llm, situation, article):
    """조문을 읽어 행동과 지시문을 만드는 함수"""
    # 참조한 항의 내용이 고른 항 뒤에서 잘리지 않도록 항마다 따로 자름
    parts = [(part["title"] + ": " if part["title"] else "") + part["text"] for part in [article] + article["references"]]
    text = " ".join([part[:READ_CHARS] for part in parts])
    # create_completion이 BOS를 스스로 붙이므로 프롬프트에 쓰지 않음
    prompt = f"<|turn>user\n{READ_PROMPT}{READ_EXAMPLES}Situation: {situation}\nProvisions:\n[1] {text or 'none'}\nAnswer:<turn|>\n<|turn>model\n"
    # 첫 호출은 행동 코드를 고르고, 두 번째 호출은 한국어 지시문을 씀
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
        WATCHING.update({"shared": shared, "clip": clip, "law": law, "laws": laws, "llm": llm, "readings": readings, "embedder": embedder, "corpora": corpora, "frame": frame})
        found = recognize(llm, frame, result, in_school_zone)
        fill_compare()
        # 아무것도 찾지 못한 주기에도 만료된 규칙을 지우려고 마지막에 한 번 더 넘김
        publish(found)
        WATCHING.pop("clip")
        STATUS["stage"] = "wait"
