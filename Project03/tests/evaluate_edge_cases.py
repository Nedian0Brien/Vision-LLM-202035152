"""엣지 케이스 판정, 조항 검색, 지시문을 개발용 또는 시험용 세트에서 검색 없는 Gemma 기준선과 함께 재는 평가 스크립트"""

import json
import sys

import cv2
from ultralytics import YOLO

sys.path.insert(0, ".")
from detect import CLASSES, IMAGE_SIZE, scene_state  # noqa: E402
from edge import ACTIONS, EDGE_CASES, active_edge_cases, sentences  # noqa: E402
from gemma import ask, load_gemma, read_scene, write_instruction  # noqa: E402
from retrieve import Retriever  # noqa: E402
from tests.edge_gold import GOLD  # noqa: E402

FRAME_SIZE = (1280, 720)
STABLE_FRAMES = 5  # main.py와 같이 YOLO 사실은 5프레임 연속일 때 인정함
GEMMA_SECONDS = 2.5  # Mac에서 Gemma 판단 한 번에 2.3~2.5초가 걸려 바로 앞 판단은 2.5초 전 프레임으로 봄
BASELINE_PROMPT = (
    "너는 자율주행 차량의 주행 규칙 안내기다. 주어진 법령을 따르는 운전자가 주어진 장면에서 할 행동 종류를 "
    + ", ".join(ACTIONS) + " 중 하나만 골라 그 낱말만 쓴다."
)


def read_frames(video, seconds, interval):
    """정답 시각보다 판단 시간과 간격만큼 앞선 프레임과 정답 시각에서 끝나는 연속 프레임을 main.py와 같은 크기로 꺼내는 함수"""
    cap = cv2.VideoCapture("videos/" + video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 영상 시작 직후 정답은 앞선 판단이 없어 첫 프레임으로 대신함
    cap.set(cv2.CAP_PROP_POS_MSEC, max(seconds - GEMMA_SECONDS - interval, 0) * 1000)
    ret, earlier = cap.read()
    cap.set(cv2.CAP_PROP_POS_MSEC, (seconds - (STABLE_FRAMES - 1) / fps) * 1000)
    frames = []
    for i in range(STABLE_FRAMES):
        ret, frame = cap.read()
        frames.append(cv2.resize(frame, FRAME_SIZE))
    cap.release()
    return cv2.resize(earlier, FRAME_SIZE), frames


split = sys.argv[1]
interval = float(sys.argv[2])
labels = json.load(open(f"tests/edge_labels_{split}.json"))
llm = load_gemma()
model = YOLO("../src/models/YOLO/yolo11n.pt")
counts = {name: {"tp": 0, "fp": 0, "fn": 0} for name in EDGE_CASES}
negatives, false_alarms = 0, 0
rows = []
for label in labels:
    earlier, frames = read_frames(label["video"], label["time"], interval)
    found = None
    for frame in frames:
        facts, boxes = scene_state(model.predict(source=frame, conf=0.4, iou=0.5, verbose=False, classes=CLASSES, imgsz=IMAGE_SIZE)[0], frame)
        found = facts if found is None else found & facts
    seen_before, sign = read_scene(llm, earlier)
    seen_now, sign = read_scene(llm, frames[-1])
    # gemma.watch와 같이 두 번의 판단 중 한 번이라도 예인 항목을 인정함
    seen = seen_before | seen_now
    predicted = set(active_edge_cases(found | seen))
    gold = set(label["edges"])
    for name in predicted & gold:
        counts[name]["tp"] += 1
    for name in predicted - gold:
        counts[name]["fp"] += 1
    for name in gold - predicted:
        counts[name]["fn"] += 1
    if not gold:
        negatives += 1
        false_alarms += bool(predicted)
    rows.append({"video": label["video"], "time": label["time"], "gold": sorted(gold), "predicted": sorted(predicted),
                 "yolo": sorted(found), "gemma": sorted(seen), "gemma_before": sorted(seen_before), "gemma_now": sorted(seen_now),
                 "sign": sign})
    print(label["video"], label["time"], "정답", sorted(gold), "판정", sorted(predicted), flush=True)

print(f"\n{'엣지 케이스':<14}{'정답':>5}{'맞힘':>5}{'오판':>5}")
for name, c in counts.items():
    if c["tp"] + c["fp"] + c["fn"]:
        print(f"{name:<14}{c['tp'] + c['fn']:>5}{c['tp']:>5}{c['fp']:>5}")
tp = sum(c["tp"] for c in counts.values())
fp = sum(c["fp"] for c in counts.values())
fn = sum(c["fn"] for c in counts.values())
print(f"\n판정 재현율 {tp}/{tp + fn} = {tp / (tp + fn):.0%}")
print(f"판정 정밀도 {tp}/{tp + fp} = {tp / max(tp + fp, 1):.0%}")
print(f"음성 프레임 오판율 {false_alarms}/{negatives} = {false_alarms / negatives:.0%}")

# 지시문은 엣지 케이스 목록의 짝수 번째를 개발용, 홀수 번째를 시험용으로 씀
names = list(EDGE_CASES)[0::2] if split == "dev" else list(EDGE_CASES)[1::2]
written = []
for corpus_path, gold in GOLD.items():
    retriever = Retriever(corpus_path)
    for name in names:
        provisions, key, action = gold[name]
        if not provisions:
            continue
        article = retriever.search(retriever.queries[name], 1)[0][1]
        instruction = write_instruction(llm, name, article)
        messages = [{"role": "system", "content": BASELINE_PROMPT}, {"role": "user", "content": f"법령: {retriever.name}\n장면: {name}"}]
        baseline = ask(llm, messages, 20)
        written.append({"corpus": corpus_path, "name": name, "article": article["id"] + " " + article["title"],
                        "found": (article["id"] + " " + article["title"]).strip() in provisions,
                        "sentence_count": len(sentences(article["text"])),
                        "hit": any(key in line for line in instruction["문장"]), "action": action, "baseline": baseline, **instruction})
        print(corpus_path, name, instruction, "기준선", baseline, flush=True)

total = len(written)
found_count = sum(w["found"] for w in written)
multi = [w for w in written if w["sentence_count"] > 1]
print(f"\n검색 정답률 {found_count}/{total} = {found_count / total:.0%}")
print(f"정답 문장 포함 {sum(w['hit'] for w in written)}/{total}, 여러 문장 조항만 {sum(w['hit'] for w in multi)}/{len(multi)}")
print(f"여러 문장 조항에서 고른 문장 수 평균 {sum(len(w['문장']) for w in multi) / max(len(multi), 1):.2f} / 조항 문장 수 평균 "
      f"{sum(w['sentence_count'] for w in multi) / max(len(multi), 1):.2f}")
print(f"행동 종류 일치 {sum(w['행동 종류'] == w['action'] for w in written)}/{total}")
print(f"검색 없는 Gemma 행동 종류 일치 {sum(w['action'] in w['baseline'] for w in written)}/{total}")
if len(sys.argv) > 3:
    json.dump({"frames": rows, "instructions": written}, open(sys.argv[3], "w"), ensure_ascii=False, indent=1)
