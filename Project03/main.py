"""주행 영상에서 법 조문이 필요한 엣지 케이스를 찾아 지시문과 근거 조문을 보여 주는 실행 파일"""

import argparse
import json
import threading
import time

import cv2
from ultralytics import YOLO

from detect import CLASSES, IMAGE_SIZE, scene_state
from edge import active_edge_cases
from gemma import load_gemma, watch
from hud import button_boxes, draw, draw_compare_panel, draw_panel
from retrieve import Retriever

FRAME_SIZE = (1280, 720)
STABLE_FRAMES = 5  # 한두 프레임만 잡힌 오탐을 버리려고 5프레임 연속 탐지된 상태만 인정함
HOLD_SECONDS = 2.0  # 신호등이 화면 밖으로 나가도 조문을 읽을 수 있도록 2초 동안 유지함
# 시험용 세트에서 간격 4초가 낡은 판단으로 정답 4개를 놓치고 오판 2개를 만들어 기본값은 쉬지 않음(2026-11-03)
GEMMA_INTERVAL = 0.0
WINDOW = "Driving Rule Retrieval"
VIDEOS = [("원주", "videos/wonju-demo.mp4"), ("비벌리힐스", "videos/beverly-demo.mp4"), ("샌프란시스코", "videos/sf-demo.mp4")]
CORPORA = [("현행 도로교통법", "corpus-current.json"), ("2021년 도로교통법", "corpus-2021.json"),
           ("캘리포니아 차량법", "corpus-california.json")]
COMPARE_PATH = "compare-instructions.json"
EMPTY_SCENE = {"facts": set(), "corpus": None, "sign": "", "article": None, "instruction": None}

parser = argparse.ArgumentParser()
parser.add_argument("video", nargs="?", default=VIDEOS[0][1])
parser.add_argument("--corpus", default=CORPORA[0][1])
parser.add_argument("--model", default="../src/models/YOLO/yolo11n.pt")
parser.add_argument("--interval", type=float, default=GEMMA_INTERVAL)
args = parser.parse_args()

llm = load_gemma()
retrievers = [Retriever(path) for name, path in CORPORA]
# 비교 모드에서만 build_compare.py가 미리 만든 세 법령 지시문을 보여 줌
with open(COMPARE_PATH, encoding="utf-8") as file:
    compare = json.load(file)

video_paths = [path for name, path in VIDEOS]
corpus_paths = [path for name, path in CORPORA]
selected = {"video": video_paths.index(args.video), "corpus": corpus_paths.index(args.corpus)}
# 법령 버튼 줄의 마지막 버튼이 비교 모드이고 번호는 len(CORPORA)임
buttons = button_boxes(FRAME_SIZE[0], {"video": [name for name, path in VIDEOS], "corpus": [name for name, path in CORPORA] + ["비교"]})


def on_click(event, x, y, flags, param):
    """버튼을 누르면 고른 영상이나 법령 번호를 바꾸는 함수"""
    if event == cv2.EVENT_LBUTTONDOWN:
        for group, index, name, (x1, y1, x2, y2) in buttons:
            if x1 <= x <= x2 and y1 <= y <= y2:
                selected[group] = index


# 지시문은 맥락 스레드만 만들어 두 곳이 같은 Gemma를 동시에 쓰지 않게 함
# active는 메인 루프가 찾은 엣지 케이스 이름 목록이고 written은 맥락 스레드가 만든 (법령 번호, 이름)별 조항과 지시문임
shared = {"frame": None, "corpus": selected["corpus"], "scene": EMPTY_SCENE, "interval": args.interval, "active": [], "written": {}}
threading.Thread(target=watch, args=(llm, retrievers, shared), daemon=True).start()
model = YOLO(args.model)
# 마우스 콜백은 이름 붙은 창에만 달 수 있어 첫 imshow 전에 창을 만듦
cv2.namedWindow(WINDOW)
cv2.setMouseCallback(WINDOW, on_click)

current_video = None
displayed_fps = 0.0
panel_key = None

while True:
    start_time = time.perf_counter()

    if selected["video"] != current_video:
        current_video = selected["video"]
        cap = cv2.VideoCapture(VIDEOS[current_video][1])
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        hold_frames = int(HOLD_SECONDS * video_fps)
        streak, last_seen, frame_index = {}, {}, 0
        shared["frame"], shared["scene"] = None, EMPTY_SCENE

    ret, frame = cap.read()

    # 영상이 끝나면 같은 영상을 처음부터 다시 엶
    if not ret:
        current_video = None
        continue

    frame = cv2.resize(frame, FRAME_SIZE)
    # draw가 이 프레임 위에 박스를 그리므로 Gemma에는 복사본을 넘김
    shared["frame"] = (current_video, frame.copy())
    # 비교 모드에서는 맥락 스레드가 지시문을 만들지 않도록 법령 번호를 None으로 넘김
    shared["corpus"] = selected["corpus"] if selected["corpus"] < len(CORPORA) else None
    results = model.predict(source=frame, conf=0.4, iou=0.5, verbose=False, classes=CLASSES, imgsz=IMAGE_SIZE)
    found, boxes = scene_state(results[0], frame)

    for fact in found | set(streak):
        if fact in found:
            streak[fact] = streak.get(fact, 0) + 1
        else:
            streak[fact] = 0
        if streak[fact] >= STABLE_FRAMES:
            last_seen[fact] = frame_index
    held = {fact for fact in last_seen if frame_index - last_seen[fact] <= hold_frames}
    scene = shared["scene"]
    corpus = selected["corpus"]
    names = active_edge_cases(held | scene["facts"])
    if names != shared["active"]:
        print("엣지 케이스", names)
    shared["active"] = names
    panel, ready = [], []
    if corpus == len(CORPORA):
        # 비교 모드 패널 항목은 엣지 케이스 이름과 세 법령의 법령 이름, 조항, 지시문으로 이루어짐
        for name in names:
            panel.append((name, [(corpus_name, compare[path][name]["article"], compare[path][name]["instruction"]) for corpus_name, path in CORPORA]))
    else:
        # 패널 항목은 엣지 케이스 이름, 근거 조항, 지시문이고 맥락 스레드가 만들기 전에는 조항과 지시문이 None임
        for name in names:
            article, instruction = shared["written"].get((corpus, name), (None, None))
            panel.append((name, article, instruction))
            ready.append(instruction is not None)
        # 법령 버튼을 누른 직후에는 이전 코퍼스로 찾은 표지판 조항을 보여 주지 않음
        if scene["article"] is not None and scene["corpus"] == corpus:
            panel.append(("표지판 " + scene["sign"], scene["article"], scene["instruction"]))
    frame_index += 1

    elapsed_time = time.perf_counter() - start_time
    current_fps = 1.0 / elapsed_time

    if displayed_fps == 0:
        displayed_fps = current_fps
    else:
        displayed_fps = 0.9 * displayed_fps + 0.1 * current_fps

    # 패널 글자는 엣지 케이스, 지시문 준비 여부, 버튼 선택이 바뀔 때만 다시 그림
    key = (tuple(item[0] for item in panel), tuple(ready), selected["video"], corpus)
    if key != panel_key:
        panel_key = key
        if corpus == len(CORPORA):
            panel_image = draw_compare_panel(FRAME_SIZE[1], FRAME_SIZE[0], panel, buttons, selected)
        else:
            panel_image = draw_panel(FRAME_SIZE[1], FRAME_SIZE[0], panel, buttons, selected)
    output_frame = draw(frame, boxes, panel_image, displayed_fps)

    cv2.imshow(WINDOW, output_frame)

    # 처리가 영상보다 빠르면 남는 시간만큼 기다려 원래 속도로 재생함
    wait_ms = max(1, int(1000 / video_fps - (time.perf_counter() - start_time) * 1000))
    if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
