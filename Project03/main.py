"""주행 영상에서 법 조문이 필요한 엣지 케이스를 찾아 지시문과 근거 조문을 보여 주는 실행 파일"""

import argparse
import threading
import time

import cv2
from ultralytics import YOLO

from detect import CLASSES, IMAGE_SIZE, scene_state
from edge import EDGE_CASES, active_edge_cases
from gemma import load_gemma, watch, write_instruction
from hud import button_boxes, draw, draw_panel
from retrieve import Retriever

FRAME_SIZE = (1280, 720)
STABLE_FRAMES = 5  # 한두 프레임만 잡힌 오탐을 버리려고 5프레임 연속 탐지된 상태만 인정함
HOLD_SECONDS = 2.0  # 신호등이 화면 밖으로 나가도 조문을 읽을 수 있도록 2초 동안 유지함
# 개발용 세트 판정 재현율이 간격 0, 2, 4초에서 33, 34, 33개로 같아 가장 긴 4초를 고름(2026-11-03)
GEMMA_INTERVAL = 4.0
WINDOW = "Driving Rule Retrieval"
VIDEOS = [("원주", "videos/wonju-demo.mp4"), ("비벌리힐스", "videos/beverly-demo.mp4"), ("샌프란시스코", "videos/sf-demo.mp4")]
CORPORA = [("현행 도로교통법", "corpus-current.json"), ("2021년 도로교통법", "corpus-2021.json"),
           ("캘리포니아 차량법", "corpus-california.json")]
EMPTY_SCENE = {"facts": set(), "corpus": None, "sign": "", "article": None, "instruction": None}

parser = argparse.ArgumentParser()
parser.add_argument("video", nargs="?", default=VIDEOS[0][1])
parser.add_argument("--corpus", default=CORPORA[0][1])
parser.add_argument("--model", default="../src/models/YOLO/yolo11n.pt")
parser.add_argument("--interval", type=float, default=GEMMA_INTERVAL)
args = parser.parse_args()

llm = load_gemma()
# 맥락 스레드가 Gemma를 쓰는 동안 메인 루프가 Gemma를 부르지 않도록 세 코퍼스의 지시문을 시작할 때 모두 만듦
retrievers, articles, instructions = [], [], []
for corpus_name, corpus_path in CORPORA:
    retriever = Retriever(corpus_path)
    found, written = {}, {}
    for name in EDGE_CASES:
        found[name] = retriever.search(retriever.queries[name], 1)[0][1]
        written[name] = write_instruction(llm, name, found[name])
        print(corpus_name, name, written[name]["행동 종류"], found[name]["id"])
    retrievers.append(retriever)
    articles.append(found)
    instructions.append(written)

video_paths = [path for name, path in VIDEOS]
corpus_paths = [path for name, path in CORPORA]
selected = {"video": video_paths.index(args.video), "corpus": corpus_paths.index(args.corpus)}
buttons = button_boxes(FRAME_SIZE[0], {"video": [name for name, path in VIDEOS], "corpus": [name for name, path in CORPORA]})


def on_click(event, x, y, flags, param):
    """버튼을 누르면 고른 영상이나 법령 번호를 바꾸는 함수"""
    if event == cv2.EVENT_LBUTTONDOWN:
        for group, index, name, (x1, y1, x2, y2) in buttons:
            if x1 <= x <= x2 and y1 <= y <= y2:
                selected[group] = index


# 시작 지시문을 다 만든 뒤 스레드를 시작해 두 곳이 같은 Gemma를 동시에 쓰지 않게 함
shared = {"frame": None, "corpus": selected["corpus"], "scene": EMPTY_SCENE, "interval": args.interval}
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
    shared["corpus"] = selected["corpus"]
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
    # 패널 항목은 엣지 케이스 이름, 근거 조항, 지시문으로 이루어짐
    panel = []
    for name in active_edge_cases(held | scene["facts"]):
        panel.append((name, articles[corpus][name], instructions[corpus][name]))
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

    # 패널 글자는 엣지 케이스나 버튼 선택이 바뀔 때만 다시 그림
    key = (tuple(item[0] for item in panel), selected["video"], selected["corpus"])
    if key != panel_key:
        panel_key = key
        panel_image = draw_panel(FRAME_SIZE[1], FRAME_SIZE[0], panel, buttons, selected)
    output_frame = draw(frame, boxes, panel_image, displayed_fps)

    cv2.imshow(WINDOW, output_frame)

    # 처리가 영상보다 빠르면 남는 시간만큼 기다려 원래 속도로 재생함
    wait_ms = max(1, int(1000 / video_fps - (time.perf_counter() - start_time) * 1000))
    if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
