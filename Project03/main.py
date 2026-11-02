"""주행 영상의 장면 상태로 도로교통법 조문을 찾아 보여 주는 실행 파일"""

import argparse
import threading
import time

import cv2
from ultralytics import YOLO

from detect import CLASSES, scene_state
from gemma import load_gemma, watch, write_instruction
from hud import draw
from retrieve import Retriever

FRAME_SIZE = (1280, 720)
STABLE_FRAMES = 5  # 한두 프레임만 잡힌 오탐을 버리려고 5프레임 연속 탐지된 상태만 인정함
HOLD_SECONDS = 2.0  # 신호등이 화면 밖으로 나가도 조문을 읽을 수 있도록 2초 동안 유지함

parser = argparse.ArgumentParser()
parser.add_argument("video")
parser.add_argument("--corpus", default="corpus-current.json")
parser.add_argument("--model", default="../src/models/YOLO/yolo11n.pt")
args = parser.parse_args()

llm = load_gemma()
retriever = Retriever(args.corpus)
# 상태마다 검색 문장이 하나로 정해져 있어 검색 결과와 지시문을 시작할 때 한 번만 만듦
articles = {}
instructions = {}
for state, query in retriever.queries.items():
    articles[state] = retriever.search(query, 1)[0][1]
    instructions[state] = write_instruction(llm, state, articles[state])
    print(state, instructions[state])

# 시작 지시문을 다 만든 뒤 스레드를 시작해 두 곳이 같은 Gemma를 동시에 쓰지 않게 함
shared = {"frame": None, "scene": {"facts": set(), "sign": "", "article": None, "instruction": ""}}
threading.Thread(target=watch, args=(llm, retriever, shared), daemon=True).start()
model = YOLO(args.model)
cap = cv2.VideoCapture(args.video)
video_fps = cap.get(cv2.CAP_PROP_FPS)
hold_frames = int(HOLD_SECONDS * video_fps)

streak = {state: 0 for state in retriever.queries}
last_seen = {state: -hold_frames - 1 for state in retriever.queries}
frame_index = 0
displayed_fps = 0.0

while True:
    start_time = time.perf_counter()

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, FRAME_SIZE)
    # draw가 이 프레임 위에 박스를 그리므로 Gemma에는 복사본을 넘김
    shared["frame"] = frame.copy()
    results = model.predict(source=frame, conf=0.4, iou=0.5, verbose=False, classes=CLASSES)
    found, boxes = scene_state(results[0], frame)

    for state in retriever.queries:
        if state in found:
            streak[state] += 1
        else:
            streak[state] = 0
        if streak[state] >= STABLE_FRAMES:
            last_seen[state] = frame_index
    # 패널 항목은 이름, 근거 조항, 지시문, Gemma 비전에서 왔는지로 이루어짐
    scene = shared["scene"]
    panel = []
    for state in retriever.queries:
        if frame_index - last_seen[state] <= hold_frames:
            panel.append((state, articles[state], instructions[state], False))
        elif state in scene["facts"]:
            panel.append((state, articles[state], instructions[state], True))
    if scene["article"] is not None:
        panel.append(("표지판 " + scene["sign"], scene["article"], scene["instruction"], True))
    frame_index += 1

    elapsed_time = time.perf_counter() - start_time
    current_fps = 1.0 / elapsed_time

    if displayed_fps == 0:
        displayed_fps = current_fps
    else:
        displayed_fps = 0.9 * displayed_fps + 0.1 * current_fps

    output_frame = draw(frame, boxes, panel, retriever.name, displayed_fps)

    cv2.imshow("Driving Rule Retrieval", output_frame)

    # 처리가 영상보다 빠르면 남는 시간만큼 기다려 원래 속도로 재생함
    wait_ms = max(1, int(1000 / video_fps - (time.perf_counter() - start_time) * 1000))
    if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
