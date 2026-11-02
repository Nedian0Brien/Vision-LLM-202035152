"""주행 영상의 장면 상태로 도로교통법 조문을 찾아 보여 주는 실행 파일"""

import argparse
import time

import cv2
from ultralytics import YOLO

from detect import CLASSES, scene_state
from hud import draw
from instruct import write_instructions
from retrieve import Retriever

FRAME_SIZE = (1280, 720)
STABLE_FRAMES = 5  # 한두 프레임만 잡힌 오탐을 버리려고 5프레임 연속 탐지된 상태만 인정함
HOLD_SECONDS = 2.0  # 신호등이 화면 밖으로 나가도 조문을 읽을 수 있도록 2초 동안 유지함

parser = argparse.ArgumentParser()
parser.add_argument("video")
parser.add_argument("--corpus", default="corpus-current.json")
parser.add_argument("--model", default="../src/models/YOLO/yolo11n.pt")
args = parser.parse_args()

retriever = Retriever(args.corpus)
# 상태마다 검색 문장이 하나로 정해져 있어 검색 결과와 지시문을 시작할 때 한 번만 만듦
articles = {state: retriever.search(query, 1)[0] for state, query in retriever.queries.items()}
# 지시문을 만든 뒤 Gemma가 해제되고 나서 YOLO를 불러와 두 모델이 메모리에 함께 올라가지 않음
instructions = write_instructions(articles)
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
    results = model.predict(source=frame, conf=0.4, iou=0.5, verbose=False, classes=CLASSES)
    found, boxes = scene_state(results[0], frame)

    for state in retriever.queries:
        if state in found:
            streak[state] += 1
        else:
            streak[state] = 0
        if streak[state] >= STABLE_FRAMES:
            last_seen[state] = frame_index
    states = [state for state in retriever.queries if frame_index - last_seen[state] <= hold_frames]
    frame_index += 1

    elapsed_time = time.perf_counter() - start_time
    current_fps = 1.0 / elapsed_time

    if displayed_fps == 0:
        displayed_fps = current_fps
    else:
        displayed_fps = 0.9 * displayed_fps + 0.1 * current_fps

    output_frame = draw(frame, boxes, states, articles, instructions, retriever.name, displayed_fps)

    cv2.imshow("Driving Rule Retrieval", output_frame)

    # 처리가 영상보다 빠르면 남는 시간만큼 기다려 원래 속도로 재생함
    wait_ms = max(1, int(1000 / video_fps - (time.perf_counter() - start_time) * 1000))
    if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
