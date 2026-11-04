"""주행 영상에 YOLO 탐지 결과를 그려 보여 주는 실행 파일"""

import argparse
import time

import cv2
from ultralytics import YOLO

from detect import CLASSES, IMAGE_SIZE, detections
from hud import button_boxes, draw, draw_panel

FRAME_SIZE = (1280, 720)
WINDOW = "Driving Rule Retrieval"
VIDEOS = [("원주", "videos/wonju-demo.mp4"), ("비벌리힐스", "videos/beverly-demo.mp4"), ("샌프란시스코", "videos/sf-demo.mp4")]

parser = argparse.ArgumentParser()
parser.add_argument("video", nargs="?", default=VIDEOS[0][1])
parser.add_argument("--model", default="../src/models/YOLO/yolo11n.pt")
args = parser.parse_args()

video_paths = [path for name, path in VIDEOS]
selected = {"video": video_paths.index(args.video)}
buttons = button_boxes(FRAME_SIZE[0], {"video": [name for name, path in VIDEOS]})


def on_click(event, x, y, flags, param):
    """버튼을 누르면 고른 영상 번호를 바꾸는 함수"""
    if event == cv2.EVENT_LBUTTONDOWN:
        for group, index, name, (x1, y1, x2, y2) in buttons:
            if x1 <= x <= x2 and y1 <= y <= y2:
                selected[group] = index


model = YOLO(args.model)
# 마우스 콜백은 이름 붙은 창에만 달 수 있어 첫 imshow 전에 창을 만듦
cv2.namedWindow(WINDOW)
cv2.setMouseCallback(WINDOW, on_click)

current_video = None
displayed_fps = 0.0

while True:
    start_time = time.perf_counter()

    if selected["video"] != current_video:
        current_video = selected["video"]
        cap = cv2.VideoCapture(VIDEOS[current_video][1])
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        panel_image = draw_panel(FRAME_SIZE[1], FRAME_SIZE[0], buttons, selected)

    ret, frame = cap.read()

    # 영상이 끝나면 같은 영상을 처음부터 다시 엶
    if not ret:
        current_video = None
        continue

    frame = cv2.resize(frame, FRAME_SIZE)
    results = model.predict(source=frame, conf=0.4, iou=0.5, verbose=False, classes=CLASSES, imgsz=IMAGE_SIZE)
    boxes = detections(results[0])

    elapsed_time = time.perf_counter() - start_time
    current_fps = 1.0 / elapsed_time

    if displayed_fps == 0:
        displayed_fps = current_fps
    else:
        displayed_fps = 0.9 * displayed_fps + 0.1 * current_fps

    output_frame = draw(frame, boxes, panel_image, displayed_fps)

    cv2.imshow(WINDOW, output_frame)

    # 처리가 영상보다 빠르면 남는 시간만큼 기다려 원래 속도로 재생함
    wait_ms = max(1, int(1000 / video_fps - (time.perf_counter() - start_time) * 1000))
    if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
