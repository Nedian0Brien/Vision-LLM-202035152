"""주행 영상에서 상황을 인식하고 법령 조문을 읽어 지시문을 보여 주는 실행 파일"""

import argparse
import json
import threading
import time

import cv2
from ultralytics import YOLO

from detect import CLASSES, DETECT_CONF, IMAGE_SIZE
from gemma import load_gemma, load_law, watch
from hud import button_boxes, draw, draw_panel

FRAME_SIZE = (1280, 720)
WINDOW = "Driving Rule Retrieval"
OUTPUT_PATH = "vla-output.jsonl"
CAMERA = "camera"
VIDEOS = [("터널", "videos/tunnel-korea.mp4"), ("금문교", "videos/golden-gate.mp4"), ("샌프란시스코", "videos/sf-demo.mp4"), ("보호구역 낱장", "videos/school-zone-stills.mp4"), ("카메라", CAMERA)]
LAWS = [("현행 도로교통법", "current"), ("캘리포니아 차량법", "california")]
# 처리가 늦으면 오래된 프레임을 버려 지연이 쌓이지 않음
pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

parser = argparse.ArgumentParser()
parser.add_argument("video", nargs="?", default=VIDEOS[0][1])
parser.add_argument("--model", default="../src/models/YOLO/yolo11n.pt")
args = parser.parse_args()

selected = {"video": [path for name, path in VIDEOS].index(args.video), "law": 0}
buttons = button_boxes(FRAME_SIZE[0], {"video": [name for name, path in VIDEOS], "law": [name for name, short in LAWS]})


def on_click(event, x, y, flags, param):
    """버튼을 누르면 고른 영상이나 법령 번호를 바꾸는 함수"""
    if event == cv2.EVENT_LBUTTONDOWN:
        for group, index, name, (x1, y1, x2, y2) in buttons:
            if x1 <= x <= x2 and y1 <= y <= y2:
                selected[group] = index


model = YOLO(args.model)
# 마우스 콜백은 이름 붙은 창에만 달 수 있어 첫 imshow 전에 창을 만듦
cv2.namedWindow(WINDOW)
cv2.setMouseCallback(WINDOW, on_click)

# 영상 루프가 Gemma를 기다리지 않도록 Gemma는 별도 스레드에서 돌림
shared = {"input": None, "law": LAWS[selected["law"]][1], "clip": {"last_seen": {}, "rules": []}}
laws = {short: load_law(short) for name, short in LAWS}
threading.Thread(target=watch, args=(load_gemma(), laws, shared), daemon=True).start()

cap = cv2.VideoCapture()
current_video = None
drawn_view = None
displayed_fps = 0.0

while True:
    start_time = time.perf_counter()

    if selected["video"] != current_video:
        current_video = selected["video"]
        is_camera = VIDEOS[current_video][1] == CAMERA
        # 카메라는 앞서 연 장치를 닫아야 다시 열 수 있음
        cap.release()
        if is_camera:
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            frame_ms = 1
        else:
            cap = cv2.VideoCapture(VIDEOS[current_video][1])
            frame_ms = 1000 / cap.get(cv2.CAP_PROP_FPS)
        # 묶음을 새로 만들면 돌고 있던 판단의 결과는 버려진 묶음에 쓰여 화면에 나오지 않음
        # 입력을 먼저 비워야 watch가 새 묶음과 이전 영상 프레임을 함께 읽지 않음
        shared["input"] = None
        shared["clip"] = {"last_seen": {}, "rules": []}

    ret, frame = cap.read()

    # 영상이 끝나면 같은 영상을 처음부터 다시 열고, 카메라가 없거나 끊기면 화면이 멈추지 않도록 첫 영상으로 돌아감
    if not ret:
        if is_camera:
            selected["video"] = 0
        current_video = None
        continue

    # 640×480 카메라를 영상과 같은 16:9로 잘라 탐지와 크롭 위치가 찌그러지지 않게 함
    if is_camera:
        frame = frame[60:420]
    frame = cv2.resize(frame, FRAME_SIZE)
    results = model.predict(source=frame, conf=DETECT_CONF, iou=0.5, verbose=False, classes=CLASSES, imgsz=IMAGE_SIZE)
    # draw가 프레임에 박스를 그리므로 Gemma에는 복사본을 넘김
    shared["input"] = (frame.copy(), results[0])
    shared["law"] = LAWS[selected["law"]][1]

    # 프레임마다 PIL로 글자를 그리면 느려서 화면 내용이 바뀔 때만 패널을 다시 그림
    rules = shared["clip"]["rules"]
    view = (selected["video"], selected["law"], rules)
    if view != drawn_view:
        drawn_view = view
        panel_image = draw_panel(FRAME_SIZE[1], FRAME_SIZE[0], buttons, selected, rules)
        # llama_cpp가 호출 중 stdout을 막아 print가 사라지므로 파일에 씀
        with open(OUTPUT_PATH, "a", encoding="utf-8") as file:
            file.write(json.dumps({"time": time.time(), "rules": rules}, ensure_ascii=False) + "\n")

    elapsed_time = time.perf_counter() - start_time
    current_fps = 1.0 / elapsed_time

    if displayed_fps == 0:
        displayed_fps = current_fps
    else:
        displayed_fps = 0.9 * displayed_fps + 0.1 * current_fps

    output_frame = draw(frame, results[0], panel_image, displayed_fps)

    cv2.imshow(WINDOW, output_frame)

    # 처리가 영상보다 빠르면 남는 시간만큼 기다려 원래 속도로 재생함
    wait_ms = max(1, int(frame_ms - (time.perf_counter() - start_time) * 1000))
    if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
