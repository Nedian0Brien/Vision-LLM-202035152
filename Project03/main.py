"""주행 영상에서 상황을 인식하고 법령 조문을 읽어 지시문을 보여 주는 실행 파일"""

import argparse
import json
import threading
import time

import cv2
from ultralytics import YOLO

from detect import CLASSES, DETECT_CONF, IMAGE_SIZE
from gemma import NO_LAW, STATUS, load_corpus, load_embedder, load_gemma, load_law, recognize, watch
from hud import CALLS, MIC_BOX, SCREEN_HEIGHT, VIDEO_WIDTH, button_boxes, draw, draw_loading, draw_nodes, draw_panel
from voice import listen, speak_loop

FRAME_SIZE = (1280, 720)
WINDOW = "Driving Rule Retrieval"
OUTPUT_PATH = "vla-output.jsonl"
CAMERA = "camera"
# 세 번째 값은 원래 속도로 볼 구간(초)이고 2026-11-06 시연 영상을 4초 간격으로 판단한 결과에서 지시가 이어지는 곳을 골랐음
VIDEOS = [("터널", "videos/tunnel-korea.mp4", [(0, 18)]), ("금문교", "videos/golden-gate.mp4", [(0, 30)]),
          ("보호구역", "videos/school-zone-busan.mp4", [(0, 26)]), ("비", "videos/rain-seoul.mp4", [(0, 36)]), ("통학버스", "videos/school-bus-korea.mp4", [(0, 26)]),
          ("소화전", "videos/hydrant-sf.mp4", [(0, 35)]), ("카메라", CAMERA, [])]
FAST_SPEED = 8
# YOLO 결과가 약 3프레임 늦게 나오므로 화면도 그만큼 늦춰 박스를 맞춤(2026-11-07 Mac 측정, 어긋남 평균 2.3프레임에서 0.7프레임)
DISPLAY_DELAY = 3
# 판단 기회를 늘리려고 영상을 느껴지지 않을 만큼만 늦춤(2026-11-07 사용자 지시)
PLAY_SPEED = 0.9
ANSWER_SECONDS = 15
SPEECH_REPEAT_SECONDS = 10
READING_SECONDS = 3
SCENE_SECONDS = 1
LAWS = [("현행 도로교통법", "current"), ("2006년 도로교통법", "2006"), ("캘리포니아 차량법", "california"), ("조문 없음", NO_LAW)]
# 처리가 늦으면 오래된 프레임을 버려 지연이 쌓이지 않음
pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 ! "
    "appsink drop=true max-buffers=1 sync=false"
)
# pipeline부터 차례로 열어 보며 장비마다 다른 카메라에 대응함, 모두 실패하면 현장 카메라에 맞게 pipeline을 고침
SINK = "videoconvert ! video/x-raw,format=BGR ! queue leaky=downstream max-size-buffers=1 ! appsink drop=true max-buffers=1 sync=false"
USB_DEVICES = [0, 1, 2, 3]
CAMERA_PIPELINES = ([pipeline]
                    + [f"v4l2src device=/dev/video{device} ! video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! {SINK}" for device in USB_DEVICES[1:]]
                    + [f"nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! {SINK}"]
                    + [f"v4l2src device=/dev/video{device} ! decodebin ! {SINK}" for device in USB_DEVICES])
CAMERA_LOG = "jetson-camera.txt"
# 단계 이름과 걸리는 시간(초)이고 시간은 단계 안에서 막대를 움직이는 데만 씀(2026-11-07 Jetson 측정)
LOAD_STEPS = [("탐지 모델 불러오기", 6), ("Gemma 불러오기", 40), ("임베딩 모델 불러오기", 3), ("조문 불러오기", 1), ("Gemma 예열", 24)]

parser = argparse.ArgumentParser()
parser.add_argument("video", nargs="?", default=VIDEOS[0][1])
parser.add_argument("--model", default="../src/models/YOLO/yolo11n.pt")
args = parser.parse_args()

selected = {"video": [path for name, path, focus in VIDEOS].index(args.video), "law": 0}
buttons = button_boxes(VIDEO_WIDTH, {"video": [name for name, path, focus in VIDEOS], "law": [name for name, short in LAWS]})


def start_listening():
    """듣는 중이 아니면 음성 질문 스레드를 시작하는 함수"""
    if not shared["listening"]:
        threading.Thread(target=listen, args=(shared,), daemon=True).start()


def terminal_questions():
    """터미널에 입력한 줄을 질문 칸에 넣는 함수"""
    # OpenCV 창은 한글 조합 입력을 받지 못해 터미널에서 받음
    # 터미널 없이 실행하면 입력이 바로 끝나므로 이 스레드만 멈춤
    try:
        while True:
            line = input().strip()
            if line:
                shared["question"] = line
    except EOFError:
        return


def on_click(event, x, y, flags, param):
    """버튼을 누르면 고른 영상이나 법령 번호를 바꾸거나 음성 질문을 시작하는 함수"""
    if event == cv2.EVENT_LBUTTONDOWN:
        if MIC_BOX[0] <= x <= MIC_BOX[2] and MIC_BOX[1] <= y <= MIC_BOX[3]:
            start_listening()
        for group, index, name, (x1, y1, x2, y2) in buttons:
            if x1 <= x <= x2 and y1 <= y <= y2:
                selected[group] = index


# 마우스 콜백은 이름 붙은 창에만 달 수 있어 첫 imshow 전에 창을 만듦
# 시연 장비의 모니터가 1600×900보다 작아도 창 크기를 줄여 전체를 볼 수 있게 함
cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WINDOW, on_click)

def open_camera():
    """프레임이 읽히는 첫 카메라 pipeline으로 장치를 여는 함수"""
    for candidate in CAMERA_PIPELINES:
        camera = cv2.VideoCapture(candidate, cv2.CAP_GSTREAMER)
        if camera.isOpened() and camera.read()[0]:
            # llama_cpp가 호출 중 stdout을 막아 print가 사라지므로 파일에 씀
            with open(CAMERA_LOG, "w", encoding="utf-8") as file:
                file.write(candidate + "\n")
            return camera
        camera.release()
    with open(CAMERA_LOG, "w", encoding="utf-8") as file:
        file.write("열린 카메라 없음\n")
    return camera


def next_step():
    """로딩 화면의 단계를 하나 올리는 함수"""
    loading["step"] += 1
    loading["since"] = time.perf_counter()


def load_models():
    """모델과 조문을 차례로 불러오고 처음 한 번 오래 걸리는 Gemma 호출을 미리 돌리는 함수"""
    loading["model"] = YOLO(args.model)
    next_step()
    loading["llm"] = load_gemma()
    next_step()
    loading["embedder"] = load_embedder()
    next_step()
    loading["laws"] = {short: load_law(short) for name, short in LAWS}
    loading["corpora"] = {short: load_corpus(short) for name, short in LAWS}
    next_step()
    # 첫 인식이 약 24초 걸리고 다음부터 5초 안팎이라 영상 시작 전에 끝냄(2026-11-07 Jetson 측정)
    capture = cv2.VideoCapture(VIDEOS[0][1])
    frame = cv2.resize(capture.read()[1], FRAME_SIZE)
    capture.release()
    result = loading["model"].predict(source=frame, conf=DETECT_CONF, iou=0.5, verbose=False, classes=CLASSES, imgsz=IMAGE_SIZE)[0]
    recognize(loading["llm"], frame, result, False)
    STATUS["answers"] = {}
    STATUS["images"] = {}
    STATUS["stage"] = "wait"
    next_step()


def detect_boxes():
    """가장 새로운 프레임에서 박스를 찾아 화면과 Gemma에 넘기는 함수"""
    while True:
        if shared["frame"] is None:
            time.sleep(0.001)
            continue
        video, image = shared["frame"]
        shared["frame"] = None
        found = model.predict(source=image, conf=DETECT_CONF, iou=0.5, verbose=False, classes=CLASSES, imgsz=IMAGE_SIZE)[0]
        # 탐지 중에 영상이 바뀌면 이전 영상의 박스를 새 영상에 그리지 않음
        if video == current_video:
            shared["result"] = found
            shared["input"] = (image, found)


# 영상 루프가 YOLO와 Gemma를 기다리지 않도록 둘 다 별도 스레드에서 돌림(2026-11-07 Mac 측정, 12~14 FPS에서 22~25 FPS)
shared = {"question": None, "thinking": None, "answer": None, "answer_time": 0.0, "speech": None, "listening": False, "speaking": False, "voice_error": None,
          "frame": None, "result": None, "input": None, "law": LAWS[selected["law"]][1], "clip": {"last_seen": {}, "rules": [], "nodes": []}}
loading = {"step": 0, "since": time.perf_counter()}
loader = threading.Thread(target=load_models, daemon=True)
loader.start()
while loader.is_alive():
    cv2.imshow(WINDOW, draw_loading(SCREEN_HEIGHT, LOAD_STEPS, loading["step"], time.perf_counter() - loading["since"]))
    if cv2.waitKey(100) & 0xFF == ord("q"):
        raise SystemExit
# 불러오기가 실패하면 스레드가 끝나도 마지막 단계에 닿지 못함
if loading["step"] < len(LOAD_STEPS):
    raise SystemExit("모델을 불러오지 못했습니다")
model = loading["model"]
threading.Thread(target=watch, args=(loading["llm"], loading["laws"], shared, loading["embedder"], loading["corpora"]), daemon=True).start()
threading.Thread(target=speak_loop, args=(shared,), daemon=True).start()
threading.Thread(target=terminal_questions, daemon=True).start()
print("터미널에 질문을 입력하거나 창에서 m 키로 음성 질문을 시작합니다")

cap = cv2.VideoCapture()
current_video = None
threading.Thread(target=detect_boxes, daemon=True).start()
waiting = []
spoken_time = {}
shown_time = time.perf_counter()
next_time = shown_time
drawn_view = None
drawn_state = None
drawn_nodes = None
shown_rules = []
reading = None
reading_since = 0.0
scene = "choice"
scene_since = 0.0
displayed_fps = 0.0

while True:
    if selected["video"] != current_video:
        current_video = selected["video"]
        is_camera = VIDEOS[current_video][1] == CAMERA
        # 카메라는 앞서 연 장치를 닫아야 다시 열 수 있음
        cap.release()
        if is_camera:
            cap = open_camera()
            frame_ms = 1
        else:
            cap = cv2.VideoCapture(VIDEOS[current_video][1])
            frame_ms = 1000 / (cap.get(cv2.CAP_PROP_FPS) * PLAY_SPEED)
        # 묶음을 새로 만들면 돌고 있던 판단의 결과는 버려진 묶음에 쓰여 화면에 나오지 않음
        # 입력을 먼저 비워야 watch가 새 묶음과 이전 영상 프레임을 함께 읽지 않음
        shared["frame"] = None
        shared["result"] = None
        shared["input"] = None
        shared["clip"] = {"last_seen": {}, "rules": [], "nodes": []}
        waiting.clear()

    ret, frame = cap.read()

    # 영상이 끝나면 카메라 탭을 건너뛰고 다음 영상을 열고, 카메라가 없거나 끊기면 화면이 멈추지 않도록 첫 영상으로 돌아감
    if not ret:
        selected["video"] = 0 if is_camera else (selected["video"] + 1) % (len(VIDEOS) - 1)
        current_video = None
        continue

    # 카메라를 영상과 같은 16:9로 잘라 탐지와 크롭 위치가 찌그러지지 않게 함
    if is_camera:
        top = max((frame.shape[0] - frame.shape[1] * 9 // 16) // 2, 0)
        frame = frame[top:frame.shape[0] - top]
    frame = cv2.resize(frame, FRAME_SIZE)
    position = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    fast = not is_camera and not [start for start, end in VIDEOS[current_video][2] if start <= position < end]
    if fast:
        # 배속 구간에서는 프레임을 건너뛰고 입력을 비워 YOLO와 Gemma를 돌리지 않음
        for skipped in range(FAST_SPEED - 1):
            cap.grab()
        result = None
        shared["frame"] = None
        if shared["input"] is not None:
            shared["input"] = None
            shared["clip"] = {"last_seen": {}, "rules": [], "nodes": []}
            STATUS["stage"] = "skip"
    else:
        # draw가 프레임에 박스를 그리므로 YOLO와 Gemma에는 복사본을 넘김
        shared["frame"] = (current_video, frame.copy())
        waiting.append(frame)
        if len(waiting) <= DISPLAY_DELAY:
            continue
        frame = waiting.pop(0)
        result = shared["result"]
    shared["law"] = LAWS[selected["law"]][1]

    # 프레임마다 PIL로 글자를 그리면 느려서 화면 내용이 바뀔 때만 패널을 다시 그림
    rules = shared["clip"]["rules"]
    view = (selected["video"], selected["law"], rules)
    now = time.perf_counter()
    if shared["answer"] and now - shared["answer_time"] >= ANSWER_SECONDS:
        shared["answer"] = None
    # 새 규칙이 나오면 조문, 읽기, 출력을 1초씩 보여 주고 읽어 줌
    for rule in rules:
        if rule not in shown_rules:
            reading = rule
            reading_since = now
            # 유지 시간이 끊겼다 다시 나온 같은 지시는 다시 읽지 않고, 답변을 읽는 동안에도 끼어들지 않음
            if now - spoken_time.get(rule["instruction"], -SPEECH_REPEAT_SECONDS) >= SPEECH_REPEAT_SECONDS and not shared["answer"]:
                shared["speech"] = rule["instruction"]
                spoken_time[rule["instruction"]] = now
    shown_rules = rules
    if now - reading_since >= READING_SECONDS:
        reading = None
    phase = int(now - reading_since) if reading else 0
    # 0.2초짜리 호출에 화면이 깜빡이지 않도록 장면을 최소 시간만큼 유지함
    if fast:
        scene = "skip"
        reading = None
    elif STATUS["stage"] in CALLS and STATUS["stage"] != scene and now - scene_since >= SCENE_SECONDS:
        scene = STATUS["stage"]
        scene_since = now
    answer = shared["answer"]
    voice = {"listening": shared["listening"], "thinking": shared["thinking"], "answer": answer, "error": shared["voice_error"]}
    state = (scene, reading is None, phase, STATUS["answers"].get(scene), voice["listening"], voice["thinking"], id(answer), voice["error"])
    nodes = shared["clip"]["nodes"]
    if answer:
        nodes = [{"situation": article["id"], "article": article, "fresh": True, "instruction": "질문의 근거 후보"} for article in answer["articles"]]
    # 생성 중인 노드가 있을 때만 스피너 번호가 바뀌어 열을 다시 그림
    spinner = int(now * 8) % 8 if [node for node in nodes if node["instruction"] is None] else 0
    node_state = ([(node["situation"], node["fresh"], node["instruction"]) for node in nodes], spinner)
    if node_state != drawn_nodes:
        drawn_nodes = node_state
        nodes_image = draw_nodes(nodes, spinner)
    if view != drawn_view or state != drawn_state:
        drawn_state = state
        panel_image = draw_panel(SCREEN_HEIGHT, VIDEO_WIDTH, buttons, selected, rules, STATUS, scene, reading, phase, voice)
    if view != drawn_view:
        drawn_view = view
        # llama_cpp가 호출 중 stdout을 막아 print가 사라지므로 파일에 씀
        with open(OUTPUT_PATH, "a", encoding="utf-8") as file:
            file.write(json.dumps({"time": time.time(), "rules": rules}, ensure_ascii=False) + "\n")
        STATUS["outputs"] += 1

    elapsed_time = time.perf_counter() - shown_time
    shown_time = time.perf_counter()

    # 배속 구간은 모델을 돌리지 않아 처리 속도를 나타내지 않음
    if displayed_fps == 0:
        displayed_fps = 1.0 / elapsed_time
    elif not fast:
        # 간격을 평균 내야 재생 시각에 맞추는 짧은 간격이 FPS를 부풀리지 않음
        displayed_fps = 1.0 / (0.9 / displayed_fps + 0.1 * elapsed_time)

    output_frame = draw(frame, result, panel_image, nodes_image, displayed_fps)

    cv2.imshow(WINDOW, output_frame)

    # waitKey가 요청보다 오래 걸려도 오차가 쌓이지 않도록 재생 시각을 기준으로 기다림(2026-11-07 Mac 측정, 25 FPS에서 29 FPS)
    next_time = max(next_time + frame_ms / 1000, time.perf_counter())
    wait_ms = max(1, int((next_time - time.perf_counter()) * 1000))
    key = cv2.waitKey(wait_ms) & 0xFF
    if key == ord("q"):
        break
    if key == ord("m"):
        start_listening()

cap.release()
cv2.destroyAllWindows()
