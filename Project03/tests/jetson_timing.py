"""main.py를 고치지 않고 실행하면서 호출별 시간, 실제 FPS, 프레임에서 패널까지 지연을 재는 스크립트"""

import os
import runpy
import statistics
import sys
import time
import traceback

import cv2
from ultralytics import YOLO

sys.path.insert(0, os.getcwd())

import gemma
import hud

# llama_cpp가 호출 중 stdout을 막아 print가 사라지므로 파일에 씀
RESULT_PATH = "jetson-timing.txt"
times = {"판단 전체": [], "객관식": [], "조문 읽기": [], "YOLO": [], "프레임에서 패널까지": []}
state = {"frame_time": None, "judged_frame": None, "asked": 0.0, "shown": 0, "first_shown": None, "video": False}

predict, recognize, ask_word, read_provision, draw_panel, draw, imshow = YOLO.predict, gemma.recognize, gemma.ask_word, gemma.read_provision, hud.draw_panel, hud.draw, cv2.imshow


def timed_predict(self, *args, **kwargs):
    """YOLO 시간을 재고 프레임이 입력 칸에 들어가는 시각을 남기는 함수"""
    start = time.perf_counter()
    results = predict(self, *args, **kwargs)
    state["frame_time"] = time.perf_counter()
    times["YOLO"].append(state["frame_time"] - start)
    return results


def timed_ask_word(llm, stage, image, question, words):
    """크롭 호출 시간을 질문별로 재는 함수"""
    start = time.perf_counter()
    answer = ask_word(llm, stage, image, question, words)
    elapsed = time.perf_counter() - start
    state["asked"] += elapsed
    times.setdefault(stage + " 호출", []).append(elapsed)
    return answer


def timed_recognize(llm, frame, result, in_school_zone):
    """판단 전체와 객관식 시간을 재고 판단한 프레임의 시각을 남기는 함수"""
    frame_time = state["frame_time"]
    state["asked"] = 0.0
    start = time.perf_counter()
    found = recognize(llm, frame, result, in_school_zone)
    elapsed = time.perf_counter() - start
    times["판단 전체"].append(elapsed)
    times["객관식"].append(elapsed - state["asked"])
    state["judged_frame"] = frame_time
    return found


def timed_read_provision(llm, situation, article):
    """조문 읽기 시간을 재는 함수"""
    start = time.perf_counter()
    rule = read_provision(llm, situation, article)
    times["조문 읽기"].append(time.perf_counter() - start)
    return rule


def timed_draw_panel(*args):
    """규칙이 패널에 새로 나올 때 그 프레임부터 걸린 시간을 재는 함수"""
    rules = args[4]
    # 규칙이 비는 경우는 영상 전환이거나 유지가 끝난 경우라 판단한 프레임을 쓰지 않음
    if rules and state["judged_frame"] is not None:
        times["프레임에서 패널까지"].append(time.perf_counter() - state["judged_frame"])
        state["judged_frame"] = None
    return draw_panel(*args)


def timed_draw(*args):
    """영상 화면을 처음 그릴 때 로딩 중에 잰 값을 비우는 함수"""
    if not state["video"]:
        state["video"] = True
        for values in times.values():
            values.clear()
    return draw(*args)


def counted_imshow(window, image):
    """화면에 실제로 나온 프레임 수를 세는 함수"""
    if not state["video"]:
        imshow(window, image)
        return
    if state["first_shown"] is None:
        state["first_shown"] = time.perf_counter()
    state["shown"] += 1
    imshow(window, image)


YOLO.predict = timed_predict
gemma.ask_word = timed_ask_word
gemma.recognize = timed_recognize
gemma.read_provision = timed_read_provision
hud.draw_panel = timed_draw_panel
hud.draw = timed_draw
cv2.imshow = counted_imshow

arguments = sys.argv[1:]
sys.argv = ["main.py"] + arguments
error = ""
# 측정 기회가 한 번이라 main.py가 중간에 멈춰도 그때까지의 값과 오류를 남김
try:
    runpy.run_path("main.py", run_name="__main__")
except Exception:
    error = traceback.format_exc()
with open(RESULT_PATH, "a", encoding="utf-8") as file:
    file.write(f"실행 인자: {arguments}\n")
    for name, values in times.items():
        if values:
            file.write(f"{name}: 횟수 {len(values)}, 중앙값 {statistics.median(values):.3f}초, 최대 {max(values):.3f}초\n")
    if state["first_shown"] is not None:
        seconds = time.perf_counter() - state["first_shown"]
        file.write(f"실제 FPS: {state['shown'] / seconds:.1f} ({state['shown']}프레임, {seconds:.0f}초)\n")
    file.write(error)
# 프로그램이 끝날 때 llama_cpp 정리 과정에서 멈춘 적이 있어 바로 끝냄
os._exit(0)
