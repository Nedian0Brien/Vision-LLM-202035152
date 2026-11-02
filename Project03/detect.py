"""YOLO 탐지 결과를 장면 상태로 바꾸는 모듈"""

import cv2

# COCO 클래스 번호: 0 사람, 1 자전거, 2 승용차, 7 트럭, 9 신호등, 11 정지 표지
CLASSES = [0, 1, 2, 7, 9, 11]
# 개발용 세트에서 신호등과 자전거 엣지 케이스를 640 입력은 10개 중 5개, 1280 입력은 10개 모두 맞힘(2026-11-03)
IMAGE_SIZE = 1280
STATE_OF_CLASS = {0: "보행자", 1: "자전거", 11: "정지 표지"}

MIN_AREA = 0.002  # 화면의 0.2%보다 작은 사람·자전거·표지는 멀어서 판단에 쓰지 않음
NEAR_AREA = 0.03  # 화면의 3% 이상인 차가 가운데 있으면 바로 앞차로 봄
HOOD_Y = 0.8  # 블랙박스 화면 아래 20%는 보닛과 유리창 스티커라 오탐이 생김
MIN_LIGHT_PIXELS = 20  # 켜진 등의 색 픽셀이 이보다 적으면 꺼진 신호등으로 봄

# 켜진 등은 채도와 명도가 높아 S, V 하한을 100, 150으로 둠
# 빨강은 H가 0 부근과 179 부근에 걸쳐 있어 두 범위를 합침
LIGHT_RANGES = {
    "적색": [((0, 100, 150), (10, 255, 255)), ((170, 100, 150), (179, 255, 255))],
    "황색": [((15, 100, 150), (35, 255, 255))],
    "녹색": [((40, 100, 150), (100, 255, 255))],
}


def light_color(crop):
    """신호등 박스 안에서 픽셀이 가장 많은 등 색을 고르는 함수"""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    best_color, best_pixels = None, MIN_LIGHT_PIXELS
    for color, ranges in LIGHT_RANGES.items():
        pixels = 0
        for lower, upper in ranges:
            pixels += cv2.countNonZero(cv2.inRange(hsv, lower, upper))
        if pixels > best_pixels:
            best_color, best_pixels = color, pixels
    return best_color


def scene_state(result, frame):
    """탐지 결과에서 장면 상태 이름 목록과 그릴 박스 목록을 만드는 함수"""
    height, width = frame.shape[:2]
    found = set()
    boxes = []
    for box, cls in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist()):
        x1, y1, x2, y2 = map(int, box)
        cls = int(cls)
        # 박스 중심으로 거르면 발이 화면 아래에 걸친 가까운 보행자까지 지워졌음
        if y1 > height * HOOD_Y:
            continue
        area = (x2 - x1) * (y2 - y1) / (width * height)
        center_x = (x1 + x2) / 2 / width
        state = None
        if cls == 9:
            color = light_color(frame[y1:y2, x1:x2])
            if color is not None:
                state = color + " 신호"
        elif cls in (2, 7):
            if area >= NEAR_AREA and 1 / 3 < center_x < 2 / 3:
                state = "앞차"
        elif area >= MIN_AREA:
            state = STATE_OF_CLASS[cls]
        if state is not None:
            found.add(state)
        boxes.append((x1, y1, x2, y2, state))
    return found, boxes
