"""YOLO 탐지 결과를 Gemma 입력 크롭으로 바꾸는 모듈"""

import cv2
import numpy as np

# COCO 클래스 번호: 0 사람, 1 자전거, 2 승용차, 5 버스, 7 트럭, 9 신호등, 10 소화전, 11 정지 표지
CLASS_NAMES = {
    0: "사람",
    1: "자전거",
    2: "승용차",
    5: "버스",
    7: "트럭",
    9: "신호등",
    10: "소화전",
    11: "정지 표지",
}
# predict의 classes 인자로 넘겨 YOLO가 이 클래스만 찾게 함
CLASSES = list(CLASS_NAMES)

# 장면 종류마다 크롭을 만들 때 쓰는 클래스 묶음
PERSON_CLASSES = [0]
LIGHT_CLASSES = [9]
SIGN_CLASSES = [9, 11]
BUS_CLASSES = [5, 7]
HYDRANT_CLASSES = [10]

# 개발용 세트의 신호등과 정지 표지 장면 13개에서 640 입력은 7개, 1280 입력은 12개를 찾음(2026-11-03)
# 1280×720 프레임의 입력은 736×1280이고, 고정 크기 TensorRT 엔진에 정수 1280을 넘기면 1280×1280 입력이 되어 멈춤
IMAGE_SIZE = [736, 1280]

# 차가 다음 탐지까지 움직여도 같은 물체로 볼 만큼 겹치는 비율
MATCH_OVERLAP = 0.2

# 크롭에는 먼 신호등도 넣으려고 낮은 신뢰도까지 쓰고, 화면에는 기존 기준의 박스만 그림
DETECT_CONF = 0.25
SHOW_CONF = 0.4

# 이어 붙인 크롭의 높이이고 크롭마다 이 높이로 맞춤
STRIP_HEIGHT = 400


def class_strip(frame, result, classes):
    """지정한 클래스의 박스 주변을 잘라 가로로 이은 이미지를 만드는 함수"""
    height, width = frame.shape[:2]
    areas = []
    for (x1, y1, x2, y2), cls in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist()):
        if int(cls) in classes:
            areas.append(((x2 - x1) * (y2 - y1), x1, y1, x2, y2))

    # 박스가 없으면 Gemma에 물을 대상이 없음을 None으로 알림
    if not areas:
        return None

    # 넓이가 큰 박스가 가까운 물체이므로 큰 순서로 세 개만 씀
    areas.sort(reverse=True)
    crops = []
    for area, x1, y1, x2, y2 in areas[:3]:
        # 박스 크기만큼 사방을 넓혀 신호등 기둥이나 차로 같은 주변 모습이 함께 보이게 함
        size = max(x2 - x1, y2 - y1)
        crop = frame[
            max(0, int(y1 - size)) : min(height, int(y2 + size)),
            max(0, int(x1 - 1.5 * size)) : min(width, int(x2 + 1.5 * size)),
        ]

        # 적색 신호 판정이 전체 프레임에서는 0~2/9, 크롭에서는 9/9였음(2026-11-05 개발용 측정)
        crop_width = int(crop.shape[1] * STRIP_HEIGHT / crop.shape[0])
        crops.append(cv2.resize(crop, (crop_width, STRIP_HEIGHT), interpolation=cv2.INTER_CUBIC))

        # 크롭 사이에 폭 16픽셀의 검은 띠를 넣어 서로 다른 물체임을 구분함
        crops.append(np.zeros((STRIP_HEIGHT, 16, 3), np.uint8))

    return np.hstack(crops)


def road_crop(frame):
    """노면 글자가 보이는 차로 앞쪽 영역을 자르는 함수"""
    height, width = frame.shape[:2]

    # 노면 글자는 차 앞 가운데에 있어 세로는 위에서 55~92%, 가로는 15~85% 구간을 자름
    return frame[int(0.55 * height) : int(0.92 * height), int(0.15 * width) : int(0.85 * width)]


def box_overlap(first, second):
    """두 박스의 겹침 비율을 구하는 함수"""
    width = min(first[2], second[2]) - max(first[0], second[0])
    height = min(first[3], second[3]) - max(first[1], second[1])
    if width <= 0 or height <= 0:
        return 0.0

    # 합집합 넓이는 두 박스 넓이의 합에서 겹친 넓이를 한 번 뺀 값이고, 겹침 비율은 교집합을 합집합으로 나눈 값임
    both = width * height
    return both / (
        (first[2] - first[0]) * (first[3] - first[1])
        + (second[2] - second[0]) * (second[3] - second[1])
        - both
    )


def interpolate_boxes(before, after, ratio):
    """앞뒤 탐지 결과 사이의 박스 위치를 계산하는 함수"""
    # 같은 클래스에서 가장 많이 겹치는 뒤 박스를 같은 물체로 보고 ratio만큼 옮김
    # ratio는 앞 탐지가 0, 뒤 탐지가 1일 때 표시할 프레임의 위치임
    boxes = []

    # 이미 짝지은 뒤 박스는 다른 앞 박스와 다시 짝짓지 않음
    used = []
    for box, cls, conf in before:
        matches = [
            (box_overlap(box, other[0]), index)
            for index, other in enumerate(after)
            if other[1] == cls and index not in used
        ]
        overlap, index = max(matches) if matches else (0.0, -1)
        if overlap >= MATCH_OVERLAP:
            used.append(index)
            boxes.append(
                (
                    [start + (end - start) * ratio for start, end in zip(box, after[index][0])],
                    cls,
                    conf,
                )
            )

        # 짝이 없는 박스는 가까운 쪽 탐지 결과에 있을 때만 그림
        elif ratio < 0.5:
            boxes.append((box, cls, conf))

    if ratio >= 0.5:
        boxes += [item for index, item in enumerate(after) if index not in used]

    return boxes
