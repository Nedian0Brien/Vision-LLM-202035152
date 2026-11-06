"""YOLO 탐지 결과를 Gemma 입력 크롭으로 바꾸는 모듈"""

import cv2
import numpy as np

# COCO 클래스 번호: 0 사람, 1 자전거, 2 승용차, 7 트럭, 9 신호등, 11 정지 표지
CLASS_NAMES = {0: "사람", 1: "자전거", 2: "승용차", 7: "트럭", 9: "신호등", 11: "정지 표지"}
CLASSES = list(CLASS_NAMES)
PERSON_CLASSES = [0]
LIGHT_CLASSES = [9]
SIGN_CLASSES = [9, 11]
# 개발용 세트의 신호등과 정지 표지 장면 13개에서 640 입력은 7개, 1280 입력은 12개를 찾음(2026-11-03)
# 1280×720 프레임의 입력은 736×1280이고, 고정 크기 TensorRT 엔진에 정수 1280을 넘기면 1280×1280 입력이 되어 멈춤
IMAGE_SIZE = [736, 1280]
# 크롭에는 먼 신호등도 넣으려고 낮은 신뢰도까지 쓰고, 화면에는 기존 기준의 박스만 그림
DETECT_CONF = 0.25
SHOW_CONF = 0.4
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
    areas.sort(reverse=True)
    crops = []
    for area, x1, y1, x2, y2 in areas[:3]:
        size = max(x2 - x1, y2 - y1)
        crop = frame[max(0, int(y1 - size)):min(height, int(y2 + size)), max(0, int(x1 - 1.5 * size)):min(width, int(x2 + 1.5 * size))]
        # 적색 신호 판정이 전체 프레임에서는 0~2/9, 크롭에서는 9/9였음(2026-11-05 개발용 측정)
        crop_width = int(crop.shape[1] * STRIP_HEIGHT / crop.shape[0])
        crops.append(cv2.resize(crop, (crop_width, STRIP_HEIGHT), interpolation=cv2.INTER_CUBIC))
        crops.append(np.zeros((STRIP_HEIGHT, 16, 3), np.uint8))
    return np.hstack(crops)


def road_crop(frame):
    """노면 글자가 보이는 차로 앞쪽 영역을 자르는 함수"""
    height, width = frame.shape[:2]
    return frame[int(0.55 * height):int(0.92 * height), int(0.15 * width):int(0.85 * width)]
