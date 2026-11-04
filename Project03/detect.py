"""YOLO 탐지 결과를 탐지 목록으로 바꾸는 모듈"""

# COCO 클래스 번호: 0 사람, 1 자전거, 2 승용차, 7 트럭, 9 신호등, 11 정지 표지
CLASS_NAMES = {0: "사람", 1: "자전거", 2: "승용차", 7: "트럭", 9: "신호등", 11: "정지 표지"}
CLASSES = list(CLASS_NAMES)
# 개발용 세트의 신호등과 정지 표지 장면 13개에서 640 입력은 7개, 1280 입력은 12개를 찾음(2026-11-03)
IMAGE_SIZE = 1280


def detections(result):
    """탐지 결과에서 박스 좌표와 클래스 이름 목록을 만드는 함수"""
    boxes = []
    for box, cls in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist()):
        x1, y1, x2, y2 = map(int, box)
        boxes.append((x1, y1, x2, y2, CLASS_NAMES[int(cls)]))
    return boxes
