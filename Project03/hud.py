"""탐지 박스, 버튼, 지시문과 근거 조항을 화면에 그리는 모듈"""

import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from detect import CLASS_NAMES, SHOW_CONF

FONT_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Jetson Ubuntu의 fonts-noto-cjk 글꼴
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Ubuntu의 fonts-nanum 글꼴
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS 기본 한글 글꼴
]
FONT_PATH = [path for path in FONT_PATHS if os.path.exists(path)][0]
SMALL_FONT = ImageFont.truetype(FONT_PATH, 15)
MEDIUM_FONT = ImageFont.truetype(FONT_PATH, 19)
LARGE_FONT = ImageFont.truetype(FONT_PATH, 28)

# 1280 픽셀 영상 옆에 붙여 전체 폭이 1920 픽셀 화면에 맞게 함
PANEL_WIDTH = 640

# 프레임이 BGR 순서라 PIL에 넘기는 색도 BGR 순서로 적음
PANEL = (28, 28, 28)
STATE = (40, 180, 255)
BODY = (200, 200, 200)
MUTED = (130, 130, 130)
BUTTON = (55, 55, 55)
CHOSEN = (235, 235, 235)

# 프레임마다 PIL로 글자를 그리면 한 프레임에 25ms가 걸려 YOLO보다 오래 걸렸음(2026-11-03 Mac 측정)
LABELS = {}
for name in CLASS_NAMES.values():
    image = Image.new("RGB", (int(MEDIUM_FONT.getlength(name)) + 12, 28), PANEL)
    ImageDraw.Draw(image).text((6, 2), name, font=MEDIUM_FONT, fill=STATE)
    LABELS[name] = np.array(image)

BUTTON_HEIGHT = 34
BUTTON_GAP = 8
# 버튼 두 줄 아래부터 지시문을 그림
RULES_TOP = 2 * (BUTTON_HEIGHT + BUTTON_GAP) + 24
RULE_TEXT_LINES = 4


def button_boxes(frame_width, groups):
    """버튼 묶음마다 한 줄씩 패널 위쪽에 놓을 버튼의 묶음 이름, 번호, 글자, 사각형 목록을 만드는 함수"""
    buttons = []
    y = BUTTON_GAP
    for group, names in groups.items():
        width = (PANEL_WIDTH - 48 - BUTTON_GAP * (len(names) - 1)) // len(names)
        for index, name in enumerate(names):
            x = frame_width + 24 + index * (width + BUTTON_GAP)
            buttons.append((group, index, name, (x, y, x + width, y + BUTTON_HEIGHT)))
        y += BUTTON_HEIGHT + BUTTON_GAP
    return buttons


def wrap(text, font, width):
    """글자 폭을 재어 패널 폭에 맞게 줄을 나누는 함수"""
    lines = []
    line = ""
    # 한국어 조문은 띄어쓰기가 드물어 낱말이 아니라 글자 단위로 나눔
    for char in text:
        if font.getlength(line + char) > width:
            lines.append(line)
            line = ""
        line += char
    return lines + [line]


def draw_panel(height, frame_width, buttons, selected, rules):
    """버튼, 지시문, 근거 조항, 읽은 조문을 그린 패널 이미지를 만드는 함수"""
    image = Image.new("RGB", (PANEL_WIDTH, height), PANEL)
    pen = ImageDraw.Draw(image)
    # 버튼 사각형은 클릭 판정에 쓰는 창 좌표라 패널 왼쪽 끝만큼 빼서 그림
    for group, index, name, (x1, y1, x2, y2) in buttons:
        chosen = selected[group] == index
        pen.rectangle((x1 - frame_width, y1, x2 - frame_width, y2), fill=CHOSEN if chosen else BUTTON)
        pen.text(((x1 + x2) // 2 - frame_width, (y1 + y2) // 2), name, font=SMALL_FONT, fill=PANEL if chosen else BODY, anchor="mm")
    width = PANEL_WIDTH - 48
    y = RULES_TOP
    for rule in rules:
        for line in wrap(rule["instruction"], LARGE_FONT, width):
            pen.text((24, y), line, font=LARGE_FONT, fill=STATE)
            y += 36
        pen.text((24, y + 2), rule["law"], font=MEDIUM_FONT, fill=BODY)
        y += 30
        for line in wrap(rule["text"], SMALL_FONT, width)[:RULE_TEXT_LINES]:
            pen.text((24, y), line, font=SMALL_FONT, fill=MUTED)
            y += 21
        y += 22
    return np.array(image)


def draw(frame, result, panel_image, fps):
    """프레임에 박스와 클래스 이름을 그리고 미리 그린 패널을 오른쪽에 붙이는 함수"""
    for box, cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
        if conf >= SHOW_CONF:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), STATE, 2)
            label = LABELS[CLASS_NAMES[int(cls)]]
            height, width = label.shape[:2]
            top = max(y1 - height, 0)
            width = min(width, frame.shape[1] - x1)
            frame[top:top + height, x1:x1 + width] = label[:, :width]
    canvas = cv2.copyMakeBorder(frame, 0, 0, 0, PANEL_WIDTH, cv2.BORDER_CONSTANT, value=PANEL)
    canvas[:, frame.shape[1]:] = panel_image
    cv2.putText(canvas, f"{fps:.1f} FPS", (canvas.shape[1] - 110, canvas.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, MUTED, 1)
    return canvas
