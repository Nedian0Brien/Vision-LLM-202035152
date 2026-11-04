"""탐지 박스와 버튼 패널을 화면에 그리는 모듈"""

import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Jetson Ubuntu의 fonts-noto-cjk 글꼴
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Ubuntu의 fonts-nanum 글꼴
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS 기본 한글 글꼴
]
FONT_PATH = [path for path in FONT_PATHS if os.path.exists(path)][0]
SMALL_FONT = ImageFont.truetype(FONT_PATH, 15)
MEDIUM_FONT = ImageFont.truetype(FONT_PATH, 19)

# 1280 픽셀 영상 옆에 붙여 전체 폭이 1920 픽셀 화면에 맞게 함
PANEL_WIDTH = 640

# 프레임이 BGR 순서라 PIL에 넘기는 색도 BGR 순서로 적음
PANEL = (28, 28, 28)
STATE = (40, 180, 255)
BODY = (200, 200, 200)
MUTED = (130, 130, 130)
BUTTON = (55, 55, 55)
CHOSEN = (235, 235, 235)

LABELS = {}

BUTTON_HEIGHT = 34
BUTTON_GAP = 8


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


def label_image(text):
    """클래스 이름 글자를 처음 한 번만 그려 두고 다시 쓰는 함수"""
    # 프레임마다 PIL로 글자를 그리면 한 프레임에 25ms가 걸려 YOLO보다 오래 걸렸음(2026-11-03 Mac 측정)
    if text not in LABELS:
        image = Image.new("RGB", (int(MEDIUM_FONT.getlength(text)) + 12, 28), PANEL)
        ImageDraw.Draw(image).text((6, 2), text, font=MEDIUM_FONT, fill=STATE)
        LABELS[text] = np.array(image)
    return LABELS[text]


def draw_buttons(pen, frame_width, buttons, selected):
    """고른 버튼은 밝게, 나머지는 어둡게 버튼을 그리는 함수"""
    # 버튼 사각형은 클릭 판정에 쓰는 창 좌표라 패널 왼쪽 끝만큼 빼서 그림
    for group, index, name, (x1, y1, x2, y2) in buttons:
        chosen = selected[group] == index
        pen.rectangle((x1 - frame_width, y1, x2 - frame_width, y2), fill=CHOSEN if chosen else BUTTON)
        pen.text(((x1 + x2) // 2 - frame_width, (y1 + y2) // 2), name, font=SMALL_FONT, fill=PANEL if chosen else BODY, anchor="mm")


def draw_panel(height, frame_width, buttons, selected):
    """버튼을 그린 패널 이미지를 만드는 함수"""
    image = Image.new("RGB", (PANEL_WIDTH, height), PANEL)
    draw_buttons(ImageDraw.Draw(image), frame_width, buttons, selected)
    return np.array(image)


def draw(frame, boxes, panel_image, fps):
    """프레임에 박스와 상태 이름을 그리고 미리 그린 패널을 오른쪽에 붙이는 함수"""
    for x1, y1, x2, y2, state in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), STATE if state else OTHER, 2)
        if state:
            label = label_image(state)
            height, width = label.shape[:2]
            top = max(y1 - height, 0)
            width = min(width, frame.shape[1] - x1)
            frame[top:top + height, x1:x1 + width] = label[:, :width]
    canvas = cv2.copyMakeBorder(frame, 0, 0, 0, PANEL_WIDTH, cv2.BORDER_CONSTANT, value=PANEL)
    canvas[:, frame.shape[1]:] = panel_image
    cv2.putText(canvas, f"{fps:.1f} FPS", (canvas.shape[1] - 110, canvas.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, MUTED, 1)
    return canvas
