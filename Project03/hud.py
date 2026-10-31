"""탐지 박스와 근거 조문 패널을 화면에 그리는 모듈"""

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
SMALL_FONT = ImageFont.truetype(FONT_PATH, 17)
LARGE_FONT = ImageFont.truetype(FONT_PATH, 22)

PANEL_WIDTH = 480
MAX_LINES = 5

# 프레임이 BGR 순서라 PIL에 넘기는 색도 BGR 순서로 적음
PANEL = (28, 28, 28)
STATE = (40, 180, 255)
OTHER = (140, 140, 140)
TITLE = (245, 245, 245)
BODY = (200, 200, 200)
MUTED = (130, 130, 130)


def wrap(pen, text, width):
    """낱말을 하나씩 붙이다가 폭을 넘으면 줄을 바꾼 줄 목록을 만드는 함수"""
    lines = []
    for word in text.split(" "):
        if lines and pen.textlength(lines[-1] + " " + word, font=SMALL_FONT) <= width:
            lines[-1] += " " + word
        else:
            lines.append(word)
    if len(lines) > MAX_LINES:
        lines = lines[:MAX_LINES]
        lines[-1] = lines[-1][:-1] + "…"
    return lines


def draw(frame, boxes, states, articles, corpus_name, fps):
    """프레임에 박스를 그리고 오른쪽에 장면 상태별 조문 패널을 붙이는 함수"""
    for x1, y1, x2, y2, state in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), STATE if state else OTHER, 2)
    canvas = cv2.copyMakeBorder(frame, 0, 0, 0, PANEL_WIDTH, cv2.BORDER_CONSTANT, value=PANEL)
    # cv2.putText는 한글을 그리지 못해 글자는 모두 PIL로 그림
    image = Image.fromarray(canvas)
    pen = ImageDraw.Draw(image)
    for x1, y1, x2, y2, state in boxes:
        if state:
            pen.text((x1, max(y1 - 26, 0)), state, font=SMALL_FONT, fill=STATE)

    x = frame.shape[1] + 24
    pen.text((x, 20), corpus_name, font=SMALL_FONT, fill=MUTED)
    pen.text((canvas.shape[1] - 24, 20), f"{fps:.1f} FPS", font=SMALL_FONT, fill=MUTED, anchor="ra")
    y = 64
    for state in states:
        article = articles[state]
        pen.text((x, y), state, font=LARGE_FONT, fill=STATE)
        y += 34
        pen.text((x, y), article["id"] + " " + article["title"], font=SMALL_FONT, fill=TITLE)
        y += 28
        for line in wrap(pen, article["text"], PANEL_WIDTH - 48):
            pen.text((x, y), line, font=SMALL_FONT, fill=BODY)
            y += 25
        y += 22
    return np.array(image)
