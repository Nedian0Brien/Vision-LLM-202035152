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
SMALL_FONT = ImageFont.truetype(FONT_PATH, 15)
MEDIUM_FONT = ImageFont.truetype(FONT_PATH, 19)
LARGE_FONT = ImageFont.truetype(FONT_PATH, 22)

PANEL_WIDTH = 480
TEXT_WIDTH = PANEL_WIDTH - 48

# 프레임이 BGR 순서라 PIL에 넘기는 색도 BGR 순서로 적음
PANEL = (28, 28, 28)
STATE = (40, 180, 255)
VISION = (230, 200, 80)
OTHER = (140, 140, 140)
TITLE = (245, 245, 245)
BODY = (200, 200, 200)
MUTED = (130, 130, 130)


def wrap(pen, text, font, max_lines):
    """낱말을 하나씩 붙이다가 패널 폭을 넘으면 줄을 바꾼 줄 목록을 만드는 함수"""
    lines = []
    for word in text.split(" "):
        if lines and pen.textlength(lines[-1] + " " + word, font=font) <= TEXT_WIDTH:
            lines[-1] += " " + word
        else:
            lines.append(word)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:-1] + "…"
    return lines


def draw(frame, boxes, panel, corpus_name, fps):
    """프레임에 박스를 그리고 오른쪽에 상태별 지시문과 근거 조문 패널을 붙이는 함수"""
    for x1, y1, x2, y2, state in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), STATE if state else OTHER, 2)
    canvas = cv2.copyMakeBorder(frame, 0, 0, 0, PANEL_WIDTH, cv2.BORDER_CONSTANT, value=PANEL)
    # cv2.putText는 한글을 그리지 못해 글자는 모두 PIL로 그림
    image = Image.fromarray(canvas)
    pen = ImageDraw.Draw(image)
    for x1, y1, x2, y2, state in boxes:
        if state:
            pen.text((x1, max(y1 - 26, 0)), state, font=MEDIUM_FONT, fill=STATE)

    x = frame.shape[1] + 24
    pen.text((x, 20), corpus_name, font=SMALL_FONT, fill=MUTED)
    pen.text((canvas.shape[1] - 24, 20), f"{fps:.1f} FPS", font=SMALL_FONT, fill=MUTED, anchor="ra")
    # 상태 3개가 720 픽셀 높이에 들어가도록 지시문은 3줄, 근거 원문은 2줄까지만 보여 줌
    y = 64
    shown = []
    for name, article, instruction, from_vision in panel:
        # 보행자와 횡단보도처럼 같은 조항을 가리키는 상태는 한 번만 보여 줌
        if article["id"] + article["title"] in shown or len(shown) == 3:
            continue
        shown.append(article["id"] + article["title"])
        pen.text((x, y), name, font=LARGE_FONT, fill=VISION if from_vision else STATE)
        y += 34
        for line in wrap(pen, instruction, MEDIUM_FONT, 3):
            pen.text((x, y), line, font=MEDIUM_FONT, fill=TITLE)
            y += 26
        y += 6
        pen.text((x, y), article["id"] + " " + article["title"], font=SMALL_FONT, fill=MUTED)
        y += 21
        for line in wrap(pen, article["text"], SMALL_FONT, 2):
            pen.text((x, y), line, font=SMALL_FONT, fill=BODY)
            y += 21
        y += 22
    return np.array(image)
