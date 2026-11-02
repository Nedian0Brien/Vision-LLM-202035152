"""탐지 박스와 엣지 케이스별 지시문과 근거 문장 패널을 화면에 그리는 모듈"""

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

# 1280 픽셀 영상 옆에 붙여 전체 폭이 1920 픽셀 화면에 맞게 함
PANEL_WIDTH = 640
COLUMN_WIDTH = 284  # 패널 좌우 여백 24와 두 열 사이 간격 24를 뺀 폭을 반으로 나눔

# 프레임이 BGR 순서라 PIL에 넘기는 색도 BGR 순서로 적음
PANEL = (28, 28, 28)
STATE = (40, 180, 255)
OTHER = (140, 140, 140)
TITLE = (245, 245, 245)
BODY = (200, 200, 200)
MUTED = (130, 130, 130)


def wrap(pen, text, font, max_lines):
    """낱말을 하나씩 붙이다가 열 폭을 넘으면 줄을 바꾼 줄 목록을 만드는 함수"""
    lines = []
    for word in text.split(" "):
        if lines and pen.textlength(lines[-1] + " " + word, font=font) <= COLUMN_WIDTH:
            lines[-1] += " " + word
        else:
            lines.append(word)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:-1] + "…"
    return lines


def draw(frame, boxes, panel, corpus_name, fps):
    """프레임에 박스를 그리고 오른쪽 패널의 왼쪽 열에 지시문, 오른쪽 열에 근거 조문을 붙이는 함수"""
    for x1, y1, x2, y2, state in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), STATE if state else OTHER, 2)
    canvas = cv2.copyMakeBorder(frame, 0, 0, 0, PANEL_WIDTH, cv2.BORDER_CONSTANT, value=PANEL)
    # cv2.putText는 한글을 그리지 못해 글자는 모두 PIL로 그림
    image = Image.fromarray(canvas)
    pen = ImageDraw.Draw(image)
    for x1, y1, x2, y2, state in boxes:
        if state:
            pen.text((x1, max(y1 - 26, 0)), state, font=MEDIUM_FONT, fill=STATE)

    left = frame.shape[1] + 24
    right = left + COLUMN_WIDTH + 24
    pen.text((left, 20), corpus_name, font=SMALL_FONT, fill=MUTED)
    pen.text((canvas.shape[1] - 24, 20), f"{fps:.1f} FPS", font=SMALL_FONT, fill=MUTED, anchor="ra")
    # 엣지 케이스 3개가 720 픽셀 높이에 들어가도록 의무 하나는 3줄, 근거 문장은 7줄까지만 보여 줌
    y = 64
    shown = []
    for name, article, instruction in panel:
        # 정지 표지와 양보 표지처럼 같은 조항을 가리키는 엣지 케이스는 한 번만 보여 줌
        if article["id"] + article["title"] in shown or len(shown) == 3:
            continue
        shown.append(article["id"] + article["title"])
        pen.text((left, y), name, font=LARGE_FONT, fill=STATE)
        pen.text((left, y + 30), instruction["행동 종류"], font=MEDIUM_FONT, fill=STATE)
        left_y = y + 62
        # 의무 하나가 여러 줄로 넘어가도 의무끼리 구분되도록 의무 사이에 간격을 둠
        for duty in instruction["지시"].split("\n"):
            for line in wrap(pen, duty.strip(), MEDIUM_FONT, 3):
                pen.text((left, left_y), line, font=MEDIUM_FONT, fill=TITLE)
                left_y += 25
            left_y += 8
        right_y = y + 4
        for line in wrap(pen, article["id"] + " " + article["title"], SMALL_FONT, 2):
            pen.text((right, right_y), line, font=SMALL_FONT, fill=MUTED)
            right_y += 20
        right_y += 4
        for line in wrap(pen, " ".join(instruction["문장"]), SMALL_FONT, 7):
            pen.text((right, right_y), line, font=SMALL_FONT, fill=BODY)
            right_y += 20
        y = max(left_y, right_y) + 24
    return np.array(image)
