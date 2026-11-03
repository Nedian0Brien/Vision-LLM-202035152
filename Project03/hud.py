"""탐지 박스와 버튼, 엣지 케이스별 지시문과 근거 조문 패널을 화면에 그리는 모듈"""

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
ACTION_FONT = ImageFont.truetype(FONT_PATH, 28)

# 1280 픽셀 영상 옆에 붙여 전체 폭이 1920 픽셀 화면에 맞게 함
PANEL_WIDTH = 640
COLUMN_WIDTH = 284  # 패널 좌우 여백 24와 두 열 사이 간격 24를 뺀 폭을 반으로 나눔
FULL_WIDTH = 592  # 패널 좌우 여백 24를 뺀 폭

# 프레임이 BGR 순서라 PIL에 넘기는 색도 BGR 순서로 적음
PANEL = (28, 28, 28)
STATE = (40, 180, 255)
OTHER = (140, 140, 140)
TITLE = (245, 245, 245)
BODY = (200, 200, 200)
MUTED = (130, 130, 130)
BUTTON = (55, 55, 55)
CHOSEN = (235, 235, 235)

LABELS = {}

BUTTON_HEIGHT = 34
BUTTON_GAP = 8


def wrap(pen, text, font, width, max_lines=None):
    """낱말을 하나씩 붙이다가 폭을 넘으면 줄을 바꾼 줄 목록을 만드는 함수"""
    lines = []
    for word in text.split(" "):
        if lines and pen.textlength(lines[-1] + " " + word, font=font) <= width:
            lines[-1] += " " + word
        else:
            lines.append(word)
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:-1] + "…"
    return lines


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
    """상태 이름 글자를 처음 한 번만 그려 두고 다시 쓰는 함수"""
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


def draw_panel(height, frame_width, panel, buttons, selected):
    """버튼, 지시문 열, 근거 조문 열을 그린 패널 이미지를 만드는 함수"""
    image = Image.new("RGB", (PANEL_WIDTH, height), PANEL)
    pen = ImageDraw.Draw(image)
    draw_buttons(pen, frame_width, buttons, selected)

    left = 24
    right = left + COLUMN_WIDTH + 24
    # 엣지 케이스 3개가 720 픽셀 높이에 들어가도록 의무 하나는 3줄, 근거 문장은 7줄까지만 보여 줌
    y = buttons[-1][3][3] + 28
    shown = []
    for name, article, instruction in panel:
        # 정지 표지와 양보 표지처럼 같은 조항을 가리키는 엣지 케이스는 한 번만 보여 줌
        key = name if article is None else article["id"] + article["title"]
        if key in shown or len(shown) == 3:
            continue
        shown.append(key)
        pen.text((left, y), name, font=LARGE_FONT, fill=STATE)
        # 지시문을 만드는 동안에는 장면을 찾았다는 것만 이름으로 보여 줌
        if instruction is None:
            y += 54
            continue
        pen.text((left, y + 30), instruction["행동 종류"], font=MEDIUM_FONT, fill=STATE)
        left_y = y + 62
        # 의무 하나가 여러 줄로 넘어가도 의무끼리 구분되도록 의무 사이에 간격을 둠
        for duty in instruction["지시"].split("\n"):
            for line in wrap(pen, duty.strip(), MEDIUM_FONT, COLUMN_WIDTH, 3):
                pen.text((left, left_y), line, font=MEDIUM_FONT, fill=TITLE)
                left_y += 25
            left_y += 8
        right_y = y + 4
        for line in wrap(pen, article["id"] + " " + article["title"], SMALL_FONT, COLUMN_WIDTH, 2):
            pen.text((right, right_y), line, font=SMALL_FONT, fill=MUTED)
            right_y += 20
        right_y += 4
        for line in wrap(pen, " ".join(instruction["문장"]), SMALL_FONT, COLUMN_WIDTH, 7):
            pen.text((right, right_y), line, font=SMALL_FONT, fill=BODY)
            right_y += 20
        y = max(left_y, right_y) + 24
    return np.array(image)


def draw_compare_panel(height, frame_width, panel, buttons, selected):
    """엣지 케이스마다 세 법령의 행동 종류와 지시문을 위아래로 쌓은 비교 패널 이미지를 만드는 함수"""
    image = Image.new("RGB", (PANEL_WIDTH, height), PANEL)
    pen = ImageDraw.Draw(image)
    draw_buttons(pen, frame_width, buttons, selected)

    left = 24
    y = buttons[-1][3][3] + 28
    for name, blocks in panel:
        # 법령마다 다른 내용은 의무 끝의 동사에 있어 지시문 줄을 자르지 않음
        wrapped = [[wrap(pen, duty.strip(), MEDIUM_FONT, FULL_WIDTH) for duty in instruction["지시"].split("\n")]
                   for corpus_name, article, instruction in blocks]
        # 첫 법령 블록까지 들어가지 않으면 이름만 남지 않도록 그 엣지 케이스부터 그리지 않음
        if y + 40 + 58 + sum(len(lines) * 25 + 6 for lines in wrapped[0]) > height:
            break
        pen.text((left, y), name, font=LARGE_FONT, fill=TITLE)
        y += 40
        for index, (corpus_name, article, instruction) in enumerate(blocks):
            duties = wrapped[index]
            # 다 그리지 못하는 블록은 일부만 보여 주지 않고 거기서 멈춤
            if y + 58 + sum(len(lines) * 25 + 6 for lines in duties) > height:
                break
            pen.text((left, y), corpus_name + "  " + article["id"], font=SMALL_FONT, fill=MUTED)
            pen.text((left, y + 20), instruction["행동 종류"], font=ACTION_FONT, fill=STATE)
            y += 58
            for lines in duties:
                for line in lines:
                    pen.text((left, y), line, font=MEDIUM_FONT, fill=TITLE)
                    y += 25
                y += 6
            y += 14
        y += 16
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
