"""탐지 박스, 탭, 지시문, 근거 조항, 처리 단계를 화면에 그리는 모듈"""

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
TINY_FONT = ImageFont.truetype(FONT_PATH, 14)
SMALL_FONT = ImageFont.truetype(FONT_PATH, 15)
MEDIUM_FONT = ImageFont.truetype(FONT_PATH, 17)
RESULT_FONT = ImageFont.truetype(FONT_PATH, 26)
LARGE_FONT = ImageFont.truetype(FONT_PATH, 36)

# 1280 픽셀 영상 옆에 붙여 전체 폭이 1920 픽셀 화면에 맞게 함
PANEL_WIDTH = 640

# 프레임이 BGR 순서라 PIL에 넘기는 색도 BGR 순서로 적음
PANEL = (39, 35, 32)
INK = (236, 234, 232)
SOFT = (166, 160, 154)
FAINT = (103, 97, 92)
ACCENT = (132, 184, 214)
LINE = (71, 66, 62)

# 프레임마다 PIL로 글자를 그리면 한 프레임에 25ms가 걸려 YOLO보다 오래 걸렸음(2026-11-03 Mac 측정)
FAST_LABEL = "8배속"
LABELS = {}
for name in list(CLASS_NAMES.values()) + [FAST_LABEL]:
    image = Image.new("RGB", (int(SMALL_FONT.getlength(name)) + 12, 24), PANEL)
    ImageDraw.Draw(image).text((6, 2), name, font=SMALL_FONT, fill=INK)
    LABELS[name] = np.array(image)

TAB_ROWS = [16, 54]
HERO_TOP = 104
STEPS = ["보기", "인식", "조문", "읽기", "출력"]
STEP_X = [60, 190, 320, 450, 580]
RAIL_Y = 356
CARD_TOP = 424
IMAGE_SIZE = (292, 164)
# 호출마다 보여 줄 입력 이미지와 지금 하는 일
CALLS = {"choice": ("frame", "Gemma가 장면을 보고 있습니다"), "light": ("lights", "Gemma가 신호 색을 읽고 있습니다"),
         "pedestrian": ("people", "Gemma가 걷는 사람을 찾고 있습니다"), "crosswalk": ("road", "Gemma가 횡단보도를 확인합니다"), "rain": ("frame", "Gemma가 비가 오는지 봅니다"), "bus": ("buses", "Gemma가 통학버스인지 봅니다"),
         "hydrant": ("hydrants", "Gemma가 소화전인지 봅니다"), "zebra": ("road", "Gemma가 보호구역 횡단보도를 찾고 있습니다")}


def button_boxes(frame_width, groups):
    """묶음마다 한 줄씩 놓을 탭의 묶음 이름, 번호, 글자, 사각형 목록을 만드는 함수"""
    buttons = []
    for (group, names), y in zip(groups.items(), TAB_ROWS):
        x = frame_width + 28
        for index, name in enumerate(names):
            width = int(SMALL_FONT.getlength(name))
            buttons.append((group, index, name, (x, y, x + width, y + 26)))
            x += width + 22
    return buttons


def wrap(text, font, width):
    """글자 폭을 재어 패널 폭에 맞게 줄을 나누는 함수"""
    lines = []
    line = ""
    for char in text:
        if font.getlength(line + char) > width:
            # 낱말 중간에서 끊기지 않도록 마지막 띄어쓰기에서 나누고, 띄어쓰기가 없으면 그 글자에서 나눔
            cut = line.rfind(" ")
            if cut > 0:
                lines.append(line[:cut])
                line = line[cut + 1:]
            else:
                lines.append(line)
                line = ""
        line += char
    return lines + [line]


def draw_rail(pen, current):
    """처리 단계를 가로 한 줄로 그리고 지금 단계를 표시하는 함수"""
    pen.line((STEP_X[0], RAIL_Y, STEP_X[-1], RAIL_Y), fill=LINE, width=2)
    if current > 0:
        pen.line((STEP_X[0], RAIL_Y, STEP_X[current], RAIL_Y), fill=ACCENT, width=2)
    for index, (x, name) in enumerate(zip(STEP_X, STEPS)):
        if index == current:
            pen.ellipse((x - 9, RAIL_Y - 9, x + 9, RAIL_Y + 9), fill=PANEL, outline=ACCENT, width=3)
            pen.ellipse((x - 4, RAIL_Y - 4, x + 4, RAIL_Y + 4), fill=ACCENT)
        elif index < current:
            pen.ellipse((x - 5, RAIL_Y - 5, x + 5, RAIL_Y + 5), fill=ACCENT)
        else:
            pen.ellipse((x - 5, RAIL_Y - 5, x + 5, RAIL_Y + 5), fill=PANEL, outline=FAINT, width=2)
        color = INK if index == current else (SOFT if index < current else FAINT)
        pen.text((x, RAIL_Y + 24), name, font=SMALL_FONT, fill=color, anchor="ma")


def card_image(image, picture):
    """지금 호출의 입력 이미지를 비율을 유지해 모서리가 둥근 칸에 그리는 함수"""
    width, height = IMAGE_SIZE
    scale = min(width / picture.shape[1], height / picture.shape[0])
    size = (max(1, int(picture.shape[1] * scale)), max(1, int(picture.shape[0] * scale)))
    canvas = np.full((height, width, 3), LINE, np.uint8)
    top, left = (height - size[1]) // 2, (width - size[0]) // 2
    canvas[top:top + size[1], left:left + size[0]] = cv2.resize(picture, size, interpolation=cv2.INTER_AREA)
    mask = Image.new("L", IMAGE_SIZE, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, width - 1, height - 1), 8, fill=255)
    image.paste(Image.fromarray(canvas), (28, CARD_TOP), mask)


def top_odds(answer):
    """답과 가장 높은 확률을 한 줄 글로 만드는 함수"""
    if answer[2]:
        return f"{answer[0]}  {max(probability for word, probability in answer[2]) * 100:.0f}%"
    return answer[0]


def draw_recognition_card(image, pen, status, scene):
    """지금 호출의 입력 이미지, 하는 일, 답을 그리는 함수"""
    source, doing = CALLS[scene]
    picture = status["images"].get(source)
    if picture is not None:
        card_image(image, picture)
    pen.text((344, CARD_TOP + 2), doing, font=SMALL_FONT, fill=INK)
    answer = status["answers"].get(scene)
    if answer:
        y = CARD_TOP + 32
        for line in wrap(top_odds(answer), RESULT_FONT, 270)[:2]:
            pen.text((344, y), line, font=RESULT_FONT, fill=ACCENT)
            y += 34
        pen.text((344, y + 6), f"입력 {answer[3]}토큰", font=TINY_FONT, fill=SOFT)
        pen.text((344, y + 28), f"{answer[1]:.2f}초", font=TINY_FONT, fill=SOFT)


def draw_law_card(pen, status, rule, phase):
    """새 규칙이 나올 때 조문, 읽기, 출력을 한 단계씩 그리는 함수"""
    if phase == 0:
        pen.text((28, CARD_TOP), "법령에서 조문을 찾았습니다" if rule["text"] else "조문을 찾지 않습니다", font=SMALL_FONT, fill=INK)
        pen.text((28, CARD_TOP + 30), rule["law"], font=RESULT_FONT, fill=ACCENT)
    elif phase == 1:
        pen.text((28, CARD_TOP), "Gemma가 조문을 읽고 행동을 골랐습니다" if rule["text"] else "Gemma가 조문 없이 행동을 골랐습니다", font=SMALL_FONT, fill=INK)
        odds = status["odds"].get(rule["instruction"], [])
        best = f"  {max(probability for word, probability in odds) * 100:.0f}%" if odds else ""
        pen.text((28, CARD_TOP + 30), rule["action"] + best, font=RESULT_FONT, fill=ACCENT)
        y = CARD_TOP + 76
        for probability, word in sorted([(probability, word) for word, probability in odds], reverse=True)[1:3]:
            pen.text((28, y), f"{word}  {probability * 100:.0f}%", font=TINY_FONT, fill=SOFT)
            y += 22
    else:
        pen.text((28, CARD_TOP), "VLA 모델로 내보냅니다", font=SMALL_FONT, fill=INK)
        pen.text((28, CARD_TOP + 30), rule["action"], font=RESULT_FONT, fill=ACCENT)
        y = CARD_TOP + 76
        for line in wrap(f'"law": "{rule["law"]}", "instruction": "{rule["instruction"]}"', SMALL_FONT, 584)[:3]:
            pen.text((28, y), line, font=SMALL_FONT, fill=SOFT)
            y += 22


def draw_panel(height, frame_width, buttons, selected, rules, status, scene, reading, phase):
    """탭, 지시문, 근거 조항, 처리 단계를 그린 패널 이미지를 만드는 함수"""
    image = Image.new("RGB", (PANEL_WIDTH, height), PANEL)
    pen = ImageDraw.Draw(image)
    # 탭 사각형은 클릭 판정에 쓰는 창 좌표라 패널 왼쪽 끝만큼 빼서 그림
    for group, index, name, (x1, y1, x2, y2) in buttons:
        chosen = selected[group] == index
        pen.text((x1 - frame_width, y1), name, font=SMALL_FONT, fill=INK if chosen else FAINT)
        if chosen:
            pen.line((x1 - frame_width, y2 - 1, x2 - frame_width, y2 - 1), fill=ACCENT, width=2)
    y = HERO_TOP
    if rules:
        # 규칙은 처음 나온 순서로 쌓이므로 가장 새로 나온 규칙을 크게 그림
        rule = rules[-1]
        for line in wrap(rule["instruction"], LARGE_FONT, 584)[:2]:
            pen.text((28, y), line, font=LARGE_FONT, fill=INK)
            y += 44
        pen.text((28, y + 8), rule["law"], font=MEDIUM_FONT, fill=ACCENT)
        y += 34
        for line in wrap(rule["text"], SMALL_FONT, 584)[:2]:
            pen.text((28, y), line, font=SMALL_FONT, fill=SOFT)
            y += 22
        if len(rules) > 1:
            pen.text((28, y + 4), f"함께 적용: {rules[-2]['instruction']}" + (f" 외 {len(rules) - 2}개" if len(rules) > 2 else ""), font=SMALL_FONT, fill=SOFT)
    else:
        pen.text((28, y), "적용할 규칙 없음", font=LARGE_FONT, fill=FAINT)
    if scene == "skip":
        draw_rail(pen, -1)
        pen.text((28, CARD_TOP), "모델을 멈추고 다음 장면으로 건너뜁니다", font=SMALL_FONT, fill=SOFT)
    elif reading is not None:
        draw_rail(pen, 2 + phase)
        draw_law_card(pen, status, reading, phase)
    else:
        draw_rail(pen, 1)
        if scene in CALLS:
            draw_recognition_card(image, pen, status, scene)
    pen.text((28, 618), "VLA 출력", font=TINY_FONT, fill=FAINT)
    pen.text((100, 616), wrap("  ".join([rule["action"] for rule in rules]) if rules else "없음", SMALL_FONT, 500)[0], font=SMALL_FONT, fill=INK if rules else FAINT)
    pen.text((28, 696), f"vla-output.jsonl  {status['outputs']}줄", font=TINY_FONT, fill=FAINT)
    return np.array(image)


def draw(frame, result, panel_image, fps):
    """프레임에 박스와 클래스 이름을 그리고 미리 그린 패널을 오른쪽에 붙이는 함수"""
    # 배속 구간에는 탐지 결과가 없음
    if result is None:
        label = LABELS[FAST_LABEL]
        frame[24:24 + label.shape[0], 24:24 + label.shape[1]] = label
        boxes = []
    else:
        boxes = zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist())
    for box, cls, conf in boxes:
        if conf >= SHOW_CONF:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), ACCENT, 1)
            label = LABELS[CLASS_NAMES[int(cls)]]
            height, width = label.shape[:2]
            top = max(y1 - height, 0)
            width = min(width, frame.shape[1] - x1)
            frame[top:top + height, x1:x1 + width] = label[:, :width]
    canvas = cv2.copyMakeBorder(frame, 0, 0, 0, PANEL_WIDTH, cv2.BORDER_CONSTANT, value=PANEL)
    canvas[:, frame.shape[1]:] = panel_image
    cv2.putText(canvas, f"{fps:.1f} FPS", (canvas.shape[1] - 110, canvas.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, FAINT, 1)
    return canvas
