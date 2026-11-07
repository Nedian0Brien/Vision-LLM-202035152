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
NODE_WIDTH = 320
# 세 열을 합쳐 1920 폭 화면에 들어가도록 영상을 줄여 그림
VIDEO_WIDTH = 960
VIDEO_HEIGHT = 540
VIDEO_SCALE = 0.75
LOAD_BAR_WIDTH = 720
SPINNER_POINTS = [(0, -7), (5, -5), (7, 0), (5, 5), (0, 7), (-5, 5), (-7, 0), (-5, -5)]

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
# 지시문 목록 한 줄의 높이와 단계 표시 위 여백
ROW_HEIGHT = 27
RAIL_GAP = 18
CARD_TOP = 424
# 법령 탭 줄의 오른쪽 끝에 놓는 버튼의 창 좌표
MIC_BOX = (VIDEO_WIDTH + 540, TAB_ROWS[1], VIDEO_WIDTH + PANEL_WIDTH - 28, TAB_ROWS[1] + 26)
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


def draw_answer_card(pen, voice):
    """질문과 답 또는 음성 입력 상태를 그리는 함수"""
    y = CARD_TOP
    if voice["listening"]:
        pen.text((28, y), "듣고 있습니다", font=RESULT_FONT, fill=ACCENT)
        return
    if voice["thinking"]:
        pen.text((28, y), wrap(voice["thinking"], MEDIUM_FONT, 584)[0], font=MEDIUM_FONT, fill=SOFT)
        pen.text((28, y + 32), "법령을 찾아 답을 쓰고 있습니다", font=RESULT_FONT, fill=ACCENT)
        return
    pen.text((28, y), wrap(voice["answer"]["question"], MEDIUM_FONT, 584)[0], font=MEDIUM_FONT, fill=SOFT)
    y += 30
    for line in wrap(voice["answer"]["answer"], RESULT_FONT, 584)[:4]:
        pen.text((28, y), line, font=RESULT_FONT, fill=INK)
        y += 34


def draw_panel(height, frame_width, buttons, selected, rules, status, scene, reading, phase, voice):
    """탭, 지시문, 근거 조항, 처리 단계를 그린 패널 이미지를 만드는 함수"""
    image = Image.new("RGB", (PANEL_WIDTH, height), PANEL)
    pen = ImageDraw.Draw(image)
    # 탭 사각형은 클릭 판정에 쓰는 창 좌표라 패널 왼쪽 끝만큼 빼서 그림
    for group, index, name, (x1, y1, x2, y2) in buttons:
        chosen = selected[group] == index
        pen.text((x1 - frame_width, y1), name, font=SMALL_FONT, fill=INK if chosen else FAINT)
        if chosen:
            pen.line((x1 - frame_width, y2 - 1, x2 - frame_width, y2 - 1), fill=ACCENT, width=2)
    pen.text((MIC_BOX[0] - frame_width, MIC_BOX[1]), "음성 질문", font=SMALL_FONT, fill=ACCENT if voice["listening"] else SOFT)
    y = HERO_TOP
    if rules:
        # 규칙은 처음 나온 순서로 쌓이므로 가장 새로 나온 규칙을 크게 그림
        rule = rules[-1]
        for line in wrap(rule["instruction"], LARGE_FONT, 584)[:2]:
            pen.text((28, y), line, font=LARGE_FONT, fill=INK)
            y += 44
        pen.text((28, y + 8), rule["law"], font=MEDIUM_FONT, fill=ACCENT)
        y += 34
        others = rules[-2::-1]
        if not others:
            for line in wrap(rule["text"], SMALL_FONT, 584)[:2]:
                pen.text((28, y), line, font=SMALL_FONT, fill=SOFT)
                y += 22
        # 지시문 줄 수에 따라 단계 표시 위에 남는 만큼만 최신 순으로 그림
        rows = (RAIL_Y - RAIL_GAP - y) // ROW_HEIGHT
        hidden = len(others) - rows + 1 if len(others) > rows else 0
        for other in others[:rows - 1] if hidden else others:
            pen.text((28, y), wrap(other["instruction"], MEDIUM_FONT, 408)[0], font=MEDIUM_FONT, fill=INK)
            pen.text((448, y + 2), other["law"], font=SMALL_FONT, fill=ACCENT)
            y += ROW_HEIGHT
        if hidden:
            pen.text((28, y), f"외 {hidden}개", font=SMALL_FONT, fill=SOFT)
    else:
        pen.text((28, y), "적용할 규칙 없음", font=LARGE_FONT, fill=FAINT)
    if voice["listening"] or voice["thinking"] or voice["answer"]:
        draw_rail(pen, 1 if voice["listening"] else 3 if voice["thinking"] else 4)
        draw_answer_card(pen, voice)
    elif scene == "skip":
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
    pen.text((28, 696), voice["error"] or f"vla-output.jsonl  {status['outputs']}줄", font=TINY_FONT, fill=ACCENT if voice["error"] else FAINT)
    return np.array(image)


def draw_loading(height, steps, step, elapsed):
    """불러오는 단계와 진행 막대를 그린 시작 화면을 만드는 함수"""
    width = VIDEO_WIDTH + PANEL_WIDTH + NODE_WIDTH
    image = Image.new("RGB", (width, height), PANEL)
    pen = ImageDraw.Draw(image)
    left = (width - LOAD_BAR_WIDTH) // 2
    top = height // 2 - 60
    total = sum([seconds for name, seconds in steps])
    # 단계가 예상보다 길어도 막대가 다음 단계까지 넘어가지 않게 95%에서 멈춤
    current = steps[min(step, len(steps) - 1)]
    done = sum([seconds for name, seconds in steps[:step]]) + (min(elapsed, current[1] * 0.95) if step < len(steps) else 0)
    pen.text((left, top), "자율주행 도로교통법 지시문 모듈", font=LARGE_FONT, fill=INK)
    pen.rectangle((left, top + 70, left + LOAD_BAR_WIDTH, top + 76), fill=LINE)
    pen.rectangle((left, top + 70, left + int(LOAD_BAR_WIDTH * done / total), top + 76), fill=ACCENT)
    pen.text((left, top + 92), current[0], font=MEDIUM_FONT, fill=SOFT)
    pen.text((left + LOAD_BAR_WIDTH - 40, top + 92), f"{min(step + 1, len(steps))} / {len(steps)}", font=SMALL_FONT, fill=FAINT)
    return np.array(image)


def draw_spinner(pen, x, y, spinner):
    """점 여덟 개 중 하나를 밝혀 도는 스피너를 그리는 함수"""
    for index in range(8):
        dot_x = x + SPINNER_POINTS[index][0]
        dot_y = y + SPINNER_POINTS[index][1]
        pen.ellipse((dot_x - 1, dot_y - 1, dot_x + 1, dot_y + 1), fill=ACCENT if index == spinner else LINE)


def draw_nodes(height, nodes, spinner):
    """검색된 조문 노드를 이번 판단에서 찾은 것부터 그린 열 이미지를 만드는 함수"""
    image = Image.new("RGB", (NODE_WIDTH, height), PANEL)
    pen = ImageDraw.Draw(image)
    pen.line((0, 0, 0, height), fill=LINE, width=1)
    pen.text((20, 20), "검색된 조문", font=SMALL_FONT, fill=SOFT)
    y = 56
    drawn = []
    for node in [node for node in nodes if node["fresh"]] + [node for node in nodes if not node["fresh"]]:
        article = node["article"]
        # 여러 상황이 같은 조문을 찾으면 한 번만 그림
        if article["id"] in drawn:
            continue
        results = wrap(node["instruction"], SMALL_FONT, 284)[:2] if node["instruction"] else ["지시문 생성 중..."]
        lines = wrap(article["text"], TINY_FONT, 284)[:2]
        if y + 48 + 21 * len(results) + 19 * len(lines) > height - 40:
            pen.text((20, y), "더 찾은 조문 있음", font=TINY_FONT, fill=SOFT)
            break
        drawn.append(article["id"])
        pen.text((20, y), article["id"], font=MEDIUM_FONT, fill=ACCENT if node["fresh"] else SOFT)
        pen.text((20, y + 24), article["title"], font=SMALL_FONT, fill=SOFT if node["fresh"] else FAINT)
        y += 48
        if node["instruction"]:
            for line in results:
                pen.text((20, y), line, font=SMALL_FONT, fill=INK)
                y += 21
        else:
            draw_spinner(pen, 28, y + 10, spinner)
            pen.text((44, y), results[0], font=SMALL_FONT, fill=ACCENT)
            y += 21
        for line in lines:
            pen.text((20, y), line, font=TINY_FONT, fill=FAINT)
            y += 19
        y += 16
    if not drawn:
        pen.text((20, y), "검색된 조문 없음", font=MEDIUM_FONT, fill=FAINT)
    return np.array(image)


def draw(frame, result, panel_image, nodes_image, fps):
    """프레임에 박스와 클래스 이름을 그리고 패널과 조문 열을 오른쪽에 붙이는 함수"""
    frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
    # 배속 구간에는 탐지 결과가 없음
    if result is None:
        label = LABELS[FAST_LABEL]
        frame[24:24 + label.shape[0], 24:24 + label.shape[1]] = label
        boxes = []
    else:
        boxes = zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist())
    for box, cls, conf in boxes:
        if conf >= SHOW_CONF:
            x1, y1, x2, y2 = [int(value * VIDEO_SCALE) for value in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), ACCENT, 1)
            label = LABELS[CLASS_NAMES[int(cls)]]
            height, width = label.shape[:2]
            top = max(y1 - height, 0)
            width = min(width, frame.shape[1] - x1)
            frame[top:top + height, x1:x1 + width] = label[:, :width]
    canvas = cv2.copyMakeBorder(frame, 0, panel_image.shape[0] - VIDEO_HEIGHT, 0, PANEL_WIDTH + NODE_WIDTH, cv2.BORDER_CONSTANT, value=PANEL)
    canvas[:, VIDEO_WIDTH:VIDEO_WIDTH + PANEL_WIDTH] = panel_image
    canvas[:, VIDEO_WIDTH + PANEL_WIDTH:] = nodes_image
    cv2.putText(canvas, f"{fps:.1f} FPS", (canvas.shape[1] - 110, canvas.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, FAINT, 1)
    return canvas
