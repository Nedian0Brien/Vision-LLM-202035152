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

# 16:9 모니터에 맞도록 왼쪽에 영상과 조문, 오른쪽에 패널을 두어 전체를 1600×900으로 맞춤(2026-11-07 사용자 지시)
SCREEN_HEIGHT = 900
PANEL_WIDTH = 640
NODE_HEIGHT = 360
NODE_COLUMNS = [20, 340, 660]
VIDEO_WIDTH = 960
VIDEO_HEIGHT = 540
VIDEO_SCALE = 0.75
LOAD_BAR_WIDTH = 720
WRAPPED = {}
SPINNER_POINTS = [(0, -7), (5, -5), (7, 0), (5, 5), (0, 7), (-5, 5), (-7, 0), (-5, -5)]

# 프레임이 BGR 순서라 PIL에 넘기는 색도 BGR 순서로 적음
PANEL = (39, 35, 32)
INK = (236, 234, 232)
SOFT = (166, 160, 154)
FAINT = (103, 97, 92)
ACCENT = (132, 184, 214)
LINE = (71, 66, 62)

# 프레임마다 PIL로 글자를 그리면 한 프레임에 25ms가 걸려 YOLO보다 오래 걸렸음(2026-11-03 Mac 측정)
LABELS = {}
RECORDING_LABEL = "녹음 중"
for name in list(CLASS_NAMES.values()) + [RECORDING_LABEL]:
    image = Image.new("RGB", (int(SMALL_FONT.getlength(name)) + 12, 24), PANEL)
    ImageDraw.Draw(image).text((6, 2), name, font=SMALL_FONT, fill=INK)
    LABELS[name] = np.array(image)

TAB_ROWS = [16, 54]
HERO_TOP = 104
STEPS = ["보기", "인식", "조문", "읽기", "출력"]
STEP_X = [60, 190, 320, 450, 580]
RAIL_Y = 440
# 지시문 목록과 단계 표시 사이 여백
RAIL_GAP = 18
CARD_TOP = 508
# 법령 탭 줄의 오른쪽 끝에 놓는 버튼의 창 좌표
MIC_BOX = (VIDEO_WIDTH + 540, TAB_ROWS[1], VIDEO_WIDTH + PANEL_WIDTH - 28, TAB_ROWS[1] + 26)
# 화면 배열이 BGR 순서라 빨강으로 보이도록 파랑 자리를 작게 둠
RECORDING = (60, 60, 225)
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
    # 같은 문장을 다시 그릴 때 글자마다 폭을 재지 않게 함(Mac 프로파일에서 그리기 시간의 약 40%)
    key = (text, font.size, width)
    if key in WRAPPED:
        return WRAPPED[key]
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
    WRAPPED[key] = lines + [line]
    return WRAPPED[key]


def draw_rail(pen, current):
    """처리 단계를 가로 한 줄로 그리고 지금 단계를 표시하는 함수"""
    pen.line((STEP_X[0], RAIL_Y, STEP_X[-1], RAIL_Y), fill=LINE, width=2)
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
        pen.text((28, CARD_TOP), "조문 요약에서 행동을 가져왔습니다" if rule["text"] else "Gemma가 조문 없이 행동을 골랐습니다", font=SMALL_FONT, fill=INK)
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
    if voice["recording"]:
        # 0.5초마다 점을 껐다 켜서 녹음 중임을 보임
        if int(voice["elapsed"] * 2) % 2 == 0:
            pen.ellipse((28, y + 8, 44, y + 24), fill=RECORDING)
        pen.text((56, y), "듣고 있습니다", font=RESULT_FONT, fill=INK)
        pen.text((56, y + 38), "다시 누르면 멈추고 질문합니다", font=SMALL_FONT, fill=SOFT)
        done = min(voice["elapsed"] / voice["limit"], 1)
        pen.rectangle((56, y + 70, 456, y + 74), fill=LINE)
        pen.rectangle((56, y + 70, 56 + int(400 * done), y + 74), fill=RECORDING)
        pen.text((468, y + 62), f"{int(voice['elapsed'])}초 / {voice['limit']}초", font=SMALL_FONT, fill=SOFT)
        return
    if voice["listening"]:
        pen.text((28, y), "질문을 글로 바꾸고 있습니다" if voice["transcribing"] else "음성 출력이 끝나기를 기다립니다", font=RESULT_FONT, fill=ACCENT)
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
    pen.text((MIC_BOX[0] - frame_width, MIC_BOX[1]), "멈추기" if voice["recording"] else "음성 질문", font=SMALL_FONT, fill=RECORDING if voice["recording"] else ACCENT if voice["listening"] else SOFT)
    y = HERO_TOP
    if rules:
        # 한 지시문이 더 중요해 보이지 않도록 모두 같은 크기로 최신 순으로 그림(2026-11-07 사용자 지시)
        newest = rules[::-1]
        for index, rule in enumerate(newest):
            lines = wrap(rule["instruction"], RESULT_FONT, 584)[:2]
            # 마지막 규칙이 아니면 외 N개를 쓸 한 줄을 남김
            bottom = RAIL_Y - RAIL_GAP - (0 if index == len(newest) - 1 else 22)
            if y + len(lines) * 34 + 30 > bottom:
                pen.text((28, y), f"외 {len(newest) - index}개", font=SMALL_FONT, fill=SOFT)
                break
            for line in lines:
                pen.text((28, y), line, font=RESULT_FONT, fill=INK)
                y += 34
            pen.text((28, y + 2), rule["law"], font=SMALL_FONT, fill=ACCENT)
            y += 34
        if len(rules) == 1:
            for line in wrap(rules[0]["text"], SMALL_FONT, 584)[:2]:
                pen.text((28, y), line, font=SMALL_FONT, fill=SOFT)
                y += 22
    else:
        pen.text((28, y), "적용할 규칙 없음", font=RESULT_FONT, fill=FAINT)
    # 아래 카드는 음성 질문, 새 규칙의 조문, 지금 호출 순서로 하나만 그림
    if voice["listening"] or voice["thinking"] or voice["answer"]:
        draw_rail(pen, 1 if voice["listening"] else 3 if voice["thinking"] else 4)
        draw_answer_card(pen, voice)
    elif reading is not None:
        draw_rail(pen, 2 + phase)
        draw_law_card(pen, status, reading, phase)
    else:
        draw_rail(pen, 1)
        # 영상을 바꾼 직후에는 이전 영상 판단이 끝나야 새 영상을 판단함
        if status.get("stale"):
            pen.text((28, CARD_TOP), "이전 영상 판단을 마치는 중입니다", font=RESULT_FONT, fill=SOFT)
            pen.text((28, CARD_TOP + 40), "끝나면 새 영상으로 판단을 시작합니다", font=SMALL_FONT, fill=FAINT)
        elif scene in CALLS:
            draw_recognition_card(image, pen, status, scene)
    pen.text((28, 806), "VLA 출력", font=TINY_FONT, fill=FAINT)
    pen.text((100, 804), wrap("  ".join([rule["action"] for rule in rules]) if rules else "없음", SMALL_FONT, 500)[0], font=SMALL_FONT, fill=INK if rules else FAINT)
    pen.text((28, 868), voice["error"] or f"vla-output.jsonl  {status['outputs']}줄", font=TINY_FONT, fill=ACCENT if voice["error"] else FAINT)
    return np.array(image)


def draw_loading(height, steps, step, elapsed):
    """불러오는 단계와 진행 막대를 그린 시작 화면을 만드는 함수"""
    width = VIDEO_WIDTH + PANEL_WIDTH
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


def draw_nodes(nodes, spinner):
    """검색된 조문 노드를 이번 판단에서 찾은 것부터 영상 아래 세 칸에 그린 이미지를 만드는 함수"""
    image = Image.new("RGB", (VIDEO_WIDTH, NODE_HEIGHT), PANEL)
    pen = ImageDraw.Draw(image)
    pen.line((0, 0, VIDEO_WIDTH, 0), fill=LINE, width=1)
    pen.text((20, 14), "검색된 조문", font=SMALL_FONT, fill=SOFT)
    x = NODE_COLUMNS[0]
    column = 0
    y = 44
    drawn = []
    shown = []
    # 여러 상황이 같은 조문을 찾으면 한 번만 그림
    for node in [node for node in nodes if node["fresh"]] + [node for node in nodes if not node["fresh"]]:
        if (node["article"]["id"], node["instruction"]) not in [(other["article"]["id"], other["instruction"]) for other in shown]:
            shown.append(node)
    # 조문이 적으면 칸 높이를 나눠 조문 원문을 길게 보여 줌
    height = (NODE_HEIGHT - 52) // max(1, -(-len(shown) // len(NODE_COLUMNS)))
    for node in shown:
        article = node["article"]
        results = wrap(node["instruction"], SMALL_FONT, 284)[:2] if node["instruction"] else ["지시문 생성 중..."]
        lines = wrap(article["text"], TINY_FONT, 284)
        count = max(2, (height - 64 - 21 * len(results)) // 19)
        if len(lines) > count:
            lines = lines[:count - 1] + [lines[count - 1][:-1] + "…"]
        if y + 48 + 21 * len(results) + 19 * len(lines) > NODE_HEIGHT - 8:
            column += 1
            if column == len(NODE_COLUMNS):
                pen.text((VIDEO_WIDTH - 130, 14), "더 찾은 조문 있음", font=TINY_FONT, fill=SOFT)
                break
            x = NODE_COLUMNS[column]
            y = 44
        drawn.append(article["id"])
        pen.text((x, y), article["id"], font=MEDIUM_FONT, fill=ACCENT if node["fresh"] else SOFT)
        pen.text((x, y + 24), article["title"], font=SMALL_FONT, fill=SOFT if node["fresh"] else FAINT)
        y += 48
        if node["instruction"]:
            for line in results:
                pen.text((x, y), line, font=SMALL_FONT, fill=INK)
                y += 21
        else:
            draw_spinner(pen, x + 8, y + 10, spinner)
            pen.text((x + 24, y), results[0], font=SMALL_FONT, fill=ACCENT)
            y += 21
        for line in lines:
            pen.text((x, y), line, font=TINY_FONT, fill=FAINT)
            y += 19
        y += 16
    if not drawn:
        pen.text((x, y), "검색된 조문 없음", font=MEDIUM_FONT, fill=FAINT)
    return np.array(image)


def draw(frame, boxes, panel_image, nodes_image, fps, recording):
    """프레임에 박스와 클래스 이름을 그리고 조문을 아래에, 패널을 오른쪽에 붙이는 함수"""
    frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
    for box, cls, conf in boxes:
        if conf >= SHOW_CONF:
            x1, y1, x2, y2 = [int(value * VIDEO_SCALE) for value in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), ACCENT, 1)
            label = LABELS[CLASS_NAMES[int(cls)]]
            height, width = label.shape[:2]
            top = max(y1 - height, 0)
            width = min(width, frame.shape[1] - x1)
            frame[top:top + height, x1:x1 + width] = label[:, :width]
    # 패널을 보지 않아도 녹음 중임을 알 수 있게 영상 왼쪽 위에 표시함
    if recording:
        cv2.circle(frame, (24, 24), 8, RECORDING, -1)
        label = LABELS[RECORDING_LABEL]
        frame[12:12 + label.shape[0], 40:40 + label.shape[1]] = label
    canvas = cv2.copyMakeBorder(frame, 0, NODE_HEIGHT, 0, PANEL_WIDTH, cv2.BORDER_CONSTANT, value=PANEL)
    canvas[:, VIDEO_WIDTH:] = panel_image
    canvas[VIDEO_HEIGHT:, :VIDEO_WIDTH] = nodes_image
    cv2.putText(canvas, f"{fps:.1f} FPS", (canvas.shape[1] - 110, canvas.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, FAINT, 1)
    return canvas
