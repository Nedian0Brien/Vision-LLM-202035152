"""탐지 박스, 버튼, 지시문과 근거 조항을 화면에 그리는 모듈"""

import os
import time

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
TINY_FONT = ImageFont.truetype(FONT_PATH, 13)
LARGE_FONT = ImageFont.truetype(FONT_PATH, 28)

# 1280 픽셀 영상 옆에 붙여 전체 폭이 1920 픽셀 화면에 맞게 함
PANEL_WIDTH = 640

# 프레임이 BGR 순서라 PIL에 넘기는 색도 BGR 순서로 적음
PANEL = (34, 31, 28)
PAINT = (232, 239, 241)
SIGNAL = (30, 190, 255)
PROHIBIT = (58, 66, 235)
MUTED = (150, 147, 144)
BUTTON = (68, 65, 62)
# 도로 전광판처럼 어두운 판에 호박색 글자가 켜짐
BOARD = (14, 12, 10)
UNLIT = (70, 66, 62)
# 금지와 정지는 규제 표지의 적색, 나머지는 주의 표지의 황색으로 그림
ACTION_COLORS = {"STOP": PROHIBIT, "KEEP_OUT": PROHIBIT, "NO_OVERTAKING": PROHIBIT}

# 프레임마다 PIL로 글자를 그리면 한 프레임에 25ms가 걸려 YOLO보다 오래 걸렸음(2026-11-03 Mac 측정)
FAST_LABEL = "8배속"
LABELS = {}
for name in list(CLASS_NAMES.values()) + [FAST_LABEL]:
    image = Image.new("RGB", (int(MEDIUM_FONT.getlength(name)) + 12, 28), PANEL)
    ImageDraw.Draw(image).text((6, 2), name, font=MEDIUM_FONT, fill=SIGNAL)
    LABELS[name] = np.array(image)

BUTTON_HEIGHT = 32
BUTTON_GAP = 8
# 버튼 두 줄 아래부터 지시문을 그림
RULES_TOP = 2 * (BUTTON_HEIGHT + BUTTON_GAP) + 12
RULE_TEXT_LINES = 2
MAX_RULES = 2
TILES = ["frame", "signs", "road", "lights", "people"]
TILE_SIZE = (118, 66)
TILE_TOP = 306
# 호출마다 입력으로 받는 이미지 칸
CALL_TILES = {"choice": [0, 1, 2], "light": [3], "pedestrian": [4], "crosswalk": [2]}
CALLS = [("choice", "상황 15개 객관식"), ("light", "신호 색"), ("pedestrian", "보행자"), ("crosswalk", "횡단보도"), ("action", "행동"), ("instruction", "지시문")]
BOARD_TOP = 424
BOARD_COLUMNS = 5


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


def tile_left(index):
    """이미지 칸의 왼쪽 좌표를 구하는 함수"""
    return 12 + index * (TILE_SIZE[0] + 6)


def draw_recognition(image, pen, status):
    """모델이 보는 이미지와 호출 진행을 그리는 함수"""
    stage, answers, images = status["stage"], status["answers"], status["images"]
    lit_tiles = CALL_TILES.get(stage, [])
    for index, name in enumerate(TILES):
        left = tile_left(index)
        picture = images.get(name)
        if picture is None:
            pen.rectangle((left, TILE_TOP, left + TILE_SIZE[0] - 1, TILE_TOP + TILE_SIZE[1] - 1), fill=BOARD)
            continue
        small = cv2.resize(picture, TILE_SIZE, interpolation=cv2.INTER_AREA)
        # 지금 호출의 입력만 밝게 두어 모델이 어디를 보는지 나타냄
        if stage in CALL_TILES and index not in lit_tiles:
            small = (small * 0.3).astype(np.uint8)
        image.paste(Image.fromarray(small), (left, TILE_TOP))
        if index in lit_tiles:
            pen.rectangle((left - 2, TILE_TOP - 2, left + TILE_SIZE[0] + 1, TILE_TOP + TILE_SIZE[1] + 1), outline=SIGNAL, width=2)
    # 호출 진행 막대: 칸 너비가 걸린 시간에 비례함
    times = [max(answers[call][1], 0.3) if call in answers else 0.3 for call, name in CALLS]
    x = 12
    top = TILE_TOP + TILE_SIZE[1] + 8
    for (call, name), seconds in zip(CALLS, times):
        width = int(616 * seconds / sum(times))
        active = stage == call
        answer = answers.get(call)
        pen.rectangle((x, top, x + width - 3, top + 3), fill=SIGNAL if active else (PAINT if answer else BUTTON))
        pen.text((x, top + 6), name, font=TINY_FONT, fill=SIGNAL if active else (PAINT if answer else UNLIT))
        if answer:
            pen.text((x, top + 22), wrap(f"{answer[0]}  {answer[1]:.2f}초", TINY_FONT, width - 6)[0], font=TINY_FONT, fill=MUTED)
        x += width


def draw_reading(pen, status, rule):
    """조문을 읽어 행동 코드가 나오는 순간을 크게 그리는 함수"""
    answers = status["answers"]
    pen.text((12, TILE_TOP - 2), rule["law"], font=MEDIUM_FONT, fill=SIGNAL)
    y = TILE_TOP + 26
    for line in wrap(rule["text"], SMALL_FONT, 420)[:4]:
        pen.text((12, y), line, font=SMALL_FONT, fill=PAINT)
        y += 21
    middle = TILE_TOP + 54
    pen.line((442, middle, 462, middle), fill=SIGNAL, width=2)
    pen.polygon(((462, middle - 6), (472, middle), (462, middle + 6)), fill=SIGNAL)
    color = ACTION_COLORS.get(rule["action"], SIGNAL)
    pen.rounded_rectangle((478, middle - 26, 628, middle + 26), 4, fill=color)
    pen.text((553, middle), rule["action"], font=SMALL_FONT, fill=PAINT if color == PROHIBIT else BOARD, anchor="mm")
    seconds = answers["action"][1] + answers["instruction"][1] if "action" in answers and "instruction" in answers else None
    if seconds is not None:
        pen.text((553, middle + 32), f"Gemma {seconds:.2f}초", font=TINY_FONT, fill=MUTED, anchor="ma")


def draw_pipeline(image, pen, status, rules, board, reading):
    """장면, 상황 전광판, 내보내는 행동을 그리는 함수"""
    # 새 규칙이 나오면 조문 읽기 장면으로 바꾸고 평소에는 인식 장면을 보여 줌
    if reading is None:
        draw_recognition(image, pen, status)
    else:
        draw_reading(pen, status, reading)
    # 상황 전광판: 켜진 상황은 유지 시간이 지나며 어두워짐
    pen.rounded_rectangle((12, BOARD_TOP, 628, BOARD_TOP + 112), 6, fill=BOARD)
    width = (616 - 12) // BOARD_COLUMNS
    for index, (name, glow, has_provision) in enumerate(board):
        x = 18 + (index % BOARD_COLUMNS) * width
        y = BOARD_TOP + 6 + (index // BOARD_COLUMNS) * 34
        if glow > 0 and has_provision:
            color = tuple(int(u + (s - u) * (0.35 + 0.65 * glow)) for u, s in zip(UNLIT, SIGNAL))
            pen.rectangle((x, y, x + width - 5, y + 30), fill=tuple(c // 5 for c in color), outline=color)
            pen.text((x + (width - 5) // 2, y + 15), name, font=TINY_FONT, fill=color, anchor="mm")
        elif has_provision:
            pen.text((x + (width - 5) // 2, y + 15), name, font=TINY_FONT, fill=UNLIT, anchor="mm")
        else:
            # 고른 법령에 조문이 없는 상황은 취소선으로 나타냄
            pen.text((x + (width - 5) // 2, y + 15), name, font=TINY_FONT, fill=BUTTON, anchor="mm")
            pen.line((x + 12, y + 15, x + width - 17, y + 15), fill=BUTTON)
    # 내보내는 값: 행동 코드를 표지처럼 찍음
    y = BOARD_TOP + 122
    for rule in rules[:2]:
        color = ACTION_COLORS.get(rule["action"], SIGNAL)
        pen.rounded_rectangle((12, y, 212, y + 42), 4, fill=color)
        pen.text((112, y + 21), rule["action"], font=MEDIUM_FONT, fill=PAINT if color == PROHIBIT else BOARD, anchor="mm")
        pen.text((224, y + 21), rule["law"], font=MEDIUM_FONT, fill=PAINT, anchor="lm")
        y += 48
    pen.text((12, BOARD_TOP + 222), f"vla-output.jsonl  {status['outputs']}줄", font=TINY_FONT, fill=MUTED)


def draw_panel(height, frame_width, buttons, selected, rules, status, board, reading):
    """버튼, 지시문, 근거 조항, 읽은 조문, 모듈 상태를 그린 패널 이미지를 만드는 함수"""
    image = Image.new("RGB", (PANEL_WIDTH, height), PANEL)
    pen = ImageDraw.Draw(image)
    # 버튼 사각형은 클릭 판정에 쓰는 창 좌표라 패널 왼쪽 끝만큼 빼서 그림
    for group, index, name, (x1, y1, x2, y2) in buttons:
        chosen = selected[group] == index
        pen.rectangle((x1 - frame_width, y1, x2 - frame_width, y2), fill=PAINT if chosen else BUTTON)
        pen.text(((x1 + x2) // 2 - frame_width, (y1 + y2) // 2), name, font=SMALL_FONT, fill=PANEL if chosen else PAINT, anchor="mm")
    width = PANEL_WIDTH - 48
    y = RULES_TOP
    for rule in rules[:MAX_RULES]:
        for line in wrap(rule["instruction"], LARGE_FONT, width):
            pen.text((24, y), line, font=LARGE_FONT, fill=ACTION_COLORS.get(rule["action"], SIGNAL))
            y += 34
        pen.text((24, y), rule["law"], font=SMALL_FONT, fill=PAINT)
        y += 21
        for line in wrap(rule["text"], SMALL_FONT, width)[:RULE_TEXT_LINES]:
            pen.text((24, y), line, font=SMALL_FONT, fill=MUTED)
            y += 20
        y += 6
    draw_pipeline(image, pen, status, rules, board, reading)
    return np.array(image)


def draw(frame, result, panel_image, fps, stage, reading):
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
            cv2.rectangle(frame, (x1, y1), (x2, y2), SIGNAL, 2)
            label = LABELS[CLASS_NAMES[int(cls)]]
            height, width = label.shape[:2]
            top = max(y1 - height, 0)
            width = min(width, frame.shape[1] - x1)
            frame[top:top + height, x1:x1 + width] = label[:, :width]
    canvas = cv2.copyMakeBorder(frame, 0, 0, 0, PANEL_WIDTH, cv2.BORDER_CONSTANT, value=PANEL)
    canvas[:, frame.shape[1]:] = panel_image
    # 지금 호출의 입력 위로 스캔선을 움직여 처리 중임을 나타냄
    offset = int(time.perf_counter() * 60) % TILE_SIZE[1]
    for index in [] if reading else CALL_TILES.get(stage, []):
        left = frame.shape[1] + tile_left(index)
        cv2.line(canvas, (left, TILE_TOP + offset), (left + TILE_SIZE[0] - 1, TILE_TOP + offset), SIGNAL, 2)
    cv2.putText(canvas, f"{fps:.1f} FPS", (canvas.shape[1] - 110, canvas.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, MUTED, 1)
    return canvas
