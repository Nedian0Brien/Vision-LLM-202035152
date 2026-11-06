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
# 아스팔트 바탕에 노면 표시 색을 씀
PANEL = (47, 45, 43)
PAINT = (228, 235, 236)
CENTER_LINE = (24, 197, 245)
PROHIBIT = (60, 72, 229)
MUTED = (150, 147, 144)
BUTTON = (68, 65, 62)
# 금지와 정지는 규제 표지의 적색, 나머지는 주의 표지의 황색으로 그림
ACTION_COLORS = {"STOP": PROHIBIT, "KEEP_OUT": PROHIBIT, "NO_OVERTAKING": PROHIBIT}

# 프레임마다 PIL로 글자를 그리면 한 프레임에 25ms가 걸려 YOLO보다 오래 걸렸음(2026-11-03 Mac 측정)
FAST_LABEL = "8배속"
LABELS = {}
for name in list(CLASS_NAMES.values()) + [FAST_LABEL]:
    image = Image.new("RGB", (int(MEDIUM_FONT.getlength(name)) + 12, 28), PANEL)
    ImageDraw.Draw(image).text((6, 2), name, font=MEDIUM_FONT, fill=CENTER_LINE)
    LABELS[name] = np.array(image)

BUTTON_HEIGHT = 32
BUTTON_GAP = 8
# 버튼 두 줄 아래부터 지시문을 그림
RULES_TOP = 2 * (BUTTON_HEIGHT + BUTTON_GAP) + 12
RULE_TEXT_LINES = 2
MAX_RULES = 2
PIPELINE_TOP = 316
SMALL_THUMB = (72, 41)
LARGE_THUMB = (112, 63)
# 첫 칸은 이미지 3장을 받는 객관식 호출이라 넓게 잡음
CALLS = [("choice", "객관식", 24, ["frame", "signs", "road"]), ("light", "신호 색", 262, ["lights"]), ("pedestrian", "보행자", 382, ["people"]), ("crosswalk", "횡단보도", 502, ["road"])]


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


def lane(pen, x1, y1, x2, y2, active):
    """두 모듈을 잇는 차선을 지금 처리 중이면 황색 실선, 아니면 흰 점선으로 그리는 함수"""
    color = CENTER_LINE if active else MUTED
    steps = max(abs(x2 - x1), abs(y2 - y1)) // 12
    for step in range(steps):
        start = (x1 + (x2 - x1) * step // steps, y1 + (y2 - y1) * step // steps)
        finish = (x1 + (x2 - x1) * (step + (1 if active else 0.6)) / steps, y1 + (y2 - y1) * (step + (1 if active else 0.6)) / steps)
        pen.line((start, finish), fill=color, width=3 if active else 2)
    if y1 == y2:
        pen.polygon(((x2, y2 - 5), (x2 + 8, y2), (x2, y2 + 5)), fill=color)
    else:
        pen.polygon(((x2 - 5, y2), (x2 + 5, y2), (x2, y2 + 8)), fill=color)


def thumbnail(image, pen, picture, x, y, size):
    """모델이 받은 이미지를 작게 그리고 없으면 빈 칸을 그리는 함수"""
    if picture is None:
        pen.rectangle((x, y, x + size[0] - 1, y + size[1] - 1), fill=BUTTON)
    else:
        # 프레임이 BGR이고 패널도 BGR 순서로 그리므로 색을 바꾸지 않음
        image.paste(Image.fromarray(cv2.resize(picture, size, interpolation=cv2.INTER_AREA)), (x, y))


def module_row(pen, y, name, value, seconds, active, color=PAINT):
    """모듈 이름, 출력값, 걸린 시간을 한 줄로 그리는 함수"""
    pen.text((48, y), name, font=SMALL_FONT, fill=CENTER_LINE if active else MUTED)
    pen.text((160, y - 2), value, font=MEDIUM_FONT, fill=color)
    if seconds is not None:
        pen.text((616, y), seconds, font=SMALL_FONT, fill=MUTED, anchor="ra")


def draw_pipeline(image, pen, status, rules, law_name):
    """입력부터 출력까지 모듈마다 입력 이미지, 처리 상태, 출력값을 그리는 함수"""
    stage, answers, boxes, images = status["stage"], status["answers"], status["boxes"], status["images"]
    y = PIPELINE_TOP
    pen.line((24, y - 8, 616, y - 8), fill=BUTTON, width=1)
    thumbnail(image, pen, images.get("frame"), 24, y, SMALL_THUMB)
    lane(pen, 102, y + 20, 128, y + 20, False)
    pen.text((146, y), "YOLO11n", font=SMALL_FONT, fill=MUTED)
    pen.text((146, y + 18), f"박스 {boxes.get('all', 0)}개", font=MEDIUM_FONT, fill=PAINT)
    pen.text((290, y + 20), f"신호등 {boxes.get('lights', 0)}   정지 표지 {boxes.get('stops', 0)}   사람 {boxes.get('people', 0)}", font=SMALL_FONT, fill=MUTED)
    y += 50
    # YOLO 결과가 호출 4개로 갈라져 내려감
    pen.line((30, y - 6, 510, y - 6), fill=MUTED, width=1)
    for call, name, x, sources in CALLS:
        active = stage == call
        lane(pen, x + 6, y - 6, x + 6, y + 2, active)
        for index, source in enumerate(sources):
            if len(sources) == 3:
                thumbnail(image, pen, images.get(source), x + index * 74, y + 11, SMALL_THUMB)
            else:
                thumbnail(image, pen, images.get(source), x, y + 11, LARGE_THUMB)
        pen.text((x, y + 78), f"Gemma {name}", font=SMALL_FONT, fill=CENTER_LINE if active else MUTED)
        answer = answers.get(call)
        pen.text((x, y + 97), answer[0] if answer else "호출 없음", font=MEDIUM_FONT, fill=PAINT if answer else MUTED)
        if answer:
            pen.text((x, y + 121), f"{answer[1]:.2f}초", font=SMALL_FONT, fill=MUTED)
    # 보행자 답이 yes일 때만 횡단보도를 물음
    lane(pen, 492, y + 42, 497, y + 42, stage == "crosswalk")
    y += 148
    found = status["found"]
    # 호출 4개의 답이 상황 합치기로 모임
    pen.line((30, y - 6, 510, y - 6), fill=MUTED, width=1)
    lane(pen, 30, y - 6, 30, y + 4, False)
    module_row(pen, y + 6, "상황 합치기", "   ".join(found) if found else "해당 상황 없음", None, False, PAINT if found else MUTED)
    y += 28
    lane(pen, 30, y + 2, 30, y + 10, False)
    module_row(pen, y + 12, "조문 찾기", f"{law_name}  {rules[0]['law']}" if rules else law_name, None, False)
    y += 28
    action, instruction = answers.get("action"), answers.get("instruction")
    lane(pen, 30, y + 8, 30, y + 16, stage in ("action", "instruction"))
    module_row(pen, y + 18, "Gemma 행동", rules[0]["action"] if rules else "호출 없음", f"{action[1]:.2f}초" if action else None, stage == "action",
               ACTION_COLORS.get(rules[0]["action"], CENTER_LINE) if rules else MUTED)
    y += 26
    instruction_text = rules[0]["instruction"] if rules else "호출 없음"
    module_row(pen, y + 18, "Gemma 지시문", instruction_text, f"{instruction[1]:.2f}초" if instruction else None, stage == "instruction", PAINT if rules else MUTED)
    y += 26
    lane(pen, 30, y + 18, 30, y + 26, False)
    module_row(pen, y + 28, "VLA 출력", f"규칙 {len(rules)}개, {status['outputs']}줄", None, False)


def draw_panel(height, frame_width, buttons, selected, rules, status, law_name):
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
            pen.text((24, y), line, font=LARGE_FONT, fill=ACTION_COLORS.get(rule["action"], CENTER_LINE))
            y += 34
        pen.text((24, y), rule["law"], font=SMALL_FONT, fill=PAINT)
        y += 21
        for line in wrap(rule["text"], SMALL_FONT, width)[:RULE_TEXT_LINES]:
            pen.text((24, y), line, font=SMALL_FONT, fill=MUTED)
            y += 20
        y += 6
    draw_pipeline(image, pen, status, rules, law_name)
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
            cv2.rectangle(frame, (x1, y1), (x2, y2), CENTER_LINE, 2)
            label = LABELS[CLASS_NAMES[int(cls)]]
            height, width = label.shape[:2]
            top = max(y1 - height, 0)
            width = min(width, frame.shape[1] - x1)
            frame[top:top + height, x1:x1 + width] = label[:, :width]
    canvas = cv2.copyMakeBorder(frame, 0, 0, 0, PANEL_WIDTH, cv2.BORDER_CONSTANT, value=PANEL)
    canvas[:, frame.shape[1]:] = panel_image
    cv2.putText(canvas, f"{fps:.1f} FPS", (canvas.shape[1] - 110, canvas.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, MUTED, 1)
    return canvas
