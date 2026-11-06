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
PANEL = (30, 28, 27)
PAINT = (232, 239, 241)
CENTER_LINE = (24, 197, 245)
PROHIBIT = (74, 84, 240)
MUTED = (150, 147, 144)
BUTTON = (68, 65, 62)
ARROW = (150, 150, 146)
# 금지와 정지는 규제 표지의 적색, 나머지는 주의 표지의 황색으로 그림
ACTION_COLORS = {"STOP": PROHIBIT, "KEEP_OUT": PROHIBIT, "NO_OVERTAKING": PROHIBIT}
# 구조도와 같은 색으로 모델, 코드, 산출물, 출력을 구분함. 값은 (바탕, 테두리, 글자)
MODEL = ((137, 52, 60), (236, 169, 175), (254, 237, 238))
CODE = ((65, 80, 8), (165, 202, 93), (238, 245, 225))
DATA = ((65, 68, 68), (169, 178, 180), (232, 239, 241))
OUTPUT = ((6, 56, 99), (39, 159, 239), (218, 238, 250))
IDLE = ((42, 40, 39), (78, 76, 74), (120, 118, 116))

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
# 첫 칸은 이미지 3장을 받는 객관식 호출이라 넓게 잡음. 값은 (호출, 이름, 왼쪽, 오른쪽, 입력 이미지)
CALLS = [("choice", "객관식", 12, 238, ["frame", "signs", "road"]), ("light", "신호 색", 246, 368, ["lights"]),
         ("pedestrian", "보행자", 376, 498, ["people"]), ("crosswalk", "횡단보도", 506, 628, ["road"])]


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


def arrow(pen, x1, y1, x2, y2, active=False):
    """모듈을 잇는 화살표를 지금 처리 중이면 황색으로 그리는 함수"""
    color = CENTER_LINE if active else ARROW
    pen.line((x1, y1, x2, y2), fill=color, width=2 if active else 1)
    if y1 == y2:
        pen.polygon(((x2 - 6, y2 - 4), (x2, y2), (x2 - 6, y2 + 4)), fill=color)
    else:
        pen.polygon(((x2 - 4, y2 - 6), (x2 + 4, y2 - 6), (x2, y2)), fill=color)


def module_box(pen, x1, y1, x2, y2, kind, lines, active=False, pill=False):
    """구조도의 상자 하나를 그리는 함수"""
    fill, edge, text = kind
    radius = (y2 - y1) // 2 if pill else 4
    pen.rounded_rectangle((x1, y1, x2, y2), radius, fill=fill, outline=CENTER_LINE if active else edge, width=3 if active else 1)
    line_height = 19
    y = (y1 + y2) // 2 - line_height * (len(lines) - 1) // 2
    for line in lines:
        pen.text(((x1 + x2) // 2, y), wrap(line, SMALL_FONT, x2 - x1 - 14)[0], font=SMALL_FONT, fill=text, anchor="mm")
        y += line_height


def thumbnail(image, pen, picture, x1, y1, x2, y2):
    """모델이 받은 이미지를 작게 그리고 없으면 빈 칸을 그리는 함수"""
    if picture is None:
        pen.rectangle((x1, y1, x2 - 1, y2 - 1), fill=IDLE[0], outline=IDLE[1])
    else:
        # 프레임이 BGR이고 패널도 BGR 순서로 그리므로 색을 바꾸지 않음
        image.paste(Image.fromarray(cv2.resize(picture, (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA)), (x1, y1))


def draw_pipeline(image, pen, status, rules, law_name):
    """입력부터 출력까지 구조도와 같은 상자와 화살표로 모듈의 입력, 상태, 출력을 그리는 함수"""
    stage, answers, boxes, images = status["stage"], status["answers"], status["boxes"], status["images"]
    thumbnail(image, pen, images.get("frame"), 12, 304, 84, 345)
    arrow(pen, 84, 324, 104, 324)
    module_box(pen, 104, 304, 236, 345, MODEL, ["YOLO11n", f"박스 {boxes.get('all', 0)}개"])
    arrow(pen, 236, 324, 256, 324)
    module_box(pen, 256, 304, 628, 345, DATA, [f"신호등 {boxes.get('lights', 0)}   정지 표지 {boxes.get('stops', 0)}   사람 {boxes.get('people', 0)}"], pill=True)
    # YOLO 결과가 호출 4개로 갈라짐
    pen.line((170, 345, 170, 352), fill=ARROW)
    pen.line((125, 352, 567, 352), fill=ARROW)
    for call, name, x1, x2, sources in CALLS:
        active = stage == call
        middle = (x1 + x2) // 2
        arrow(pen, middle, 352, middle, 359, active)
        width = (x2 - x1 - 2 * (len(sources) - 1)) // len(sources)
        for index, source in enumerate(sources):
            left = x1 + index * (width + 2)
            thumbnail(image, pen, images.get(source), left, 359, left + width, 402)
        arrow(pen, middle, 402, middle, 412, active)
        answer = answers.get(call)
        if answer:
            module_box(pen, x1, 412, x2, 458, MODEL, [f"Gemma {name}", f"{answer[0]}  {answer[1]:.2f}초"], active)
        else:
            module_box(pen, x1, 412, x2, 458, MODEL if active else IDLE, [f"Gemma {name}", "처리 중" if active else "호출 없음"], active)
        pen.line((middle, 458, middle, 466), fill=ARROW)
    # 호출의 답이 하나로 모임
    pen.line((125, 466, 567, 466), fill=ARROW)
    arrow(pen, 80, 466, 80, 474)
    pen.line((80, 466, 125, 466), fill=ARROW)
    found = status["found"]
    module_box(pen, 12, 474, 196, 512, CODE, ["답 합치기, 3.5초 유지"])
    arrow(pen, 196, 493, 216, 493)
    module_box(pen, 216, 474, 628, 512, DATA, ["   ".join(found) if found else "해당 상황 없음"], pill=True)
    arrow(pen, 80, 512, 80, 522)
    rule = rules[0] if rules else None
    module_box(pen, 12, 522, 196, 560, CODE, ["조문 찾기"])
    arrow(pen, 196, 541, 216, 541)
    module_box(pen, 216, 522, 628, 560, DATA, [f"index  {law_name}  {rule['law']}" if rule else f"index  {law_name}  조문 없음"], pill=True)
    reading = stage in ("action", "instruction")
    pen.line((80, 560, 80, 568), fill=CENTER_LINE if reading else ARROW)
    pen.line((80, 568, 410, 568), fill=CENTER_LINE if reading else ARROW)
    arrow(pen, 80, 568, 80, 576, stage == "action")
    arrow(pen, 410, 568, 410, 576, stage == "instruction")
    action, instruction = answers.get("action"), answers.get("instruction")
    module_box(pen, 12, 576, 196, 622, MODEL if rule else IDLE, ["Gemma 행동", (rule["action"] + (f"  {action[1]:.2f}초" if action else "")) if rule else "호출 없음"], stage == "action")
    module_box(pen, 204, 576, 628, 622, MODEL if rule else IDLE, ["Gemma 지시문", (rule["instruction"] if rule else "호출 없음")], stage == "instruction")
    arrow(pen, 80, 622, 80, 632)
    arrow(pen, 410, 622, 410, 632)
    module_box(pen, 12, 632, 628, 670, OUTPUT, [f"vla-output.jsonl  규칙 {len(rules)}개  {status['outputs']}줄"], pill=True)


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
