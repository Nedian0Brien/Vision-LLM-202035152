# Project03 코드 읽기 안내

`gemma.py`, `main.py`, `detect.py`, `hud.py` 네 파일을 순서대로 읽는 안내입니다. 함수별 상세 설명은 `EXPLAIN.md`에 있고, 여기서는 읽는 순서, 실제 값으로 따라가는 한 프레임, 함수 색인, 라이브러리 색인, 발표 질문을 다룹니다. `build_corpus.py`는 코퍼스를 만들 때 한 번만 쓰므로 마지막에 봅니다.

## 읽는 순서

각 단계의 질문에 코드를 보지 않고 답할 수 있으면 다음 단계로 넘어갑니다.

| 순서 | 볼 곳 | 볼 내용 | 스스로 답할 질문 |
|---|---|---|---|
| 1 | `README.md`의 구조 다이어그램 | 빌드 단계, 영상 루프, 판단 스레드 세 부분 | Sonnet과 Gemma는 각각 언제 무엇을 하는가 |
| 2 | `detect.py` 전체 | `CLASS_NAMES`, `class_strip`, `road_crop` | 신호등 크롭을 왜 세로 400으로 키우는가. 박스가 없으면 무엇을 돌려주는가 |
| 3 | `gemma.py` 위쪽 상수 | `SITUATIONS`, `CROP_SITUATIONS`, 프롬프트들, `ACTIONS` | 객관식에 들어가는 상황과 크롭 질문으로 묻는 상황은 왜 나뉘었는가 |
| 4 | `gemma.py`의 `load_gemma`, `load_law` | 모델과 인덱스 불러오기 | `n_threads`를 왜 2로 두는가. 조문 없음 모드의 법령 사전은 어떻게 생겼는가 |
| 5 | `image_part`, `first_token_bias`, `ask_word`, `answer_odds` | 이미지를 메시지로 바꾸고 답을 낱말로 제한하기 | 답이 왜 `hydrant`가 아니라 `hyd`로 돌아오는가. 확률은 어디서 읽는가 |
| 6 | `parse_situations`, `recognize` | 판단 한 번의 전체 순서 | 맑은 날 호출은 몇 번인가. 제27조 ⑦은 어떤 조건에서 묻는가 |
| 7 | `read_provision` | 행동 고르기와 지시문 쓰기 | 호출이 왜 두 번인가. 조문 없음 모드에서는 어디가 달라지는가 |
| 8 | `watch` | 8초 유지와 지시문 재사용 | 같은 상황을 다시 봐도 왜 Gemma를 다시 부르지 않는가 |
| 9 | `main.py` 위에서 아래로 | 영상 열기, YOLO, 입력 칸, 패널 갱신, 출력 기록 | 영상 루프가 Gemma를 기다리지 않는 이유는 무엇인가 |
| 10 | `hud.py`의 `draw_panel`, `draw` | 패널 그리기 | 패널을 왜 매 프레임 다시 그리지 않는가 |
| 11 | `build_corpus.py`, `tests/test_gemma.py` | 법령 수집과 동작 시험 | 인덱스는 코퍼스에서 어떻게 만들어지는가 |

## 실행 순서를 코드로 말하기

1. `main.py`가 영상을 한 프레임 읽어 1280×720으로 바꿉니다.
2. 프레임 복사본을 `shared["frame"]`에 넣습니다. 별도 스레드의 `detect_boxes`가 YOLO11n으로 박스를 찾아 프레임과 박스를 `shared["input"]`에 넣습니다. 화면은 10프레임 늦추고, `boxes_at`이 앞뒤 탐지 결과로 표시할 프레임의 박스 위치를 계산합니다.
3. 별도 스레드의 `watch`가 가장 새로운 입력을 꺼내 `recognize`를 부릅니다. 영상 루프는 이 호출을 기다리지 않습니다.
4. `recognize`가 객관식과 크롭 질문의 답을 합쳐 상황 목록을 돌려줍니다.
5. `watch`가 상황마다 본 시각을 적고, 8초 안에 본 상황 중 고른 법령에 조문이 있는 것만 남깁니다.
6. 처음 만난 법령과 상황이면 `read_provision`이 행동과 지시문을 만들고, 다음부터는 저장한 결과를 씁니다.
7. `main.py`가 규칙 목록의 변화를 보고 패널을 다시 그리고 `vla-output.jsonl`에 한 줄을 씁니다.

## 한 프레임 따라가기

`school-zone-busan.mp4`의 6초 프레임을 실제로 넣고 호출마다 기록한 값입니다. 노란 어린이 보호구역 표지, 빨간 포장, 신호 없는 횡단보도가 보이고 보행자가 횡단보도 옆에 있습니다.

| 단계 | 함수 | 입력 | 출력 | 시간 |
|---|---|---|---|---|
| 탐지 | `model.predict` | 프레임 1280×720 | 사람 0.65, 승용차 0.56, 승용차 0.54, 트럭 0.47, 승용차 0.47, 사람 0.27 | 평소 약 0.04초 |
| 크롭 | `class_strip`, `road_crop` | 프레임과 박스 | 신호등, 소화전 박스가 없어 `None`, 표지 칸은 프레임 전체, 노면 크롭 896×266 | 없음 |
| 객관식 | `recognize` | 이미지 3장과 상황 15개, 입력 1074토큰 | `어린이 보호구역` | 1.38초 |
| 보행자 | `ask_word("pedestrian")` | 사람 크롭, 입력 177토큰 | `yes`, 확률 100% | 0.24초 |
| 횡단보도 | `ask_word("crosswalk")` | 노면 크롭, 입력 147토큰 | `yes`, 확률 100% | 0.21초 |
| 비 | `ask_word("rain")` | 프레임, 입력 171토큰 | `no`, 확률 100% | 0.25초 |
| 통학버스 | `ask_word("bus")` | 트럭 크롭, 입력 123토큰 | `no`, 확률 100% | 0.19초 |
| 보호구역 횡단보도 | `ask_word("zebra")` | 노면 크롭, 입력 146토큰 | `yes`, 확률 100% | 0.22초 |
| 조문 찾기 | `watch` | 상황 3개, `laws["current"]` | 제12조 ③과 끌어오는 제12조 ①, 제27조 ①, 제27조 ⑦ | 없음 |
| 읽기 1 | `read_provision` | 제12조 ③과 ① | `SLOW_DOWN`, `시속 30킬로미터 이내로 운행해야 합니다.` | 0.60초 |
| 읽기 2 | `read_provision` | 제27조 ① | `STOP`, `보행자 횡단을 방해하지 않도록 일시정지해야 한다.` | 0.54초 |
| 읽기 3 | `read_provision` | 제27조 ⑦ | `STOP`, `보행자 횡단 여부와 관계없이 일시정지해야 합니다.` | 0.50초 |

신호등과 소화전 박스가 없어 그 질문들은 호출되지 않습니다. 같은 법령과 상황을 다시 만나면 읽기는 다시 실행되지 않습니다.

`vla-output.jsonl`에는 아래 내용이 한 줄로 기록됩니다(`text`는 앞부분만 적었습니다).

```json
{"rules": [{"law": "도로교통법 제27조 ⑦", "action": "STOP", "instruction": "보행자 횡단 여부와 관계없이 일시정지해야 합니다.", "text": "보행자의 보호: ⑦ 모든 차 또는 노면전차의 운전자는 ..."}]}
```

## 함수 색인

설명, 동작 방식, 파라미터 전체는 `EXPLAIN.md`의 같은 이름 절에 있습니다.

### detect.py

| 함수 | 하는 일 | 호출 위치 |
|---|---|---|
| `class_strip(frame, result, classes)` | 지정한 클래스의 박스 주변을 잘라 가로로 이은 이미지를 만든다. | `recognize` |
| `road_crop(frame)` | 노면 글자가 보이는 차로 앞쪽 영역을 자른다. | `recognize` |

### gemma.py

| 함수 | 하는 일 | 호출 위치 |
|---|---|---|
| `load_gemma()` | 비전 모듈을 붙인 Gemma를 불러온다. | `main.py` |
| `load_law(short)` | 법령의 상황별 조문 사전을 불러온다. | `main.py` |
| `load_embedder()` | 질문을 벡터로 바꾸는 임베딩 모델을 불러온다. | `main.py` |
| `load_corpus(short)` | 법령의 전체 항목과 미리 만든 벡터를 불러온다. | `main.py` |
| `search_law(embedder, question, corpus)` | 질문과 가장 가까운 법령 항목을 찾는다. | `answer_question` |
| `answer_question(llm, embedder, question, corpus)` | 질문에 맞는 조문을 찾아 읽고 답을 만든다. | `watch` |
| `start_stage(stage)` | 지금 도는 모델 호출을 화면 상태에 적는다. | `recognize`, `read_provision` |
| `end_stage(stage, answer, odds, tokens)` | 모델 호출의 답, 걸린 시간, 답별 확률, 입력 토큰 수를 화면 상태에 적는다. | `recognize`, `read_provision` |
| `answer_odds(llm, words)` | 방금 호출에서 모델이 낱말마다 매긴 점수를 낱말끼리의 확률로 바꾼다. | `recognize`, `read_provision` |
| `image_part(image, tokens)` | 이미지를 토큰 수에 맞게 줄여 Gemma 메시지 항목으로 만든다. | `recognize` |
| `parse_situations(answer)` | Gemma가 답한 번호 문자열을 상황 목록으로 바꾼다. | `recognize` |
| `first_token_bias(llm, words)` | 답의 첫 토큰을 낱말들의 첫 토큰으로 제한하는 `logit_bias`를 만든다. | `ask_word`, `read_provision` |
| `ask_word(llm, stage, image, question, words)` | 크롭 한 장을 보고 정해진 낱말 중 하나로 답하게 하고 답을 화면 상태에 적는다. | `recognize` |
| `recognize(llm, frame, result, in_school_zone)` | 프레임과 크롭을 보고 지금 있는 상황 목록을 만든다. | `watch` |
| `read_provision(llm, situation, article)` | 조문을 읽어 행동과 지시문을 만든다. | `watch` |
| `watch(llm, laws, shared)` | 최신 프레임으로 상황을 인식하고 조문을 읽어 규칙 목록을 계속 갱신한다. 한 번 나온 규칙은 `HOLD_SECONDS`(8초) 동안 남긴다. | `main.py`가 `threading.Thread`로 시작한다. |

### hud.py

| 함수 | 하는 일 | 호출 위치 |
|---|---|---|
| `button_boxes(frame_width, groups)` | 버튼 묶음마다 한 줄씩 패널 위쪽에 놓을 버튼 목록을 만든다. | `main.py` 시작 부분 |
| `draw_rail(pen, current)` | 처리 단계를 가로 한 줄로 그리고 지금 단계를 표시한다. | `draw_panel` |
| `card_image(image, picture)` | 지금 호출의 입력 이미지를 비율을 유지해 모서리가 둥근 칸에 그린다. | `draw_recognition_card` |
| `top_odds(answer)` | 답과 가장 높은 확률을 한 줄 글로 만든다. | `draw_recognition_card` |
| `draw_recognition_card(image, pen, status, scene)` | 지금 호출의 입력 이미지, 하는 일, 답을 그린다. | `draw_panel` |
| `draw_law_card(pen, status, rule, phase)` | 새 규칙이 나올 때 조문, 읽기, 출력을 한 단계씩 그린다. | `draw_panel` |
| `draw_panel(height, frame_width, buttons, selected, rules, status, scene, reading, phase)` | 탭, 지시문, 근거 조항, 처리 단계를 그린 패널 이미지를 만든다. | `main.py` 영상 루프 |
| `wrap(text, font, width)` | 글자 폭을 재어 패널 폭에 맞게 줄을 나눈다. | `draw_panel` |
| `draw_answer_card(pen, voice)` | 질문과 답 또는 음성 입력 상태를 그린다. | `draw_panel` |
| `draw(frame, result, panel_image, nodes_image, fps)` | 프레임에 박스와 클래스 이름을 그리고 패널과 조문 열을 붙인 화면을 만든다. | `main.py` 영상 루프 |

### main.py

| 함수 | 하는 일 | 호출 위치 |
|---|---|---|
| `detect_boxes()` | 가장 새로운 프레임에서 박스를 찾아 화면과 Gemma에 넘긴다. | `threading.Thread` |
| `toggle_listening()` | 듣는 중이면 녹음을 멈추고, 아니면 음성 질문 스레드를 시작한다. | `on_click`, 영상 루프 |
| `terminal_questions()` | 터미널에 입력한 줄을 질문 칸에 넣는다. | `threading.Thread` |
| `on_click(event, x, y, flags, param)` | 버튼을 누르면 고른 영상이나 법령 번호를 바꾼다. | `cv2.setMouseCallback`을 통해 OpenCV 창이 부른다. |

### build_corpus.py

| 함수 | 하는 일 | 호출 위치 |
|---|---|---|
| `fetch(oc, mst)` | 법령일련번호로 법령 본문 XML을 받는다. | `build_corpus.py` 맨 아래 반복문 |
| `clean(text)` | 줄바꿈과 겹친 공백을 한 칸으로 줄이고 끝의 개정 이력을 지운다. | `law_articles` |
| `law_articles(root)` | 도로교통법 XML을 항마다 한 항목으로 모은다. | `build_corpus.py` 맨 아래 반복문 |
| `signal_rows(root)` | 시행규칙 별표 2의 선 문자 표를 신호 종류마다 한 항목으로 모은다. | `build_corpus.py` 맨 아래 반복문 |
| `california_sections(zip_path)` | 캘리포니아 차량법 제11편 조문을 항마다 한 항목으로 모은다. | `build_corpus.py` 맨 아래 |
| `save(file_name, name, articles)` | 코퍼스 이름과 항목을 JSON 파일 하나로 저장한다. | `build_corpus.py` 맨 아래 |

## 라이브러리 색인

| 라이브러리 | 왜 쓰는가 |
|---|---|
| `cv2`(OpenCV) | 영상 읽기, 크기 변환, 박스 그리기, 창과 마우스 처리 |
| `numpy` | 크롭 잇기, 확률 계산 |
| `ultralytics` | YOLO11n 탐지 |
| `llama_cpp` | Gemma E2B 실행, 이미지 입력, 답 제한, 마지막 점수 읽기 |
| `PIL` | 패널의 한글 그리기 |
| `threading` | 영상 루프와 YOLO, Gemma, 음성, 터미널 입력 스레드 분리 |
| `subprocess` | Whisper, Piper, 녹음, 재생 명령 실행 |
| `base64` | 이미지를 data URL로 변환 |
| `json` | 인덱스 읽기, VLA 출력 쓰기 |
| `argparse`, `time`, `os` | 실행 인자, 시간 측정, 글꼴 경로 확인 |

함수별로는 아래와 같습니다.

### Llama

| 함수 | 하는 일 | 쓴 곳 |
|---|---|---|
| `Llama.create_chat_completion` | 메시지로 답을 생성한다 | `gemma.recognize`, `gemma.ask_word` |
| `Llama.create_completion` | 글 프롬프트에 이어 생성한다 | `gemma.read_provision` |
| `Llama.tokenize` | 글을 토큰 번호로 바꾼다 | `gemma.read_provision`, `gemma.ask_word` |

### PIL

| 함수 | 하는 일 | 쓴 곳 |
|---|---|---|
| `PIL.Image.fromarray` | 배열을 PIL 이미지로 바꾼다 | `hud.card_image` |
| `ImageDraw.rounded_rectangle` | 둥근 사각형을 그린다 | `hud.card_image` |
| `ImageDraw.polygon` | 다각형을 칠한다 | `hud.draw_panel` |
| `PIL.Image.Image.paste` | 이미지를 붙인다 | `hud.card_image` |
| `PIL.Image.new` | 빈 이미지를 만든다 | `hud.py`, `hud.draw_panel` |
| `PIL.ImageDraw.Draw` | 그리기 객체를 만든다 | `hud.py`, `hud.draw_panel` |
| `PIL.ImageFont.truetype` | 글꼴을 읽는다 | `hud.py` |

### argparse

| 함수 | 하는 일 | 쓴 곳 |
|---|---|---|
| `argparse.ArgumentParser` | 명령행 인자 해석기를 만든다 | `main.py` |

### base64

| 함수 | 하는 일 | 쓴 곳 |
|---|---|---|
| `base64.b64encode` | 바이트를 base64로 바꾼다 | `gemma.image_part` |

### cv2

| 함수 | 하는 일 | 쓴 곳 |
|---|---|---|
| `cv2.VideoCapture` | 영상을 연다 | `main.py` |
| `cv2.resize` | 영상 크기를 바꾼다 | `main.py`, `detect.class_strip`, `gemma.image_part` |
| `cv2.namedWindow` | 이름 붙은 창을 만든다 | `main.py` |
| `cv2.setMouseCallback` | 창에 마우스 함수를 연결한다 | `main.py` |
| `cv2.imshow` | 창에 이미지를 표시한다 | `main.py` |
| `cv2.waitKey` | 키 입력을 기다린다 | `main.py` |
| `cv2.destroyAllWindows` | 창을 모두 닫는다 | `main.py` |
| `cv2.rectangle` | 사각형을 그린다 | `hud.draw` |
| `cv2.line` | 선을 그린다 | `hud.draw` |
| `cv2.copyMakeBorder` | 이미지 둘레를 늘린다 | `hud.draw` |
| `cv2.putText` | 영문 글자를 그린다 | `hud.draw` |
| `cv2.imencode` | 이미지를 메모리에서 압축한다 | `gemma.image_part` |

### json

| 함수 | 하는 일 | 쓴 곳 |
|---|---|---|
| `json.load` | JSON 파일을 읽는다 | `gemma.load_law` |
| `json.dump` | JSON 파일을 쓴다 | `build_corpus.save` |
| `json.dumps` | 자료를 JSON 문자열로 바꾼다 | `main.py` |

### llama_cpp

| 함수 | 하는 일 | 쓴 곳 |
|---|---|---|
| `llama_cpp.Llama` | GGUF 모델을 불러온다 | `gemma.load_gemma` |
| `llama_cpp.llama_chat_format.Gemma4ChatHandler` | 이미지 입력 채팅 형식을 처리한다 | `gemma.load_gemma` |

### numpy

| 함수 | 하는 일 | 쓴 곳 |
|---|---|---|
| `numpy.array` | 배열을 만든다 | `hud.py`, `hud.draw_panel` |
| `numpy.hstack` | 배열을 가로로 잇는다 | `detect.class_strip` |
| `numpy.full` | 한 값으로 채운 배열을 만든다 | `hud.card_image` |
| `numpy.zeros` | 0으로 채운 배열을 만든다 | `detect.class_strip` |

### os

| 함수 | 하는 일 | 쓴 곳 |
|---|---|---|
| `os.path.exists` | 경로가 있는지 확인한다 | `hud.py` |

### threading

| 함수 | 하는 일 | 쓴 곳 |
|---|---|---|
| `threading.Thread` | 함수를 별도 스레드로 실행한다 | `main.py` |

### time

| 함수 | 하는 일 | 쓴 곳 |
|---|---|---|
| `time.perf_counter` | 정밀한 시각을 돌려준다 | `main.py`, `gemma.watch` |
| `time.time` | 현재 시각을 유닉스 초로 돌려준다 | `main.py` |
| `time.sleep` | 지정 시간 동안 멈춘다 | `gemma.watch` |

### ultralytics

| 함수 | 하는 일 | 쓴 곳 |
|---|---|---|
| `ultralytics.YOLO` | YOLO 모델을 불러온다 | `main.py` |

### urllib

| 함수 | 하는 일 | 쓴 곳 |
|---|---|---|
| `urllib.request.request.urlopen`(`urllib.request.urlopen`)` | URL을 연다 | `build_corpus.fetch` |

### xml

| 함수 | 하는 일 | 쓴 곳 |
|---|---|---|
| `xml.etree.ElementTree.fromstring` | XML 문자열을 읽는다 | `build_corpus.fetch`, `build_corpus.california_sections` |

### zipfile

| 함수 | 하는 일 | 쓴 곳 |
|---|---|---|
| `zipfile.ZipFile` | zip 파일을 연다 | `build_corpus.california_sections` |

## 발표에서 나올 질문

**실행 중에 조문을 검색하나요?** 아니요. 상황별 조문은 빌드 단계에서 Sonnet이 고르고, 실행 중에는 `index-<법령>.json`에서 찾기만 합니다. Jetson에서 E2B로 전체 조문을 고르는 방법은 시간과 정확도에서 Sonnet에 못 미쳤습니다.

**그럼 손으로 쓴 대응표와 무엇이 다른가요?** 사람이 조문을 고르지 않습니다. Sonnet이 라벨을 보지 않고 코퍼스에서 고르고, 법령을 바꾸면 같은 절차로 인덱스를 다시 만듭니다. 지시문도 표에서 꺼내지 않고 Gemma가 조문을 읽어 만듭니다. 상황 목록 자체는 사람이 정했습니다.

**조문 없음 모드는 무엇을 보여 주나요?** 같은 Gemma, 같은 상황에 법령 본문만 빼고 묻습니다. 소화전에서 조문 없는 답은 `정지하고 안전하게 통과하세요.`이고, 캘리포니아 조문을 읽으면 `소화전에서 15피트 이내 정차 금지.`입니다.

**왜 객관식에 새 상황을 넣지 않았나요?** 넣으면 다른 상황의 답이 흔들렸습니다. 제27조 ⑦을 객관식에 넣었을 때 현행 재현이 35/53에서 33/53으로 떨어졌습니다. 크롭 질문으로 물은 뒤에는 개발용 94사이클에서 기존 상황의 답이 바뀌지 않았습니다.

**노면 젖음 호출을 왜 뺐나요?** 비와 고인 물을 묻기 전에 거르던 호출입니다. 고인 물 사례를 빼자 거를 호출이 비 하나만 남아, 비를 바로 묻도록 바꿨습니다. 비 오는 장면의 호출이 3번에서 1번으로 줄었고, 맑은 날에는 호출 수가 같습니다. 개발용 재생 883사이클에서 비 오탐은 0건입니다.

**확률은 어떻게 구하나요?** `llama_get_logits_ith`로 생성 뒤 마지막 점수를 읽고, 허용한 낱말들끼리 softmax를 계산합니다. 화면 표시에만 쓰고 판단에는 쓰지 않습니다.

**왜 영상 루프와 Gemma를 나누었나요?** 판단 한 번이 Mac에서 1초가 넘게 걸립니다. 같은 스레드면 영상이 그동안 멈춥니다. 두 스레드는 `shared` 사전으로 최신 입력과 규칙 목록을 주고받습니다.

**8초 유지는 왜 있나요?** 판단이 지나간 프레임의 상황을 놓쳐도 지시가 바로 사라지지 않게 합니다. Jetson 판단이 약 5초이고 처음 읽는 조문에 약 5초가 더 걸려, 다음 판단까지 지시가 남도록 3.5초에서 8초로 늘렸습니다(2026-11-07).

**시연 영상은 실제 주행인가요?** 네. 비, 보호구역, 통학버스는 유튜브 실제 주행 영상에서, 소화전과 금문교는 위키미디어 영상에서 잘랐습니다. 터널은 실제 블랙박스 프레임이지만 사례 10개를 이어서 경계에서 장면이 끊깁니다. 라벨이 없어 시연 영상의 인식률은 성능값이 아닙니다.

**시험용 값이 왜 낮은가요?** 현행 재현 6/31입니다. 버스 전용차로 0/9와 회전 금지 0/5는 YOLO로 자를 수 없는 표지 때문이고, 다리 위 0/8은 8장 중 5장을 알아봤지만 행동 코드가 `CAUTION`이라 틀린 것으로 셌습니다. 조항만 보면 11/31입니다.

**가장 큰 한계는 무엇인가요?** 판단 라벨을 Claude 하나가 만들었고, 상황 목록 밖의 규정에는 지시를 내지 않습니다. Jetson 실측도 아직 없습니다.
