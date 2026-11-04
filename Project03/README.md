# Project03: 주행 영상 기반 도로교통법 검색 HUD

지금 프로그램은 주행 영상에 YOLO11n 탐지 박스를 그려 보여 줍니다. 법령 검색과 주행 지시문은 들어 있지 않습니다.

이전 구조는 2026-11-04에 폐기했습니다. 이전 구조는 장면을 엣지 케이스 이름 17개로 나누고, 이름마다 손으로 쓴 검색 문장으로 조항을 찾았습니다. 이전 코드와 측정 기록은 커밋 `466546c`에 있습니다.

장면을 읽어 조항을 찾는 새 구조는 개발용 세트에서 시험 중입니다.

## 파일 구조

```text
Project03/
├── main.py              영상 재생, YOLO 탐지, 버튼 처리
├── detect.py            탐지 결과를 박스와 클래스 이름 목록으로 바꿈
├── hud.py               탐지 박스와 버튼 패널 그리기
├── gemma.py             Gemma 불러오기 (새 구조용, 아직 main.py가 부르지 않음)
├── retrieve.py          BM25 검색과 조문 문장 나누기 (새 구조용, 아직 main.py가 부르지 않음)
├── build_corpus.py      코퍼스 생성
├── corpus-current.json  현행 도로교통법과 시행규칙 별표 2
├── corpus-2021.json     2021년 공포본
├── corpus-california.json  캘리포니아 차량법 제11편
├── tests/               검색 테스트와 개발용·시험용 라벨
├── EXPLAIN.md           코드 설명
└── videos/, data/       git이 무시하는 영상과 원본 자료
```

## 실행 방식

### 준비

- 모든 명령은 `Project03/` 폴더에서 실행합니다. `--model` 기본값이 상대 경로 `../src/models/YOLO/yolo11n.pt`이기 때문입니다.
- Jetson에서는 수업 노트북 `00`, `03`, `04`에서 설치한 환경을 그대로 씁니다. 추가로 설치할 패키지는 없습니다.

| 패키지 | Jetson 수업 환경 | Mac 개발 환경 (`.venv-project03`) |
|---|---|---|
| Python | 3.10 | 3.10 |
| numpy | 1.23.0 | 1.23.5 |
| opencv-python | JetPack 기본 | 4.10.0 |
| Pillow | 10.4.0 | 10.4.0 |
| ultralytics | 04 노트북 설치본 | 8.4.194 |
| torch | 2.8.0 | 2.14.1 |
| llama-cpp-python | 05 노트북 설치본(CUDA 빌드) | 0.3.36(Metal 빌드) |

- Gemma 모델은 05 노트북과 같은 두 파일을 씁니다. 레포에는 넣지 않습니다. 05 노트북의 `hf_hub_download` 셀로 받거나 아래 명령으로 받습니다.
  - `../src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf`(3.46GB, Apache 2.0)
  - `../src/models/Gemma4/mmproj-google_gemma-4-E2B-it-f16.gguf`(0.99GB, 비전 모듈)

```bash
mkdir -p ../src/models/Gemma4 && curl -L -o ../src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf "https://huggingface.co/bartowski/google_gemma-4-E2B-it-GGUF/resolve/main/google_gemma-4-E2B-it-Q4_K_M.gguf"
```

```bash
curl -L -o ../src/models/Gemma4/mmproj-google_gemma-4-E2B-it-f16.gguf "https://huggingface.co/bartowski/google_gemma-4-E2B-it-GGUF/resolve/main/mmproj-google_gemma-4-E2B-it-f16.gguf"
```

- 한글 글꼴은 Noto CJK(`fonts-noto-cjk`), 나눔고딕, Apple SD Gothic Neo 중 설치된 것을 씁니다.

### 주행 영상

영상은 세 개를 씁니다. 모두 레포에 넣지 않습니다.

**원주 영상(한국 블랙박스).** 위키미디어 공용의 [2020-04-16 원주시 도로주행](https://commons.wikimedia.org/wiki/File:2020-04-16_%EC%9B%90%EC%A3%BC%EC%8B%9C_%EB%8F%84%EB%A1%9C%EC%A3%BC%ED%96%89.webm)을 씁니다. 촬영자는 Choi Kwang-mo이고 라이선스는 CC0(퍼블릭 도메인)입니다. 원본은 1920×1080, 24 FPS, 182초, VP9(.webm), 254MB입니다.

아래 명령으로 `videos/`에 내려받습니다.

```bash
mkdir -p videos && curl -L -o videos/wonju-2020-04-16.webm "https://upload.wikimedia.org/wikipedia/commons/c/c3/2020-04-16_%EC%9B%90%EC%A3%BC%EC%8B%9C_%EB%8F%84%EB%A1%9C%EC%A3%BC%ED%96%89.webm"
```

시연에는 원본의 86~118초를 1280×720 mp4(`mp4v` 코덱)로 다시 저장한 `videos/wonju-demo.mp4`를 씁니다. Jetson의 OpenCV가 VP9을 읽지 못할 경우에 대비한 파일입니다. Mac에서 OpenCV `VideoWriter`로 만들었습니다.

**비벌리힐스 영상(미국 캘리포니아).** 위키미디어 공용의 [Driving through BEVERLY HILLS, Los Angeles, California (HD)](https://commons.wikimedia.org/wiki/File:Driving_through_BEVERLY_HILLS,_Los_Angeles,_California_(HD).webm)을 씁니다. 촬영자는 Kauko Helavuo이고 라이선스는 [CC BY 3.0](https://creativecommons.org/licenses/by/3.0)입니다. 원본은 2014-07-09 촬영, 1920×1080(4:3 화면에 좌우 검은 띠), 25 FPS, 500초, 172MB입니다. 원본을 자르고 크기를 바꿔 시연 영상을 만들었습니다.

```bash
curl -L -o videos/beverly-hills.webm "https://upload.wikimedia.org/wikipedia/commons/9/92/Driving_through_BEVERLY_HILLS%2C_Los_Angeles%2C_California_%28HD%29.webm"
```

시연에는 원본의 440~497초를 1280×720 mp4로 다시 저장한 `videos/beverly-demo.mp4`를 씁니다. 이 구간에는 STOP 표지, 황색·적색·녹색 신호, 횡단보도 앞 보행자가 나옵니다.

**샌프란시스코 영상(미국 캘리포니아).** 위키미디어 공용의 [San Francisco Downtown Driving Tour, 2023 California, USA](https://commons.wikimedia.org/wiki/File:San_Francisco_Downtown_Driving_Tour,_2023_California,_USA._Travel_Guide,_(4K_HDR).webm)을 씁니다. 촬영자는 ZERO ONE이고 라이선스는 [CC BY 3.0](https://creativecommons.org/licenses/by/3.0)입니다. 시연에는 원본의 860~950초를 1280×720, 30 FPS mp4로 다시 저장한 `videos/sf-demo.mp4`를 씁니다.

### 실행 명령

```bash
python main.py videos/wonju-demo.mp4
```

Jetson에서는 04 노트북에서 만든 TensorRT 엔진을 넘깁니다.

```bash
python main.py videos/wonju-demo.mp4 --model ../src/models/YOLO/yolo11n_fp16.engine
```

YOLO 입력은 1280이고 1280×720 영상에서는 1280×736으로 추론합니다. 04 노트북의 `yolo11n_fp16.engine`은 640×640 고정 크기라서, Jetson에서는 같은 방식으로 `imgsz=[736, 1280]` 엔진을 새로 만들어 넘겨야 합니다. 아직 만들지 않았습니다.

오른쪽 패널의 버튼으로 영상을 바꿉니다. `q`를 누르면 종료합니다.

### 테스트

```bash
python -m unittest discover -s tests -t .
```

## 평가 라벨

`tests/edge_labels_dev.json`(57장)과 `tests/edge_labels_test.json`(61장)은 프레임마다 영상, 시각, 상황 이름, 법령별 정답 조항을 담습니다. 규칙과 프롬프트는 개발용에서만 정합니다. 시험용은 마지막에 한 번만 잽니다.

## 법령 코퍼스

### 파일 형식

코퍼스 파일은 `{"name", "articles"}` 사전입니다. `articles`의 항목은 `id`, `title`, `text`를 가집니다.

### 생성

```bash
curl -o data/pubinfo_2025.zip https://downloads.leginfo.legislature.ca.gov/pubinfo_2025.zip
```

```bash
python build_corpus.py <국가법령정보 Open API 인증키(OC)> data/pubinfo_2025.zip
```

- 인증키는 [국가법령정보 공동활용](https://open.law.go.kr)에서 발급받은 본인 키를 씁니다. 레포에는 적지 않습니다.
- 법령은 「저작권법」 제7조에 따라 보호받지 못하는 저작물입니다. 그래서 생성한 JSON을 레포에 넣었습니다.

| 파일 | 법령 | 법령일련번호(MST) | 공포일 |
|---|---|---|---|
| `corpus-current.json` | 도로교통법 | 291133 | 2026-10-13 |
| `corpus-current.json` | 도로교통법 시행규칙 | 289009 | 2026-08-24 |
| `corpus-2021.json` | 도로교통법 | 237009 | 2021-11-30 |
| `corpus-2021.json` | 도로교통법 시행규칙 | 239101 | 2021-12-31 |

`corpus-2021.json`은 아래 두 개정이 공포되기 직전의 공포본으로 만들었습니다. Open API의 법령 본문은 공포 시점 조문을 담고 시행 예정 개정분을 포함하기 때문에, 시행일이 아니라 공포일로 판을 골랐습니다.

- 2022-01-11 공포 도로교통법 제27조 ① 개정: 횡단보도를 "통행하려고 하는 때"에도 일시정지해야 함
- 2022-01-21 공포 시행규칙 별표 2 개정(2023-01-22 시행): 적색 신호에서 우회전하려면 정지선 직전에서 먼저 정지해야 함

### 구성

- 도로교통법은 항마다 한 항목입니다. `id`는 `도로교통법 제27조 ①` 형식입니다.
- 시행규칙 별표 2(신호기가 표시하는 신호의 종류 및 신호의 뜻)는 신호 종류마다 한 항목입니다. 제목은 `차량신호등 적색의등화` 형식입니다.
- 캘리포니아 차량법은 제11편(Division 11, Rules of the Road) 현행 조문 567개를 항(subdivision)마다 한 항목으로 나눕니다. `id`는 `CVC §21453(b)` 형식이고, 캘리포니아 법전에는 조문 제목이 없어 `title`은 비어 있습니다.
- 항목 수는 현행 592개, 2021년 536개, 캘리포니아 1,694개입니다.

### 캘리포니아 차량법 출처

- 캘리포니아 의회가 공개하는 일괄 데이터 [pubinfo_2025.zip](https://downloads.leginfo.legislature.ca.gov/)(2025–2026 회기, 2026-10-25 갱신, 1.2GB)의 `LAW_SECTION_TBL`에서 읽습니다.
- 의회 웹사이트(leginfo.legislature.ca.gov)는 스크립트 접근을 Cloudflare 확인 페이지로 막아서 쓰지 않았습니다.
- 미국 교통 법규는 주마다 다릅니다. 비벌리힐스 영상과 같은 지역인 캘리포니아주 법을 골랐습니다.

## 한계

- 프로그램은 법령을 검색하지 않고 지시문을 만들지 않습니다.
- YOLO 박스는 COCO 클래스(사람, 자전거, 승용차, 트럭, 신호등, 정지 표지)만 나타냅니다. 횡단보도, 차선, 한국 교통안전표지는 찾지 못합니다.
- 별표 2는 선 문자 표에서 글자를 읽습니다. 칸 안 줄바꿈이 낱말 중간에서 일어나 일부 띄어쓰기(`교통을방해하지`)는 복원하지 못합니다.
- `cap.get(cv2.CAP_PROP_FPS)`가 0을 돌려주면 `main.py`가 0으로 나눕니다. Jetson에서 아직 확인하지 않았습니다.
- `../src/models/YOLO/yolo11n.pt`는 레포에 넣지 않습니다. 04 노트북 순서대로 준비합니다.

## 수업 내용과 연결

| 수업 내용 | 이 프로젝트에서 쓴 곳 |
|---|---|
| 04 노트북 YOLO 추론과 클래스 선택 | `main.py`: `model.predict(source=frame, conf, iou, verbose=False, classes=CLASSES)` |
| 04 노트북 FPS 측정 | `main.py`: `time.perf_counter()`와 지수 이동 평균(0.9, 0.1) |
| 04 노트북 TensorRT 엔진 | Jetson 실행 시 `--model ../src/models/YOLO/yolo11n_fp16.engine` |
| 05 노트북 로컬 Gemma 로드 | `gemma.load_gemma`: `Llama(n_gpu_layers=-1, …)`. 이미지 토큰 때문에 `n_ctx=4096`, `n_batch=512`로 늘림 |
