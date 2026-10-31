# Project03: 주행 영상 기반 도로교통법 검색 HUD

주행 영상에서 YOLO로 차량, 보행자, 신호등을 찾습니다. 찾은 객체를 "적색 신호", "앞차" 같은 장면 상태로 바꿉니다. 장면 상태마다 도로교통법 코퍼스를 BM25로 검색하고, 근거 조문을 화면 오른쪽에 보여 줍니다.

모델이 학습한 지식 대신 실제 법령 문서를 근거로 판단을 보여 주는 Retrieval-Augmented 구조입니다. 법령이 개정되거나 다른 나라에서 달리면 모델을 다시 학습하지 않습니다. 코퍼스 파일만 바꾸면 같은 영상에서 그 판, 그 나라의 조문이 나옵니다. 코퍼스는 한국 도로교통법 현행판, 2021년판, 캘리포니아 차량법(California Vehicle Code) 세 개입니다.

**같은 장면, 다른 판.** 원주 영상의 적색 신호 장면입니다.

| 현행 코퍼스 (`corpus-current.json`) | 2021년 코퍼스 (`corpus-2021.json`) |
|---|---|
| ![현행 코퍼스의 적색 신호 조문](docs/red-signal-current.jpg) | ![2021년 코퍼스의 적색 신호 조문](docs/red-signal-2021.jpg) |

현행 시행규칙 별표 2는 우회전하기 전에 정지선 직전에서 정지하라고 정합니다. 2021년 판은 정지 없이 우회전할 수 있다고 정합니다.

**같은 장면, 다른 나라.** 비벌리힐스 영상의 적색 신호와 보행자 장면입니다.

| 현행 코퍼스 | 2021년 코퍼스 | 캘리포니아 코퍼스 (`corpus-california.json`) |
|---|---|---|
| ![현행 코퍼스의 비벌리힐스 장면](docs/beverly-current.jpg) | ![2021년 코퍼스의 비벌리힐스 장면](docs/beverly-2021.jpg) | ![캘리포니아 코퍼스의 비벌리힐스 장면](docs/beverly-california.jpg) |

보행자 패널은 현행 제27조 ①("통행하려고 하는 때"를 포함), 2021년 제27조 ①, 캘리포니아 차량법 21950조 (a)를 보여 줍니다. 적색 신호 패널은 캘리포니아에서 21453조 (b)("after stopping … may turn right")를 보여 줍니다.

그림 다섯 장은 원주 시연 영상 25.5초(원본 111.5초)와 비벌리힐스 시연 영상 46초(원본 486초) 프레임을 저장한 것입니다. 화면 오른쪽 위 FPS 값은 저장용 스크립트가 넣은 영상 FPS이며 처리 속도 측정값이 아닙니다.

## 파일 구조

```
Project03/
├── main.py              실행 파일: 영상 → 탐지 → 장면 상태 → 검색 → 화면
├── detect.py            YOLO 탐지 결과를 장면 상태로 바꿈, 신호등 색 판정(HSV)
├── retrieve.py          BM25 검색기와 장면 상태별 검색 문장
├── hud.py               탐지 박스와 근거 조문 패널 그리기
├── build_corpus.py      코퍼스 JSON 세 개 생성 (Mac에서 한 번 실행)
├── corpus-current.json  현행 도로교통법(항 단위)과 시행규칙 별표 2(신호 종류 단위)
├── corpus-2021.json     2022년 1월 개정 직전 공포본으로 만든 같은 구성의 코퍼스
├── corpus-california.json  캘리포니아 차량법 제11편 Rules of the Road(항 단위, 영어)
├── tests/
│   └── test_retrieve.py 장면 상태별 정답 조문 검색 평가
├── docs/                README 그림
├── data/                캘리포니아 의회 일괄 파일 (git에 올리지 않음)
└── videos/              주행 영상 (git에 올리지 않음)
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

- 한글 글꼴은 Noto CJK(`fonts-noto-cjk`), 나눔고딕, Apple SD Gothic Neo 중 설치된 것을 씁니다.

### 주행 영상

영상은 두 개를 씁니다. 두 영상 모두 레포에 넣지 않습니다.

**원주 영상(한국 블랙박스).** 위키미디어 공용의 [2020-04-16 원주시 도로주행](https://commons.wikimedia.org/wiki/File:2020-04-16_%EC%9B%90%EC%A3%BC%EC%8B%9C_%EB%8F%84%EB%A1%9C%EC%A3%BC%ED%96%89.webm)을 씁니다. 촬영자는 Choi Kwang-mo이고 라이선스는 CC0(퍼블릭 도메인)입니다. 원본은 1920×1080, 24 FPS, 182초, VP9(.webm), 254MB입니다.

아래 명령으로 `videos/`에 내려받습니다.

```bash
mkdir -p videos && curl -L -o videos/wonju-2020-04-16.webm "https://upload.wikimedia.org/wikipedia/commons/c/c3/2020-04-16_%EC%9B%90%EC%A3%BC%EC%8B%9C_%EB%8F%84%EB%A1%9C%EC%A3%BC%ED%96%89.webm"
```

시연에는 원본의 86~118초를 1280×720 mp4(`mp4v` 코덱)로 다시 저장한 `videos/wonju-demo.mp4`를 씁니다. 이 구간에서 앞차는 시연 영상 4.7초와 24초 부근에, 적색 신호는 24.5~27.4초(원본 110.5초)에 패널에 나옵니다. 원본 90초의 적색 신호는 이 파일에서 5프레임 연속 탐지가 되지 않아 패널에 나오지 않습니다. Jetson의 OpenCV가 VP9을 읽지 못할 경우에 대비한 파일입니다. Mac에서 OpenCV `VideoWriter`로 만들었습니다.

**비벌리힐스 영상(미국 캘리포니아).** 위키미디어 공용의 [Driving through BEVERLY HILLS, Los Angeles, California (HD)](https://commons.wikimedia.org/wiki/File:Driving_through_BEVERLY_HILLS,_Los_Angeles,_California_(HD).webm)을 씁니다. 촬영자는 Kauko Helavuo이고 라이선스는 [CC BY 3.0](https://creativecommons.org/licenses/by/3.0)입니다. 원본은 2014-07-09 촬영, 1920×1080(4:3 화면에 좌우 검은 띠), 25 FPS, 500초, 172MB입니다. 원본을 자르고 크기를 바꿔 시연 영상을 만들었습니다.

```bash
curl -L -o videos/beverly-hills.webm "https://upload.wikimedia.org/wikipedia/commons/9/92/Driving_through_BEVERLY_HILLS%2C_Los_Angeles%2C_California_%28HD%29.webm"
```

시연에는 원본의 440~497초를 1280×720 mp4로 다시 저장한 `videos/beverly-demo.mp4`를 씁니다. 이 구간에는 STOP 표지, 황색·적색·녹색 신호, 횡단보도 앞 보행자가 나옵니다.

### 실행 명령

```bash
python main.py videos/wonju-demo.mp4
```

```bash
python main.py videos/wonju-demo.mp4 --corpus corpus-2021.json
```

```bash
python main.py videos/beverly-demo.mp4 --corpus corpus-california.json
```

Jetson에서는 04 노트북에서 만든 TensorRT 엔진을 넘깁니다.

```bash
python main.py videos/wonju-demo.mp4 --model ../src/models/YOLO/yolo11n_fp16.engine
```

`q`를 누르면 종료합니다.

### 화면

- 왼쪽: 영상과 탐지 박스. 장면 상태를 만든 박스는 주황색 박스와 상태 이름으로, 나머지 탐지는 회색 박스로 그립니다.
- 오른쪽 패널 위: 코퍼스 이름과 처리 FPS
- 오른쪽 패널: 장면 상태마다 상태 이름, 근거 조문 번호와 제목, 본문 앞 다섯 줄. 줄은 공백 위치에서 바꿉니다.

### 처리 과정 (한 프레임)

1. 프레임을 1280×720으로 줄이고 YOLO(`conf=0.4`, `iou=0.5`)로 6개 클래스를 탐지합니다.
2. 박스마다 장면 상태를 정합니다.
   - 신호등: 박스 안 HSV 색 픽셀 수로 적색·황색·녹색을 고릅니다.
   - 사람, 자전거, 정지 표지: 화면의 0.2% 이상 크기일 때만 씁니다.
   - 승용차, 트럭: 화면의 3% 이상이고 화면 가운데 3분의 1에 있을 때 앞차가 됩니다.
   - 화면 아래 20%에 중심이 있는 박스는 버립니다. 이 영역에는 보닛과 유리창 스티커가 있어 정지 표지 오탐이 생깁니다.
3. 상태마다 5프레임 연속 탐지되면 확정하고, 확정이 끊긴 뒤에도 2초 동안 패널에 남깁니다.
4. 패널의 상태 목록이 바뀐 프레임에서만, 코퍼스에 담긴 상태별 검색 문장으로 BM25 1위 항목을 검색합니다.
5. 처리가 영상보다 빠르면 남는 시간만큼 기다려 원래 속도로 재생합니다.

### 테스트와 평가

```bash
python -m unittest discover -s tests -t .
```

- 세 코퍼스 모두에서 장면 상태 7개(적색·황색·녹색 신호, 보행자, 자전거, 앞차, 정지 표지)의 BM25 1위가 정답 항목과 같은지 확인합니다. 캘리포니아 정답은 21453(b), 21452(a), 21451(a), 21950(a), 21760(c), 21703, 22450(a)입니다.
- 적색 신호(별표 2)와 보행자(제27조 ①)의 1위 본문이 두 판에서 다른지 확인합니다.
- 2026-10-31 Mac에서 2개 테스트가 모두 통과했습니다. Mac(Apple Silicon, CPU)의 프레임당 처리 시간은 약 21ms였습니다.
- 비벌리힐스 시연 영상에서 정지 표지(2초), 앞차, 보행자(12.1초, 44.3초), 황색 신호(33.8초), 적색 신호(44.6초), 녹색 신호(49.3초)가 패널에 나왔습니다.

## 법령 코퍼스

### 파일 형식

코퍼스 파일은 `{"name", "queries", "articles"}` 사전입니다. `queries`는 장면 상태 이름별 검색 문장이고, 그 법령이 쓰는 말로 씁니다. 한국어 문장으로는 영어 차량법을 검색할 수 없기 때문입니다. 검색 문장이 코드가 아니라 코퍼스에 있으므로, 나라를 바꿀 때 코드를 고치지 않습니다.

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

- 원주 영상에는 보행자 장면이 없습니다. 보행자 장면은 비벌리힐스 영상으로 확인합니다.
- 비벌리힐스 영상은 블랙박스가 아니라 손으로 든 카메라로 찍은 영상입니다. 앞차 판정(화면 가운데 3분의 1)과 화면 아래 20% 제외 규칙은 블랙박스 화면을 기준으로 정한 값입니다.
- 검색 문장은 사람이 법령 용어로 썼습니다. 캘리포니아 정지 표지 문장은 처음 쓴 문장("stop sign at entrance to intersection …")으로는 21457조(점멸 적색 신호)가 1위였고, 조문 용어("approaching a stop sign …")로 고쳐 쓴 뒤 22450조 (a)가 1위가 되었습니다. 영어를 단어 단위로 나눠 시험했을 때는 처음 문장으로도 7개 모두 1위였습니다. 문자 바이그램은 영어에서 낱말 경계를 잃어 한국어보다 검색 문장에 민감합니다.
- COCO 모델은 횡단보도, 차선, 한국 교통안전표지를 찾지 못합니다. 장면 상태는 6개 클래스와 신호등 색으로 만들 수 있는 7개뿐입니다.
- 별표 2는 선 문자 표에서 글자를 읽습니다. 칸 안 줄바꿈이 낱말 중간에서 일어나 일부 띄어쓰기(`교통을방해하지`)는 복원하지 못합니다.
- 패널은 장면 상태 3개까지 화면 안에 들어갑니다. 4개 이상이면 아래 항목이 잘립니다.
- Jetson 실행, TensorRT 엔진 사용, mp4 시연 영상 읽기는 아직 확인하지 않았습니다.
- `../src/models/YOLO/yolo11n.pt`는 레포에 넣지 않습니다. 04 노트북 순서대로 준비합니다.

## 수업 내용과 연결

| 수업 내용 | 이 프로젝트에서 쓴 곳 |
|---|---|
| 02 노트북 색공간 변환과 색 기반 분할 | `detect.light_color`: HSV 변환, `cv2.inRange`, 빨강 두 범위 합치기 |
| 04 노트북 YOLO 추론과 클래스 선택 | `main.py`: `model.predict(source=frame, conf, iou, verbose=False, classes=CLASSES)` |
| 04 노트북 FPS 측정 | `main.py`: `time.perf_counter()`와 지수 이동 평균(0.9, 0.1) |
| 04 노트북 TensorRT 엔진 | Jetson 실행 시 `--model ../src/models/YOLO/yolo11n_fp16.engine` |
