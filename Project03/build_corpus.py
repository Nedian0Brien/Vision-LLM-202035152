"""국가법령정보 Open API로 도로교통법 코퍼스 JSON을 만드는 스크립트"""

import json
import sys
import urllib.request
import xml.etree.ElementTree as ET

API_URL = "https://www.law.go.kr/DRF/lawService.do?OC={oc}&target=law&type=XML&MST={mst}"

# 법령일련번호(MST)는 Open API 법령 검색(target=eflaw)의 연혁 목록에서 확인함
# 2021년 판은 제27조 제1항 개정(2022.1.11 공포)과 별표 2 개정(2022.1.21 공포) 직전 공포본임
VERSIONS = {
    "corpus-current.json": {"law": "291133", "rule": "289009"},
    "corpus-2021.json": {"law": "237009", "rule": "239101"},
}


def fetch(oc, mst):
    with urllib.request.urlopen(API_URL.format(oc=oc, mst=mst)) as response:
        return ET.fromstring(response.read())


def clean(text):
    """줄바꿈과 겹친 공백을 한 칸으로 줄이고 끝의 개정 이력을 지우는 함수"""
    # 줄 끝의 <개정 2020.6.9> 같은 개정 이력은 검색과 표시에 쓰지 않음
    return " ".join(text.split()).split(" <")[0]


def law_articles(root):
    """도로교통법 조문을 항마다 한 항목으로 모으는 함수"""
    articles = []
    for unit in root.iter("조문단위"):
        # 장·절 제목 줄과 삭제된 조문은 조문제목이 없음
        if unit.findtext("조문여부") != "조문" or unit.findtext("조문제목") is None:
            continue
        number = "제" + unit.findtext("조문번호") + "조"
        if unit.findtext("조문가지번호"):
            number += "의" + unit.findtext("조문가지번호")
        # 조문내용은 "제2조(정의) 이 법에서 …"처럼 조문 머리와 항이 없는 조문의 본문을 담음
        head = clean(unit.findtext("조문내용").split(")", 1)[1])
        paragraphs = unit.findall("항")
        if not paragraphs:
            articles.append({"id": "도로교통법 " + number, "title": unit.findtext("조문제목"), "text": head})
        for paragraph in paragraphs:
            lines = []
            if paragraph.findtext("항번호") is None:
                lines.append(head)
            for element in paragraph.iter():
                if element.tag in ("항내용", "호내용", "목내용"):
                    lines.append(clean(element.text))
            paragraph_id = ("도로교통법 " + number + " " + paragraph.findtext("항번호", "")).strip()
            articles.append({"id": paragraph_id, "title": unit.findtext("조문제목"), "text": " ".join(lines)})
    return articles


def signal_rows(root):
    """시행규칙 별표 2 신호 표를 신호 종류마다 한 항목으로 나누는 함수"""
    for table in root.iter("별표단위"):
        if table.findtext("별표구분") == "별표" and table.findtext("별표번호") == "0002":
            break
    # 첫 번째 ┠ 줄 앞은 표 머리글이므로 버림
    body = table.findtext("별표내용").split("┠", 1)[1]
    rows = []
    group, kind, meaning = "", "", ""
    group_start = 0
    for line in body.split("\n"):
        border = line.startswith("┠") or line.startswith("┗") or "├" in line
        if line.startswith("┃") and "│" in line:
            cells = line.strip(" ┃").split("│")
            # 신호등 구분 칸은 세로로 병합되어 있어 경계 줄에도 글자가 있음
            group += cells[0].strip()
            if not border:
                kind += cells[-2].strip()
                meaning += cells[-1].strip()
        if border and meaning:
            rows.append({"id": "시행규칙 별표 2", "title": kind, "text": meaning})
            kind, meaning = "", ""
        # ┠ 줄과 ┗ 줄은 차량·보행·자전거처럼 신호등 구분이 끝나는 경계임
        if line.startswith("┠") or line.startswith("┗"):
            for row in rows[group_start:]:
                row["title"] = group.replace(" ", "") + " " + row["title"]
            group, group_start = "", len(rows)
    return rows


oc = sys.argv[1]
for file_name, version in VERSIONS.items():
    corpus = law_articles(fetch(oc, version["law"])) + signal_rows(fetch(oc, version["rule"]))
    with open(file_name, "w", encoding="utf-8") as file:
        json.dump(corpus, file, ensure_ascii=False, indent=1)
    print(file_name, len(corpus))
