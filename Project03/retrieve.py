"""도로교통법 코퍼스를 BM25로 검색하는 모듈"""

import json
import math

# BM25 논문과 대부분의 구현이 쓰는 기본값임
K1 = 1.5
B = 0.75

STATE_QUERIES = {
    "적색 신호": "차량신호등 적색의 등화 정지선 횡단보도 교차로 직전 정지 우회전",
    "황색 신호": "차량신호등 황색의 등화 정지선 교차로 직전 정지 우회전 보행자 횡단 방해",
    "녹색 신호": "차량신호등 녹색의 등화 직진 우회전 비보호좌회전",
    "보행자": "보행자 횡단보도 통행 일시정지 횡단 방해",
    "자전거": "자전거등 옆 지날 때 충돌 피할 거리 확보",
    "앞차": "앞차 뒤를 따르는 경우 충돌 피할 안전거리 확보",
    "정지 표지": "안전표지 일시정지 장소 지정",
}


def tokenize(text):
    """공백을 뺀 글자 두 개씩 묶은 바이그램 목록을 만드는 함수"""
    # 형태소 분석기 없이도 조사가 붙은 낱말이 같은 바이그램을 공유함
    text = text.replace(" ", "")
    return [text[i:i + 2] for i in range(len(text) - 1)]


class Retriever:
    """조문 코퍼스를 읽어 BM25 점수로 검색하는 클래스"""

    def __init__(self, corpus_path):
        with open(corpus_path, encoding="utf-8") as file:
            self.articles = json.load(file)
        self.counts = []
        self.lengths = []
        doc_freq = {}
        for article in self.articles:
            count = {}
            for token in tokenize(article["title"] + article["text"]):
                count[token] = count.get(token, 0) + 1
            self.counts.append(count)
            self.lengths.append(sum(count.values()))
            for token in count:
                doc_freq[token] = doc_freq.get(token, 0) + 1
        n = len(self.articles)
        self.idf = {}
        for token, freq in doc_freq.items():
            self.idf[token] = math.log((n - freq + 0.5) / (freq + 0.5) + 1)
        self.avg_length = sum(self.lengths) / n

    def search(self, query, k=3):
        query_tokens = set(tokenize(query))
        scores = []
        for i in range(len(self.articles)):
            count = self.counts[i]
            norm = K1 * (1 - B + B * self.lengths[i] / self.avg_length)
            score = 0.0
            for token in query_tokens:
                if token in count:
                    score += self.idf[token] * count[token] * (K1 + 1) / (count[token] + norm)
            scores.append((score, i))
        scores.sort(reverse=True)
        return [self.articles[i] for score, i in scores[:k]]
