"""도로교통법 코퍼스를 BM25로 검색하는 모듈"""

import json
import math

# BM25 논문과 대부분의 구현이 쓰는 기본값임
K1 = 1.5
B = 0.75


def tokenize(text):
    """영어 낱말은 낱말 그대로, 한국어는 이어 붙여 두 글자씩 묶은 토큰 목록을 만드는 함수"""
    tokens = []
    korean = ""
    for word in text.lower().split():
        # 영어를 글자 두 개씩 자르면 "on", "wa" 같은 조각이 어느 낱말에나 있어 지명도 높은 점수를 받았음
        if word.isascii():
            tokens.append(word.strip(".,;:()"))
        else:
            korean += word
    # 형태소 분석기 없이도 조사가 붙은 한국어 낱말이 같은 바이그램을 공유함
    return tokens + [korean[i:i + 2] for i in range(len(korean) - 1)]


class Retriever:
    """조문 코퍼스를 읽어 BM25 점수로 검색하는 클래스"""

    def __init__(self, corpus_path):
        with open(corpus_path, encoding="utf-8") as file:
            corpus = json.load(file)
        self.name = corpus["name"]
        self.articles = corpus["articles"]
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
        return [(score, self.articles[i]) for score, i in scores[:k]]


def sentences(text):
    """조문을 문장 목록으로 나누는 함수"""
    # 한국 조문은 "1. 교차로 2. 터널 안"처럼 호 번호에도 마침표가 있어 문장 끝인 "다. "에서만 나눔
    # 영어 조문에는 "다"가 없고 한국 조문은 모두 "-다"로 끝나는 문장을 가짐
    if "다" in text:
        parts, ending = text.split("다. "), "다."
    else:
        parts, ending = text.split(". "), "."
    # 나눌 때 잘려 나간 문장 끝을 마지막 조각을 뺀 나머지에 다시 붙임
    return [part + ending for part in parts[:-1]] + parts[-1:]
