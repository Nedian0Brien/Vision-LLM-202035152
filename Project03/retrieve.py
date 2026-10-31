"""도로교통법 코퍼스를 BM25로 검색하는 모듈"""

import json
import math

# BM25 논문과 대부분의 구현이 쓰는 기본값임
K1 = 1.5
B = 0.75


def tokenize(text):
    """공백을 빼고 소문자로 바꾼 글을 두 글자씩 묶은 바이그램 목록을 만드는 함수"""
    # 형태소 분석기 없이도 조사가 붙은 낱말이 같은 바이그램을 공유함
    # 영어 문장 첫 글자의 대문자도 같은 바이그램이 되도록 소문자로 바꿈
    text = text.replace(" ", "").lower()
    return [text[i:i + 2] for i in range(len(text) - 1)]


class Retriever:
    """조문 코퍼스를 읽어 BM25 점수로 검색하는 클래스"""

    def __init__(self, corpus_path):
        with open(corpus_path, encoding="utf-8") as file:
            corpus = json.load(file)
        self.name = corpus["name"]
        self.queries = corpus["queries"]
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
        return [self.articles[i] for score, i in scores[:k]]
