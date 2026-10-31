"""장면 상태별 정답 조문으로 검색 결과를 확인하는 테스트"""

import unittest

from retrieve import STATE_QUERIES, Retriever

EXPECTED = {
    "적색 신호": "시행규칙 별표 2 차량신호등 적색의등화",
    "황색 신호": "시행규칙 별표 2 차량신호등 황색의등화",
    "녹색 신호": "시행규칙 별표 2 차량신호등 녹색의등화",
    "보행자": "도로교통법 제27조 보행자의 보호",
    "자전거": "도로교통법 제19조 안전거리 확보 등",
    "앞차": "도로교통법 제19조 안전거리 확보 등",
    "정지 표지": "도로교통법 제31조 서행 또는 일시정지할 장소",
}


class RetrieveTest(unittest.TestCase):
    def test_every_state_finds_expected_article_first(self):
        for corpus_path in ("corpus-current.json", "corpus-2021.json"):
            retriever = Retriever(corpus_path)
            for state, expected in EXPECTED.items():
                top = retriever.search(STATE_QUERIES[state], 1)[0]
                self.assertEqual(top["id"] + " " + top["title"], expected, corpus_path + " " + state)

    def test_red_signal_text_differs_between_versions(self):
        current = Retriever("corpus-current.json").search(STATE_QUERIES["적색 신호"], 1)[0]
        old = Retriever("corpus-2021.json").search(STATE_QUERIES["적색 신호"], 1)[0]
        self.assertIn("정지한 후", current["text"])
        self.assertNotIn("정지한 후", old["text"])


if __name__ == "__main__":
    unittest.main()
