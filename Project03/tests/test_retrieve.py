"""장면 상태별 정답 조문으로 검색 결과를 확인하는 테스트"""

import unittest

from retrieve import Retriever

KOREAN_EXPECTED = {
    "적색 신호": "시행규칙 별표 2 차량신호등 적색의등화",
    "황색 신호": "시행규칙 별표 2 차량신호등 황색의등화",
    "녹색 신호": "시행규칙 별표 2 차량신호등 녹색의등화",
    "보행자": "도로교통법 제27조 ① 보행자의 보호",
    "자전거": "도로교통법 제19조 ② 안전거리 확보 등",
    "앞차": "도로교통법 제19조 ① 안전거리 확보 등",
    "정지 표지": "도로교통법 제31조 ② 서행 또는 일시정지할 장소",
}
CALIFORNIA_EXPECTED = {
    "적색 신호": "CVC §21453(b)",
    "황색 신호": "CVC §21452(a)",
    "녹색 신호": "CVC §21451(a)",
    "보행자": "CVC §21950(a)",
    "자전거": "CVC §21760(c)",
    "앞차": "CVC §21703",
    "정지 표지": "CVC §22450(a)",
}
EXPECTED = {
    "corpus-current.json": KOREAN_EXPECTED,
    "corpus-2021.json": KOREAN_EXPECTED,
    "corpus-california.json": CALIFORNIA_EXPECTED,
}


class RetrieveTest(unittest.TestCase):
    def test_every_state_finds_expected_article_first(self):
        for corpus_path, expected in EXPECTED.items():
            retriever = Retriever(corpus_path)
            for state, answer in expected.items():
                top = retriever.search(retriever.queries[state], 1)[0]
                self.assertEqual((top["id"] + " " + top["title"]).strip(), answer, corpus_path + " " + state)

    def test_amended_text_differs_between_versions(self):
        current = Retriever("corpus-current.json")
        old = Retriever("corpus-2021.json")
        # 2022.1.21 공포 별표 2 개정으로 적색 신호 우회전 전 정지 의무가 생김
        self.assertIn("정지한 후", current.search(current.queries["적색 신호"], 1)[0]["text"])
        self.assertNotIn("정지한 후", old.search(old.queries["적색 신호"], 1)[0]["text"])
        # 2022.1.11 공포 제27조 개정으로 횡단하려는 보행자 앞에서도 일시정지해야 함
        self.assertIn("통행하려고 하는 때", current.search(current.queries["보행자"], 1)[0]["text"])
        self.assertNotIn("통행하려고 하는 때", old.search(old.queries["보행자"], 1)[0]["text"])


if __name__ == "__main__":
    unittest.main()
