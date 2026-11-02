"""토큰화와 개정 전후 검색 결과 차이를 확인하는 테스트"""

import unittest

from retrieve import Retriever, tokenize


class RetrieveTest(unittest.TestCase):
    def test_tokenize_keeps_english_words_and_korean_bigrams(self):
        self.assertEqual(tokenize("Stop sign 횡단 보도"), ["stop", "sign", "횡단", "단보", "보도"])

    def test_amended_text_differs_between_versions(self):
        current = Retriever("corpus-current.json")
        old = Retriever("corpus-2021.json")
        # 2022.1.21 공포 별표 2 개정으로 적색 신호 우회전 전 정지 의무가 생김
        self.assertIn("정지한 후", current.search(current.queries["적색 신호 교차로"], 1)[0][1]["text"])
        self.assertNotIn("정지한 후", old.search(old.queries["적색 신호 교차로"], 1)[0][1]["text"])
        # 2022.1.11 공포 제27조 개정으로 횡단하려는 보행자 앞에서도 일시정지해야 함
        self.assertIn("통행하려고 하는 때", current.search(current.queries["횡단보도 보행자"], 1)[0][1]["text"])
        self.assertNotIn("통행하려고 하는 때", old.search(old.queries["횡단보도 보행자"], 1)[0][1]["text"])


if __name__ == "__main__":
    unittest.main()
