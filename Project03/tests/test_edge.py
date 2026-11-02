"""엣지 케이스 발동 판정, 조문 문장 나누기, 엣지 케이스별 검색 정답을 모델 없이 확인하는 테스트"""

import unittest

from edge import EDGE_CASES, active_edge_cases, sentences
from retrieve import Retriever
from tests.edge_gold import GOLD


class EdgeCaseTest(unittest.TestCase):
    def test_triggers(self):
        self.assertEqual(active_edge_cases({"보행자", "횡단보도", "앞차"}), ["횡단보도 보행자"])
        self.assertEqual(active_edge_cases({"보행자"}), [])
        self.assertEqual(active_edge_cases({"앞차"}), [])

    def test_sentences_keep_item_numbers_and_endings(self):
        text = "③ 모든 차의 운전자는 다음 곳에서는 앞지르지 못한다. 1. 교차로 2. 터널 안"
        self.assertEqual(sentences(text), ["③ 모든 차의 운전자는 다음 곳에서는 앞지르지 못한다.", "1. 교차로 2. 터널 안"])
        self.assertEqual(sentences("(a) A driver shall stop. (b) A driver may turn."), ["(a) A driver shall stop.", "(b) A driver may turn."])

    def test_every_corpus_has_query_for_every_edge_case(self):
        for corpus_path in GOLD:
            self.assertEqual(set(Retriever(corpus_path).queries), set(EDGE_CASES), corpus_path)

    def test_edge_case_search_finds_gold_provision_and_sentence(self):
        for corpus_path, gold in GOLD.items():
            retriever = Retriever(corpus_path)
            for name, (provisions, key, action) in gold.items():
                if not provisions:
                    continue
                top = retriever.search(retriever.queries[name], 1)[0][1]
                self.assertIn((top["id"] + " " + top["title"]).strip(), provisions, corpus_path + " " + name)
                # 정답 문장 문자열은 검색된 조항의 문장 하나에만 있어야 정답 문장을 가릴 수 있음
                matches = [line for line in sentences(top["text"]) if key in line]
                self.assertEqual(len(matches), 1, corpus_path + " " + name)


if __name__ == "__main__":
    unittest.main()
