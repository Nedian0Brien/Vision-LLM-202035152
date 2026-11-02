"""엣지 케이스 발동 판정, 지시문 검사, 엣지 케이스별 검색 정답을 모델 없이 확인하는 테스트"""

import unittest

from edge import EDGE_CASES, active_edge_cases, check_instruction
from retrieve import Retriever
from tests.edge_gold import GOLD


class EdgeCaseTest(unittest.TestCase):
    def test_triggers(self):
        self.assertEqual(active_edge_cases({"보행자", "횡단보도", "앞차"}), ["횡단보도 보행자"])
        self.assertEqual(active_edge_cases({"보행자"}), ["차도 보행자"])
        self.assertEqual(active_edge_cases({"앞차"}), [])

    def test_every_corpus_has_query_for_every_edge_case(self):
        for corpus_path in GOLD:
            self.assertEqual(set(Retriever(corpus_path).queries), set(EDGE_CASES), corpus_path)

    def test_edge_case_search_finds_gold_provision_first(self):
        for corpus_path, gold in GOLD.items():
            retriever = Retriever(corpus_path)
            for name, (provisions, action) in gold.items():
                if not provisions:
                    continue
                top = retriever.search(retriever.queries[name], 1)[0][1]
                self.assertIn((top["id"] + " " + top["title"]).strip(), provisions, corpus_path + " " + name)

    def test_check_instruction_catches_lost_negation_and_condition(self):
        article = "운전자는 보행자가 횡단보도를 통행하고 있을 때에는 일시정지하여야 하며, 진입하여서는 아니 된다."
        self.assertEqual(check_instruction(article, {"조건": "", "행동": "횡단보도로 진입하세요."}), ["부정어", "조건"])
        good = {"조건": "보행자가 횡단보도를 통행하고 있을 때", "행동": "일시정지하고 진입하지 마십시오."}
        self.assertEqual(check_instruction(article, good), [])


if __name__ == "__main__":
    unittest.main()
