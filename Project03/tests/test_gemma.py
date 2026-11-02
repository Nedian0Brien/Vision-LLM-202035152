"""Gemma 답 문자열을 읽는 부분을 모델 없이 확인하는 테스트"""

import unittest

from gemma import parse_answer


class ParseAnswerTest(unittest.TestCase):
    def test_reads_item_lines(self):
        text = "교차로: 예\n횡단보도: 아니오\n표지판 글자(없으면 없음): PUBLIC PARKING"
        self.assertEqual(parse_answer(text), {"교차로": "예", "횡단보도": "아니오", "표지판 글자": "PUBLIC PARKING"})

    def test_strips_bold_marks_and_ignores_other_lines(self):
        text = "**교차로**: 예\n설명 없는 줄\n**횡단보도:** 예"
        self.assertEqual(parse_answer(text), {"교차로": "예", "횡단보도": "예"})


if __name__ == "__main__":
    unittest.main()
