"""상황 번호 파싱, 법령 인덱스, 크롭 대체 영역을 확인하는 테스트"""

import unittest

import numpy as np

from detect import road_crop, sign_strip
from gemma import SITUATIONS, load_law, parse_situations


class Boxes:
    """YOLO 결과의 boxes 자리에 쓰는 시험용 클래스"""

    xyxy = np.array([[100.0, 100.0, 110.0, 120.0], [300.0, 300.0, 400.0, 400.0]])
    cls = np.array([9.0, 2.0])


class EmptyBoxes:
    """탐지가 없는 boxes 자리에 쓰는 시험용 클래스"""

    xyxy = np.zeros((0, 4))
    cls = np.zeros(0)


class Result:
    """YOLO 결과 자리에 쓰는 시험용 클래스"""

    def __init__(self, boxes):
        self.boxes = boxes


class GemmaTest(unittest.TestCase):
    def test_parse_keeps_valid_numbers_once(self):
        self.assertEqual(parse_situations("1, 8"), [SITUATIONS[0], SITUATIONS[7]])
        self.assertEqual(parse_situations("10, 10, 99"), [SITUATIONS[9]])
        self.assertEqual(parse_situations("0"), [])
        self.assertEqual(parse_situations("none"), [])

    def test_law_index_points_to_expected_articles(self):
        current = load_law("current")
        california = load_law("california")
        self.assertEqual(current[SITUATIONS[0]]["id"], "시행규칙 별표 2")
        self.assertEqual(california[SITUATIONS[0]]["id"], "CVC §21453(a)")
        # 캘리포니아에는 터널 등화 조문을 찾지 못해 지시문을 내지 않음
        self.assertNotIn(SITUATIONS[12], california)
        self.assertEqual(len(current), len(SITUATIONS))

    def test_sign_strip_uses_only_sign_boxes_and_falls_back(self):
        frame = np.zeros((720, 1280, 3), np.uint8)
        self.assertEqual(sign_strip(frame, Result(Boxes)).shape[0], 400)
        self.assertEqual(sign_strip(frame, Result(EmptyBoxes)).shape[:2], (360, 704))
        self.assertEqual(road_crop(frame).shape[:2], (266, 896))


if __name__ == "__main__":
    unittest.main()
