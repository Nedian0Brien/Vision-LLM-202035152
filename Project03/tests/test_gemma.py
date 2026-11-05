"""상황 번호 파싱, 법령 인덱스, 크롭, 판단 유지를 확인하는 테스트"""

import threading
import time
import unittest

import numpy as np

import gemma
from detect import LIGHT_CLASSES, class_strip, road_crop, upper_left_crop
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
        self.assertEqual(current[SITUATIONS[11]]["id"], "도로교통법 제22조 ③")
        # 캘리포니아에는 터널 등화 조문을 찾지 못해 지시문을 내지 않음
        self.assertNotIn(SITUATIONS[12], california)
        self.assertEqual(len(current), len(SITUATIONS))

    def test_class_strip_uses_only_given_classes(self):
        frame = np.zeros((720, 1280, 3), np.uint8)
        self.assertEqual(class_strip(frame, Result(Boxes), LIGHT_CLASSES).shape[0], 400)
        self.assertIsNone(class_strip(frame, Result(Boxes), [11]))
        self.assertIsNone(class_strip(frame, Result(EmptyBoxes), LIGHT_CLASSES))

    def test_fixed_crops_have_expected_size(self):
        frame = np.zeros((720, 1280, 3), np.uint8)
        self.assertEqual(upper_left_crop(frame).shape[:2], (360, 704))
        self.assertEqual(road_crop(frame).shape[:2], (266, 896))


class ActionTest(unittest.TestCase):
    def test_read_provision_returns_action_code(self):
        class Llm:
            """토큰화와 생성 결과를 정해 둔 시험용 클래스"""

            def tokenize(self, text, add_bos):
                return [text[0]]

            def create_completion(self, prompt, **kwargs):
                return {"choices": [{"text": " Stop" if kwargs["max_tokens"] == 1 else " the red light"}]}

        article = {"id": "제5조", "title": "", "text": "조문"}
        rule = gemma.read_provision(Llm(), SITUATIONS[0], article)
        self.assertEqual(rule, {"law": "제5조", "action": "STOP", "instruction": "Stop for the red light", "text": "조문"})

    def test_action_codes_are_distinct(self):
        codes = [code for code, preposition in gemma.ACTIONS.values()]
        self.assertEqual(len(set(codes)), len(gemma.ACTIONS))


class WatchTest(unittest.TestCase):
    def run_watch(self, answers):
        """가짜 인식 결과를 한 사이클씩 내주며 watch를 돌리는 메서드"""
        self.go = threading.Semaphore(0)
        self.started = threading.Semaphore(0)
        remaining = list(answers)

        def recognize(llm, frame, result):
            self.started.release()
            self.go.acquire()
            # 답을 다 내면 SystemExit로 watch 스레드를 끝냄
            if not remaining:
                raise SystemExit
            return remaining.pop(0)

        original = gemma.recognize, gemma.read_provision
        gemma.recognize = recognize
        gemma.read_provision = lambda llm, situation, article: {"law": situation}
        self.addCleanup(lambda: (setattr(gemma, "recognize", original[0]), setattr(gemma, "read_provision", original[1])))
        self.shared = {"input": ("frame", "result"), "law": "current", "clip": {"last_seen": {}, "rules": []}}
        self.thread = threading.Thread(target=gemma.watch, args=(None, {"current": {"a": {}, "b": {}}}, self.shared), daemon=True)
        self.thread.start()
        self.addCleanup(self.finish)

    def finish(self):
        """남은 인식을 풀어 watch 스레드를 끝내는 메서드"""
        self.shared["input"] = ("frame", "result")
        for _ in range(5):
            self.go.release()
        self.thread.join(2)

    def wait_rules(self, clip, expected):
        """clip의 규칙 목록이 기대값이 될 때까지 기다리는 메서드"""
        for _ in range(200):
            if clip["rules"] == expected:
                return
            time.sleep(0.01)
        self.fail(clip["rules"])

    def test_rule_stays_after_situation_disappears(self):
        self.run_watch([["a"], ["b"]])
        clip = self.shared["clip"]
        self.started.acquire()
        self.go.release()
        self.wait_rules(clip, [{"law": "a"}])
        self.started.acquire()
        self.go.release()
        self.wait_rules(clip, [{"law": "a"}, {"law": "b"}])

    def test_rule_disappears_after_hold(self):
        original = gemma.HOLD_SECONDS
        gemma.HOLD_SECONDS = 0.01
        self.addCleanup(lambda: setattr(gemma, "HOLD_SECONDS", original))
        self.run_watch([["a"], []])
        clip = self.shared["clip"]
        self.started.acquire()
        self.go.release()
        self.wait_rules(clip, [{"law": "a"}])
        time.sleep(0.05)
        self.started.acquire()
        self.go.release()
        self.wait_rules(clip, [])

    def test_video_switch_discards_running_judgement(self):
        self.run_watch([["a"], ["b"]])
        old_clip = self.shared["clip"]
        self.started.acquire()
        # main.py가 영상을 바꿀 때와 같은 순서로 비움
        self.shared["input"] = None
        new_clip = {"last_seen": {}, "rules": []}
        self.shared["clip"] = new_clip
        self.go.release()
        self.wait_rules(old_clip, [{"law": "a"}])
        self.assertEqual(new_clip, {"last_seen": {}, "rules": []})
        self.shared["input"] = ("frame", "result")
        self.started.acquire()
        self.go.release()
        self.wait_rules(new_clip, [{"law": "b"}])


if __name__ == "__main__":
    unittest.main()
