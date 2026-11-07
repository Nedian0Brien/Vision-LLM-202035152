"""상황 번호 파싱, 법령 인덱스, 크롭, 판단 유지를 확인하는 테스트"""

import threading
import time
import unittest

import numpy as np

import gemma
import voice
from detect import LIGHT_CLASSES, class_strip, interpolate_boxes, road_crop
from gemma import CROP_SITUATIONS, SITUATIONS, load_law, parse_situations


# 시험용 모델에는 점수가 없고 확률은 화면 표시에만 쓰임
gemma.answer_odds = lambda llm, words: []


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
        self.assertEqual(current[gemma.SCHOOL_CROSSWALK]["id"], "도로교통법 제27조 ⑦")
        # 2006년 법에는 보호구역 횡단보도 조문이 없음
        self.assertEqual(load_law("2006")[gemma.SCHOOL_CROSSWALK]["id"], "도로교통법 제12조 ③")
        # 캘리포니아 통학버스는 정차해 적색 점멸등을 켠 버스의 정지 의무를 기본 조문으로 둠(2026-11-07 사용자 지시)
        self.assertEqual(california[gemma.SCHOOL_BUS]["id"], "CVC §22454(a)")
        self.assertEqual(set(load_law(gemma.NO_LAW)), set(SITUATIONS + CROP_SITUATIONS))
        self.assertEqual(len(current), len(SITUATIONS) + len(CROP_SITUATIONS))

    def test_class_strip_uses_only_given_classes(self):
        frame = np.zeros((720, 1280, 3), np.uint8)
        self.assertEqual(class_strip(frame, Result(Boxes), LIGHT_CLASSES).shape[0], 400)
        self.assertIsNone(class_strip(frame, Result(Boxes), [11]))
        self.assertIsNone(class_strip(frame, Result(EmptyBoxes), LIGHT_CLASSES))

    def test_road_crop_has_expected_size(self):
        frame = np.zeros((720, 1280, 3), np.uint8)
        self.assertEqual(road_crop(frame).shape[:2], (266, 896))


class RecognizeTest(unittest.TestCase):
    def run_recognize(self, boxes, words, choice, in_school_zone=False, free="none"):
        """정해 둔 답을 내는 시험용 모델로 recognize를 부르는 메서드"""
        answers = list(words)

        class Llm:
            """객관식과 낱말 답을 정해 둔 시험용 클래스"""

            n_tokens = 0

            def tokenize(self, text, add_bos):
                return [text[0]]

            def create_chat_completion(self, messages, max_tokens, temperature, logit_bias=None):
                if messages[0]["content"][-1]["text"] == gemma.FREE_PROMPT:
                    return {"choices": [{"message": {"content": free}}]}
                return {"choices": [{"message": {"content": answers.pop(0) if logit_bias else choice}}]}

        return gemma.recognize(Llm(), np.zeros((720, 1280, 3), np.uint8), Result(boxes), in_school_zone)

    def test_light_answer_decides_signal(self):
        self.assertEqual(self.run_recognize(Boxes, ["red", "no"], "2, 14"), [SITUATIONS[0], SITUATIONS[13]])
        self.assertEqual(self.run_recognize(Boxes, ["yellow", "no"], "0"), [SITUATIONS[1]])
        self.assertEqual(self.run_recognize(Boxes, ["green", "no"], "1"), [])

    def test_free_phrases_become_one_situation(self):
        found = self.run_recognize(EmptyBoxes, ["no"], "0", free="Inside a road tunnel; speed limit sign 40.; none")
        self.assertEqual(found, ["inside a road tunnel; speed limit sign 40"])
        self.assertTrue(gemma.is_free(found[0]))
        self.assertFalse(gemma.is_free(SITUATIONS[0]) or gemma.is_free(CROP_SITUATIONS[0]))

    def test_no_boxes_skip_crop_calls(self):
        self.assertEqual(self.run_recognize(EmptyBoxes, ["no"], "14"), [SITUATIONS[13]])
        self.assertNotIn("light", gemma.STATUS["answers"])

    def test_pedestrian_needs_crosswalk(self):
        class PersonBoxes:
            """사람 박스 하나를 담은 시험용 클래스"""

            xyxy = np.array([[100.0, 100.0, 150.0, 200.0]])
            cls = np.array([0.0])

        self.assertEqual(self.run_recognize(PersonBoxes, ["yes", "yes", "no"], "0"), [SITUATIONS[7]])
        self.assertEqual(self.run_recognize(PersonBoxes, ["yes", "no", "no"], "0"), [])
        self.assertEqual(self.run_recognize(PersonBoxes, ["no", "no"], "8"), [])

    def test_rain_answer(self):
        self.assertEqual(self.run_recognize(EmptyBoxes, ["yes"], "0"), [gemma.RAIN])
        self.assertEqual(self.run_recognize(EmptyBoxes, ["no"], "0"), [])

    def test_bus_and_hydrant_need_boxes(self):
        class VehicleBoxes:
            """버스와 소화전 박스를 담은 시험용 클래스"""

            xyxy = np.array([[100.0, 100.0, 300.0, 300.0], [600.0, 300.0, 650.0, 400.0]])
            cls = np.array([5.0, 10.0])

        self.assertEqual(self.run_recognize(VehicleBoxes, ["no", "yes", "hyd"], "0"), [gemma.SCHOOL_BUS, gemma.HYDRANT])
        self.assertEqual(self.run_recognize(VehicleBoxes, ["no", "no", "cone"], "0"), [])

    def test_school_crosswalk_needs_zone_and_no_light(self):
        self.assertEqual(self.run_recognize(EmptyBoxes, ["no", "yes"], "0", True), [gemma.SCHOOL_CROSSWALK])
        self.assertEqual(self.run_recognize(EmptyBoxes, ["no", "yes"], "15"), [gemma.SCHOOL_ZONE, gemma.SCHOOL_CROSSWALK])
        self.assertEqual(self.run_recognize(EmptyBoxes, ["no"], "0"), [])
        self.assertEqual(self.run_recognize(Boxes, ["green", "no"], "0", True), [])


class SkipTest(unittest.TestCase):
    def test_ask_word_skips_old_clip(self):
        self.addCleanup(gemma.WATCHING.clear)
        gemma.WATCHING["shared"] = {"clip": {"new": True}}
        gemma.WATCHING["clip"] = {"old": True}
        # llm이 None이라 호출하면 실패하므로 빈 답은 호출하지 않았다는 뜻
        self.assertEqual(gemma.ask_word(None, "rain", None, "", ["yes", "no"]), "")


class ActionTest(unittest.TestCase):
    def test_read_provision_returns_action_code(self):
        class Llm:
            """토큰화와 생성 결과를 정해 둔 시험용 클래스"""

            n_tokens = 0

            def tokenize(self, text, add_bos):
                return [text[0]]

            def create_completion(self, prompt, **kwargs):
                return {"choices": [{"text": " Stop" if kwargs["max_tokens"] == 1 else " **적색 신호 앞에서 정지하세요.**"}]}

        article = {"id": "제5조", "title": "", "text": "조문", "references": []}
        rule = gemma.read_provision(Llm(), SITUATIONS[0], article)
        self.assertEqual(rule, {"law": "제5조", "action": "STOP", "instruction": "적색 신호 앞에서 정지하세요.", "text": "조문"})

    def test_read_provision_keeps_reference_text_uncut(self):
        prompts = []

        class Llm:
            """받은 프롬프트를 모아 두는 시험용 클래스"""

            n_tokens = 0

            def tokenize(self, text, add_bos):
                return [text[0]]

            def create_completion(self, prompt, **kwargs):
                prompts.append(prompt)
                return {"choices": [{"text": " Stop" if kwargs["max_tokens"] == 1 else "정지"}]}

        reference = {"id": "제1항", "title": "", "text": "시속 30"}
        article = {"id": "제3항", "title": "제목", "text": "가" * 300, "references": [reference]}
        rule = gemma.read_provision(Llm(), SITUATIONS[0], article)
        self.assertEqual(rule["law"], "제3항")
        self.assertEqual(rule["text"], "제목: " + "가" * (gemma.READ_CHARS - 4) + " 시속 30")
        self.assertIn("시속 30", prompts[-1])

    def test_read_provision_without_article_asks_without_law(self):
        prompts = []

        class Llm:
            """받은 프롬프트를 모아 두는 시험용 클래스"""

            n_tokens = 0

            def tokenize(self, text, add_bos):
                return [text[0]]

            def create_completion(self, prompt, **kwargs):
                prompts.append(prompt)
                return {"choices": [{"text": " Caution" if kwargs["max_tokens"] == 1 else "**멈추세요.** (Stop.)"}]}

        rule = gemma.read_provision(Llm(), SITUATIONS[0], gemma.NO_ARTICLE)
        self.assertEqual(rule, {"law": "조문 없음", "action": "CAUTION", "instruction": "멈추세요.", "text": ""})
        self.assertNotIn("Traffic law text", prompts[-1])

    def test_action_codes_are_distinct(self):
        self.assertEqual(len(set(gemma.ACTIONS.values())), len(gemma.ACTIONS))


class InterpolateTest(unittest.TestCase):
    def test_matched_box_moves_between_results(self):
        before = [([0.0, 0.0, 100.0, 100.0], 2.0, 0.9)]
        after = [([20.0, 0.0, 120.0, 100.0], 2.0, 0.8)]
        self.assertEqual(interpolate_boxes(before, after, 0.5)[0][0], [10.0, 0.0, 110.0, 100.0])

    def test_unmatched_box_follows_nearer_result(self):
        before = [([0.0, 0.0, 10.0, 10.0], 2.0, 0.9)]
        after = [([50.0, 50.0, 60.0, 60.0], 2.0, 0.8)]
        self.assertEqual(interpolate_boxes(before, after, 0.25), before)
        self.assertEqual(interpolate_boxes(before, after, 0.75), after)

    def test_other_class_does_not_match(self):
        before = [([0.0, 0.0, 100.0, 100.0], 2.0, 0.9)]
        after = [([0.0, 0.0, 100.0, 100.0], 0.0, 0.8)]
        self.assertEqual(interpolate_boxes(before, after, 0.25), before)


class QuestionTest(unittest.TestCase):
    def test_search_returns_closest_articles(self):
        class Embedder:
            """질문을 고정 벡터로 바꾸는 시험용 클래스"""

            def embed(self, text, normalize):
                return [1.0, 0.0]

        corpus = {"articles": [{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}], "vectors": np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.9, 0.1]], dtype=np.float32)}
        self.assertEqual([article["id"] for article in gemma.search_law(Embedder(), "질문", corpus)], ["b", "d", "c"])

    def test_no_law_answers_without_search(self):
        class Model:
            """프롬프트를 저장하고 고정 답을 내는 시험용 클래스"""

            def create_completion(self, prompt, **kwargs):
                self.prompt = prompt
                return {"choices": [{"text": " **답** "}]}

        model = Model()
        result = gemma.answer_question(model, None, "질문", {"articles": [], "vectors": None})
        self.assertEqual((result["answer"], result["articles"]), ("답", []))
        self.assertNotIn("Traffic law text", model.prompt)


class Recorder:
    """녹음 프로세스 대신 끝난 값을 돌려주고 멈춤 요청을 기록하는 시험용 클래스"""

    def __init__(self, code, stop=False):
        self.code = code
        self.stop = stop
        self.terminated = False

    def wait(self):
        """녹음 중에 다시 누른 것처럼 멈춤을 요청하고 끝난 값을 돌려주는 메서드"""
        if self.stop:
            voice.stop_listening(self.shared)
        return self.code

    def terminate(self):
        """멈춤 요청을 기록하는 메서드"""
        self.terminated = True


class VoiceTest(unittest.TestCase):
    def listen_with(self, recorder, question):
        """가짜 녹음기와 가짜 Whisper로 listen을 돌리는 메서드"""
        originals = voice.subprocess.Popen, voice.speech_to_text
        self.addCleanup(lambda: (setattr(voice.subprocess, "Popen", originals[0]), setattr(voice, "speech_to_text", originals[1])))
        voice.subprocess.Popen = lambda command: recorder
        voice.speech_to_text = lambda: question
        shared = {"listening": False, "speaking": False, "question": None, "voice_error": None, "recorder": None,
                  "listen_start": 0.0, "transcribing": False, "stop_requested": False}
        recorder.shared = shared
        voice.listen(shared)
        return shared

    def test_listen_puts_question(self):
        shared = self.listen_with(Recorder(0), "소화전 앞에 세워도 되나요")
        self.assertEqual((shared["question"], shared["listening"], shared["recorder"]), ("소화전 앞에 세워도 되나요", False, None))

    def test_listen_reports_missing_microphone(self):
        shared = self.listen_with(Recorder(1), "들리지 않은 질문")
        self.assertEqual((shared["listening"], shared["question"], shared["voice_error"]), (False, None, "마이크를 쓸 수 없습니다"))

    def test_stop_uses_recording_so_far(self):
        recorder = Recorder(-15, stop=True)
        shared = self.listen_with(recorder, "비 올 때 전조등을 켜야 하나요")
        self.assertTrue(recorder.terminated)
        self.assertEqual((shared["question"], shared["voice_error"]), ("비 올 때 전조등을 켜야 하나요", None))


def rule(name):
    """가짜 요약 규칙을 만드는 함수"""
    return {"law": name, "title": "", "instruction": name, "text": "조문 " + name}


class FindRulesTest(unittest.TestCase):
    def test_build_pick_comes_first_and_gemma_picks_are_added(self):
        def summary(name, vector):
            return {"id": name, "title": "", "text": "조문 " + name, "action": "CAUTION", "instruction": name, "instruction_en": name, "scene": name}, vector

        items = [summary("x", [1.0, 0.0]), summary("y", [0.9, 0.1]), summary("z", [0.0, 1.0])]
        corpus = {"rules": [item[0] for item in items], "rule_vectors": np.array([item[1] for item in items], np.float32)}

        class Embedder:
            def embed(self, text, normalize):
                return [1.0, 0.0]

        class Llm:
            n_tokens = 0

            def create_chat_completion(self, messages, **kwargs):
                return {"choices": [{"message": {"content": "1"}}]}

        found = gemma.find_rules(Llm(), Embedder(), np.zeros((720, 1280, 3), np.uint8), "situation", {"id": "z"}, corpus)
        self.assertEqual([rule["law"] for rule in found], ["z", "x"])

    def test_free_phrases_search_each_phrase_without_excluded_rules(self):
        names = ["x", "y", "z"]
        rules = [{"id": name, "title": "", "text": "", "action": "CAUTION", "instruction": name, "instruction_en": name, "scene": name, "general": name == "x"} for name in names]
        corpus = {"rules": rules, "rule_vectors": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], np.float32)}
        searched = []

        class Embedder:
            def embed(self, text, normalize):
                searched.append(text)
                return [1.0, 0.9, 0.0] if text.endswith("p") else [0.0, 0.0, 1.0]

        class Llm:
            n_tokens = 0
            prompt = None

            def create_chat_completion(self, messages, **kwargs):
                Llm.prompt = messages[0]["content"][-1]["text"]
                return {"choices": [{"message": {"content": "1, 2"}}]}

        original = gemma.RULE_TOP
        gemma.RULE_TOP = 1
        self.addCleanup(lambda: setattr(gemma, "RULE_TOP", original))
        found = gemma.find_rules(Llm(), Embedder(), np.zeros((720, 1280, 3), np.uint8), "p; q", {"id": None}, corpus, {("x", "x")})
        self.assertEqual(len(searched), 2)
        self.assertEqual([rule["law"] for rule in found], ["y", "z"])
        self.assertIn("this image: p; q.", Llm.prompt)

    def test_free_situation_skips_general_and_taken_rules(self):
        self.addCleanup(gemma.WATCHING.clear)
        rules = [{"id": name, "instruction": name, "general": name == "g"} for name in ("g", "t", "n")]
        clip = {"last_seen": {SITUATIONS[0]: time.perf_counter()}}
        gemma.WATCHING.update({"clip": clip, "llm": None, "embedder": None, "frame": None, "laws": {"current": {}}, "corpora": {"current": {"rules": rules}},
                               "readings": {("current", SITUATIONS[0]): [{"law": "t", "instruction": "t"}]}})
        calls = []
        original = gemma.find_rules
        gemma.find_rules = lambda llm, embedder, frame, situation, article, corpus, exclude=(): calls.append((article, exclude)) or []
        self.addCleanup(lambda: setattr(gemma, "find_rules", original))
        gemma.find_law_rules("current", "a red fire truck")
        self.assertEqual(calls, [({"id": None}, {("g", "g"), ("t", "t")})])


class PublishTest(unittest.TestCase):
    def setUp(self):
        original = gemma.find_rules
        gemma.find_rules = lambda llm, embedder, frame, situation, article, corpus: [rule(situation)]
        self.addCleanup(lambda: setattr(gemma, "find_rules", original))
        self.addCleanup(gemma.WATCHING.clear)
        # 시험용 상황 a, b를 목록 상황으로 다룸
        free = gemma.is_free
        gemma.is_free = lambda situation: False
        self.addCleanup(lambda: setattr(gemma, "is_free", free))
        self.clip = {"last_seen": {}, "rules": []}
        laws = {"current": {"a": {"text": "조문 a"}, "b": {"text": "조문 b"}}}
        gemma.WATCHING.update({"shared": {"clip": self.clip}, "clip": self.clip, "law": "current", "laws": laws, "llm": None, "embedder": None, "corpora": {"current": None}, "frame": None,
                               "readings": {("current", "a"): [rule("a")]}})

    def test_new_rule_is_read_and_put_last(self):
        gemma.publish(["b", "a"])
        self.assertEqual([rule["law"] for rule in self.clip["rules"]], ["a", "b"])

    def test_old_clip_is_not_changed(self):
        gemma.WATCHING["shared"] = {"clip": {"last_seen": {}, "rules": []}}
        gemma.publish(["b"])
        self.assertEqual(self.clip["rules"], [])


class WatchTest(unittest.TestCase):
    def run_watch(self, answers):
        """가짜 인식 결과를 한 사이클씩 내주며 watch를 돌리는 메서드"""
        self.go = threading.Semaphore(0)
        self.started = threading.Semaphore(0)
        remaining = list(answers)

        def recognize(llm, frame, result, in_school_zone):
            self.started.release()
            self.go.acquire()
            # 답을 다 내면 SystemExit로 watch 스레드를 끝냄
            if not remaining:
                raise SystemExit
            return remaining.pop(0)

        original = gemma.recognize, gemma.find_rules
        gemma.recognize = recognize
        gemma.find_rules = lambda llm, embedder, frame, situation, article, corpus: [rule(situation)]
        self.addCleanup(lambda: (setattr(gemma, "recognize", original[0]), setattr(gemma, "find_rules", original[1])))
        # 시험용 상황 a, b를 목록 상황으로 다룸
        free = gemma.is_free
        gemma.is_free = lambda situation: False
        self.addCleanup(lambda: setattr(gemma, "is_free", free))
        self.shared = {"question": None, "input": ("frame", "result"), "law": "current", "clip": {"last_seen": {}, "rules": []}}
        self.thread = threading.Thread(target=gemma.watch, args=(None, {"current": {"a": {"text": "조문 a"}, "b": {"text": "조문 b"}}}, self.shared, None, {"current": None}), daemon=True)
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
        self.wait_rules(clip, [rule("a")])
        self.started.acquire()
        self.go.release()
        self.wait_rules(clip, [rule("a"), rule("b")])

    def test_compare_shows_pending_before_rule(self):
        self.run_watch([["a"]])
        clip = self.shared["clip"]
        reading = threading.Event()

        def find(llm, embedder, frame, situation, article, corpus):
            reading.wait(2)
            return [rule(situation)]

        gemma.find_rules = find
        self.started.acquire()
        self.go.release()
        for _ in range(200):
            if clip.get("compare"):
                break
            time.sleep(0.01)
        self.assertEqual(clip["compare"]["current"], {"rules": [], "pending": True})
        reading.set()
        self.wait_rules(clip, [rule("a")])
        for _ in range(200):
            if not clip["compare"]["current"]["pending"]:
                break
            time.sleep(0.01)
        self.assertEqual(clip["compare"]["current"], {"rules": [rule("a")], "pending": False})

    def test_rule_disappears_after_hold(self):
        original = gemma.HOLD_SECONDS
        gemma.HOLD_SECONDS = 0.01
        self.addCleanup(lambda: setattr(gemma, "HOLD_SECONDS", original))
        self.run_watch([["a"], []])
        clip = self.shared["clip"]
        self.started.acquire()
        self.go.release()
        self.wait_rules(clip, [rule("a")])
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
        # 버려진 묶음은 조문을 읽지 않아 규칙이 생기지 않음
        for _ in range(200):
            if "clip" not in gemma.WATCHING:
                break
            time.sleep(0.01)
        self.assertEqual(old_clip["rules"], [])
        self.assertEqual(new_clip, {"last_seen": {}, "rules": []})
        self.shared["input"] = ("frame", "result")
        self.started.acquire()
        self.go.release()
        self.wait_rules(new_clip, [rule("b")])


if __name__ == "__main__":
    unittest.main()
