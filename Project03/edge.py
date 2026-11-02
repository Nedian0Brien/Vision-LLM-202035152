"""법 조문이 필요한 주행 엣지 케이스의 목록과 발동 판정, 지시문 검사를 담은 모듈"""

ACTIONS = ["정지", "양보", "서행", "진입·회전 금지", "앞지르기·차로 변경 금지", "주의 진행"]

# 엣지 케이스마다 함께 있어야 하는 사실과 있으면 안 되는 사실
EDGE_CASES = {
    "적색 신호 교차로": {"needs": {"적색 신호"}, "excludes": set()},
    "황색 신호": {"needs": {"황색 신호"}, "excludes": set()},
    "횡단보도 보행자": {"needs": {"보행자", "횡단보도"}, "excludes": set()},
    "차도 보행자": {"needs": {"보행자"}, "excludes": {"횡단보도"}},
    "정지 표지": {"needs": {"정지 표지"}, "excludes": set()},
    "양보 표지": {"needs": {"양보 표지"}, "excludes": set()},
    "회전 금지 표지": {"needs": {"회전 금지 표지"}, "excludes": set()},
    "버스 전용차로": {"needs": {"버스 전용차로"}, "excludes": set()},
    "노면전차 선로": {"needs": {"노면전차 선로"}, "excludes": set()},
    "터널": {"needs": {"터널"}, "excludes": set()},
    "공사 구간": {"needs": {"공사 구간"}, "excludes": set()},
    "자전거 근접": {"needs": {"자전거"}, "excludes": set()},
    "통학버스": {"needs": {"통학버스"}, "excludes": set()},
    "긴급자동차": {"needs": {"긴급자동차"}, "excludes": set()},
    "철길 건널목": {"needs": {"철길 건널목"}, "excludes": set()},
    "회전교차로": {"needs": {"회전교차로"}, "excludes": set()},
    "어린이보호구역": {"needs": {"어린이보호구역"}, "excludes": set()},
}

# 원문에 이 낱말이 있으면 지시문에도 같은 성격의 낱말이 있어야 뜻이 뒤집히지 않음
NEGATION_WORDS = ["아니 된다", "아니된다", "못한다", "없다", "금지", "마십시오", "마세요", "말고", "안 됩니다", "안됩니다", "않",
                  "shall not", "may not", "no person", "unlawful", "prohibit"]
CONDITION_WORDS = ["때에는", "경우에는", "경우", "때", "when", "if ", "unless", "except", "while", "upon"]


def active_edge_cases(facts):
    """사실 집합에서 발동 조건을 만족하는 엣지 케이스 이름 목록을 만드는 함수"""
    return [name for name, case in EDGE_CASES.items() if case["needs"] <= facts and not case["excludes"] & facts]


def has_any(text, words):
    """글에 낱말 목록 중 하나라도 들어 있는지 확인하는 함수"""
    lowered = text.lower()
    return any(word in lowered for word in words)


def check_instruction(article_text, fields):
    """원문의 부정어와 조건이 지시문 칸에도 남아 있는지 검사해 빠진 성격의 목록을 돌려주는 함수"""
    instruction = " ".join(fields.values())
    missing = []
    if has_any(article_text, NEGATION_WORDS) and not has_any(instruction, NEGATION_WORDS):
        missing.append("부정어")
    if has_any(article_text, CONDITION_WORDS) and not fields.get("조건", "").strip(" -없음"):
        missing.append("조건")
    return missing
