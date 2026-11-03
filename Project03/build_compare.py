"""비교 모드에서 보여 줄 세 법령의 엣지 케이스별 조항과 지시문을 미리 만들어 JSON으로 저장하는 스크립트"""

import json

from edge import EDGE_CASES
from gemma import load_gemma, write_instruction
from retrieve import Retriever

CORPUS_PATHS = ["corpus-current.json", "corpus-2021.json", "corpus-california.json"]

llm = load_gemma()
compare = {}
for path in CORPUS_PATHS:
    retriever = Retriever(path)
    compare[path] = {}
    for name in EDGE_CASES:
        article = retriever.search(retriever.queries[name], 1)[0][1]
        instruction = write_instruction(llm, name, article)
        compare[path][name] = {"article": article, "instruction": instruction}
        print(path, name, instruction["행동 종류"], article["id"])

with open("compare-instructions.json", "w", encoding="utf-8") as file:
    json.dump(compare, file, ensure_ascii=False, indent=1)
