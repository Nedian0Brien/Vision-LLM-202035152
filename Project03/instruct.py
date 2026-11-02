"""검색된 조항으로 로컬 Gemma가 주행 지시문을 만드는 모듈"""

from llama_cpp import Llama

LLM_PATH = "../src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
# "40자 이내 한 문장"으로 지시하면 조항의 첫 문장만 요약해 개정된 뒷부분이 빠졌음
SYSTEM_PROMPT = (
    "너는 자율주행 차량의 주행 규칙 안내기다. 주어진 법령 조항을 근거로, 지금 장면에서 운전자가 해야 할 행동을 "
    "한국어 명령문 두 문장 이내로 쓴다. 조항에 적힌 행동 조건을 모두 담고, 조항에 없는 내용은 쓰지 않는다."
)


def write_instructions(articles):
    """상태마다 검색된 조항을 Gemma에 넣어 주행 지시문 사전을 만드는 함수"""
    # 05 노트북의 로컬 Gemma 설정과 같음
    llm = Llama(model_path=LLM_PATH, n_gpu_layers=-1, n_ctx=2048, n_batch=32, n_ubatch=32, verbose=False)
    instructions = {}
    for state, article in articles.items():
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"장면: {state}\n조항: {article['id']} {article['title']}\n{article['text']}"},
        ]
        # temperature 0은 같은 조항에 늘 같은 지시문이 나오게 함
        response = llm.create_chat_completion(messages=messages, max_tokens=120, temperature=0)
        instructions[state] = response["choices"][0]["message"]["content"].strip()
        print(state, instructions[state])
    return instructions
