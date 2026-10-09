from functools import lru_cache

import tiktoken

LLM_ENCODING = "o200k_base"


@lru_cache
def llm_encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(LLM_ENCODING)


def count_llm_tokens(text: str) -> int:
    return len(llm_encoding().encode(text))
