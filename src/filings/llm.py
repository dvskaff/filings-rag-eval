"""Answer generation: prompts, a small open LLM run locally with llama.cpp, and citation parsing."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path

log = logging.getLogger(__name__)
IDK = "I don't know based on the provided filing."

RAG_SYSTEM = (
    "You are a careful financial analyst. Answer the question using ONLY the numbered excerpts from the company's "
    "SEC filing. Cite the excerpts you used in square brackets, like [1] or [2][3]. If the excerpts do not contain "
    f"the information needed, reply exactly: {IDK} Give the answer first (begin with Yes or No if it is a yes/no "
    "question), then at most two short sentences of explanation, including any calculation."
)
CLOSED_SYSTEM = (
    "You are a careful financial analyst. Answer the question from your own knowledge. If you do not know, reply "
    "exactly: I don't know. Give the answer first (begin with Yes or No if it is a yes/no question), then at most "
    "two short sentences of explanation."
)


def rag_messages(question: str, doc_name: str, hits) -> list[dict]:
    excerpts = "\n\n".join(f"[{i}] (page {h.page + 1})\n{h.text}" for i, h in enumerate(hits, start=1))
    user = f"Filing: {doc_name}\n\nExcerpts:\n{excerpts}\n\nQuestion: {question}"
    return [{"role": "system", "content": RAG_SYSTEM}, {"role": "user", "content": user}]


def closed_messages(question: str) -> list[dict]:
    return [{"role": "system", "content": CLOSED_SYSTEM}, {"role": "user", "content": question}]


def citations(answer: str, n_hits: int) -> list[int]:
    """Valid excerpt numbers cited in the answer, in order of appearance."""
    out = []
    for m in re.finditer(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer):
        for x in m.group(1).split(","):
            i = int(x)
            if 1 <= i <= n_hits and i not in out:
                out.append(i)
    return out


class FakeLLM:
    """Offline stand-in: quotes the first excerpt line containing a number, or says it doesn't know."""

    name = "fake-llm (test stand-in)"

    def chat(self, messages: list[dict], max_tokens: int = 256) -> str:
        user = messages[-1]["content"]
        if "Excerpts:" not in user:
            return "I don't know."
        filing = user.split("\n", 1)[0].removeprefix("Filing: ").split("_")[0].lower()
        question = user.rsplit("Question:", 1)[-1].lower()
        if filing and filing not in question.replace(" ", "").replace("&", ""):
            return IDK
        for line in user.split("\n"):
            if re.search(r"\d", line) and not line.startswith(("[", "Filing")):
                return f"{line.strip()} [1]"
        return IDK


class LlamaCppLLM:
    """A GGUF model from Hugging Face, run on CPU with llama.cpp (deterministic: temperature 0)."""

    def __init__(self, repo: str, filename: str, name: str, n_ctx: int = 4096):
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama

        path = os.environ.get("RAG_LLM_PATH") or hf_hub_download(repo_id=repo, filename=filename)
        self.name = name
        self.llm = Llama(model_path=path, n_ctx=n_ctx, n_threads=os.cpu_count(), seed=0, verbose=False)

    def chat(self, messages: list[dict], max_tokens: int = 256) -> str:
        out = self.llm.create_chat_completion(
            messages=messages, max_tokens=max_tokens, temperature=0.0, top_p=1.0
        )
        return out["choices"][0]["message"]["content"].strip()


def load_llm(cfg):
    if cfg.backend == "fake":
        return FakeLLM()
    return LlamaCppLLM(cfg.llm_repo, cfg.llm_file, cfg.llm_name)


class CachedLLM:
    """Remembers every answer by (model, prompt), so re-running an unchanged evaluation costs nothing."""

    def __init__(self, llm, path: Path):
        self.llm, self.path, self.name = llm, path, llm.name
        self.cache: dict[str, str] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                self.cache[r["key"]] = r["answer"]
        self.hits = self.misses = 0

    def chat(self, messages: list[dict], max_tokens: int = 256) -> str:
        key = hashlib.sha256(json.dumps([self.name, messages, max_tokens]).encode()).hexdigest()
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        answer = self.llm.chat(messages, max_tokens)
        self.cache[key] = answer
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps({"key": key, "answer": answer}) + "\n")
        return answer
