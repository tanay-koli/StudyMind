"""
StudyMind — Query rewriting: multi-query paraphrases fused with RRF, or HyDE
(retrieve with a hypothetical answer, rerank against the real question).
"""

import asyncio
import logging
import re
from typing import Optional

from core.llm import llm
from core.retrieval import Hit, Retriever, rrf

log = logging.getLogger(__name__)

PARAPHRASE_PROMPT = "Rewrite this question 3 different ways for searching study notes. One per line, nothing else."
HYDE_PROMPT = "Write a short passage (3-4 sentences) from study notes that would answer this question."


async def paraphrases(question: str) -> list[str]:
    try:
        text = await llm.complete(question, system=PARAPHRASE_PROMPT)
    except Exception:
        log.warning("paraphrase generation failed", exc_info=True)
        return []
    seen = {question.strip().lower()}
    out: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", line).strip()
        if line and line.lower() not in seen:
            seen.add(line.lower())
            out.append(line)
    return out[:3]


async def hypothetical_answer(question: str) -> str:
    try:
        return (await llm.complete(question, system=HYDE_PROMPT)).strip()
    except Exception:
        log.warning("hypothetical answer failed", exc_info=True)
        return ""


class RewritingRetriever:
    def __init__(self, base: Retriever, mode: str):
        self.base = base
        self.mode = mode

    async def retrieve(self, query: str, topic_id: Optional[str], k: int = 6,
                       rerank_query: Optional[str] = None) -> list[Hit]:
        if self.mode == "hyde":
            answer = await hypothetical_answer(query)
            if answer:
                return await self.base.retrieve(answer, topic_id, k, rerank_query=query)
        elif self.mode == "multi":
            queries = await paraphrases(query)
            if queries:
                queries.insert(0, query)
                lists = await asyncio.gather(*(self.base.retrieve(q, topic_id, k) for q in queries))
                hits: dict[int, Hit] = {}
                for lst in lists:
                    for h in lst:
                        hits.setdefault(h.chunk_id, h)
                fused = rrf([[h.chunk_id for h in lst] for lst in lists])
                return [hits[i] for i in fused][:k]
        return await self.base.retrieve(query, topic_id, k, rerank_query=rerank_query)
