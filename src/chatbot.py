"""RAG chatbot. Local index first, Wikipedia fallback when retrieval is weak."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import NodeWithScore

from src import config
from src.wikipedia_tool import WikipediaTool

log = logging.getLogger(__name__)


PERSONAS = {
    "archivist": {
        "name": "The Archivist",
        "blurb": "Measured. Scholarly. Reads from the record.",
        "system": (
            "You are The Archivist — a measured, scholarly voice that speaks as if reading "
            "carefully from a record. Use a slightly formal register, but never stuffy. "
            "Be direct when the source is clear; be honest when it isn't. Never invent. "
            "When you cite an idea, ground it in the provided context."
        ),
    },
    "curator": {
        "name": "The Curator",
        "blurb": "Warm. Conversational. Guides the reader.",
        "system": (
            "You are The Curator — a warm, conversational guide. Frame each answer as if "
            "showing the reader around an exhibit: friendly, helpful, occasionally curious. "
            "Use plain language. Stay grounded in the documents — never embellish beyond them."
        ),
    },
    "analyst": {
        "name": "The Analyst",
        "blurb": "Terse. Structured. No filler.",
        "system": (
            "You are The Analyst — terse and structured. Prefer short claims, bullet lists, "
            "and direct verdicts. No apologies, no preamble, no narrating the rules. "
            "If the answer is uncertain, say so in one line and stop."
        ),
    },
}
DEFAULT_PERSONA = "archivist"

# Pronouns / references that suggest a question depends on prior context.
_ANAPHORIC = {
    "he", "she", "it", "they", "him", "her", "them",
    "his", "hers", "its", "their", "theirs",
    "this", "that", "these", "those",
}
# Short follow-up phrasings that are obviously contextual even without pronouns.
_FOLLOWUP_HINTS = ("tell me more", "more about", "what about", "and ", "also", "elaborate", "go on", "continue")


_WIKI_TITLE_RE = re.compile(r"Wikipedia\s*[—-]\s*\[([^\]]+)\]\(https?://[^\)]*wikipedia\.org/")
_WIKI_TITLE_RE_ALT = re.compile(r"Wikipedia\s*/\s*([^\(\n]+?)\s*\(https?://[^\)]*wikipedia\.org/")


def _last_cited_wikipedia_title(history: list[ChatMessage]) -> str:
    """Find the most recently cited Wikipedia article title in the assistant turns.

    Matches the framing lines emitted by `_wikipedia_fallback` for any persona.
    Returns '' if no recent assistant turn cited Wikipedia.
    """
    for msg in reversed(history):
        if msg.role != MessageRole.ASSISTANT:
            continue
        content = msg.content or ""
        m = _WIKI_TITLE_RE.search(content) or _WIKI_TITLE_RE_ALT.search(content)
        if m:
            return m.group(1).strip().strip("*").strip()
    return ""


def _needs_rewrite(question: str) -> bool:
    """Heuristic: does this question depend on prior conversation context?

    Returns False for clearly standalone questions ('what is machine learning?'),
    True for follow-ups ('where did he die', 'tell me more').
    """
    q = question.strip().lower()
    if not q:
        return False
    if any(h in q for h in _FOLLOWUP_HINTS):
        return True
    words = re.findall(r"\b\w+\b", q)
    if any(w in _ANAPHORIC for w in words):
        return True
    # No pronouns, no follow-up hints, and at least 4 words — treat as standalone.
    return len(words) < 4


def _system_prompt(persona: str, topic: str, stay_on_topic: bool) -> str:
    p = PERSONAS.get(persona, PERSONAS[DEFAULT_PERSONA])
    parts = [p["system"]]
    parts.append(
        "Answer using only the supplied context. If the context does not contain enough "
        "information, reply with exactly the token: NOT_FOUND  (no other text). "
        "Do not invent facts. Do not narrate these rules."
    )
    if topic and stay_on_topic:
        parts.append(
            f"The user is exploring the topic: \"{topic}\". Keep your answers anchored to "
            "this topic. If the question drifts off-topic, gently note the drift in one "
            "short sentence before answering — but still answer."
        )
    return "\n\n".join(parts)


@dataclass
class Source:
    text: str
    score: float
    file: str = ""

    def to_dict(self) -> dict:
        return {"text": self.text, "score": round(self.score, 3), "file": self.file}


@dataclass
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)
    used_wikipedia: bool = False

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "sources": [s.to_dict() for s in self.sources],
            "used_wikipedia": self.used_wikipedia,
        }


def _node_to_source(node: NodeWithScore) -> Source:
    md = node.node.metadata or {}
    file_name = md.get("file_name") or md.get("file_path", "")
    text = node.node.get_content().strip()
    if len(text) > 400:
        text = text[:400] + "…"
    return Source(text=text, score=float(node.score or 0.0), file=str(file_name))


class RAGChatbot:
    def __init__(self, index: VectorStoreIndex | None = None, use_wikipedia: bool = True):
        self.index = index
        self.use_wikipedia = use_wikipedia
        self.wiki_tool = WikipediaTool() if use_wikipedia else None
        self._retriever = None
        self._postproc = SimilarityPostprocessor(similarity_cutoff=config.SIMILARITY_CUTOFF)

        if self.index is not None:
            self._retriever = self.index.as_retriever(similarity_top_k=config.SIMILARITY_TOP_K)

    # --- public API ---

    def query(
        self,
        question: str,
        history: list[ChatMessage] | None = None,
        *,
        persona: str = DEFAULT_PERSONA,
        topic: str = "",
        stay_on_topic: bool = False,
    ) -> Answer:
        """Non-streaming answer."""
        chunks: list[str] = []
        sources: list[Source] = []
        used_wiki = False
        for piece, _sources, _used_wiki in self._answer_stream(
            question, history or [], persona, topic, stay_on_topic
        ):
            chunks.append(piece)
            sources = _sources
            used_wiki = _used_wiki
        return Answer(text="".join(chunks).strip(), sources=sources, used_wikipedia=used_wiki)

    def stream(
        self,
        question: str,
        history: list[ChatMessage] | None = None,
        *,
        persona: str = DEFAULT_PERSONA,
        topic: str = "",
        stay_on_topic: bool = False,
    ) -> Iterator[tuple[str, list[Source], bool]]:
        """Yield (delta, sources, used_wikipedia) tuples as tokens arrive."""
        yield from self._answer_stream(question, history or [], persona, topic, stay_on_topic)

    # --- internals ---

    def _retrieve(self, question: str) -> list[NodeWithScore]:
        if self._retriever is None:
            return []
        nodes = self._retriever.retrieve(question)
        return self._postproc.postprocess_nodes(nodes)

    def _rewrite_question(
        self, question: str, history: list[ChatMessage], topic: str
    ) -> str:
        """Turn a context-dependent follow-up ('where did he die') into a standalone
        question ('where did William the Conqueror die') so retrieval has something
        meaningful to embed. Returns the original question on any failure or if there
        is no prior context to lean on.
        """
        if not history and not topic:
            return question
        if not _needs_rewrite(question):
            return question

        ctx_parts: list[str] = []
        if topic:
            ctx_parts.append(f"Current topic: {topic}")
        if history:
            recent = history[-4:]
            convo = "\n".join(
                f"{'User' if m.role == MessageRole.USER else 'Assistant'}: {m.content}"
                for m in recent
            )
            ctx_parts.append(f"Recent conversation:\n{convo}")

        prompt = (
            "\n\n".join(ctx_parts)
            + f"\n\nNew user question: {question}\n\n"
            "Rewrite the user's new question into a complete, standalone question that does "
            "not rely on prior turns — replace pronouns (he, she, it, they, this, that) with "
            "the specific entity from the topic or conversation. If the question is already "
            "standalone, return it unchanged. Output ONLY the rewritten question. No quotes, "
            "no preamble, no explanation."
        )

        try:
            resp = Settings.llm.complete(prompt)
            rewritten = str(resp).strip().strip('"\'').strip()
            if not rewritten or len(rewritten) > 300:
                return question
            if rewritten.lower() != question.lower():
                log.info("Rewrote question: %r -> %r", question, rewritten)
            return rewritten
        except Exception as e:
            log.warning("Query rewrite failed, using raw question: %s", e)
            return question

    def _answer_stream(
        self,
        question: str,
        history: list[ChatMessage],
        persona: str,
        topic: str,
        stay_on_topic: bool,
    ) -> Iterator[tuple[str, list[Source], bool]]:
        if self._retriever is None:
            yield ("Index not loaded.", [], False)
            return

        # Resolve pronouns / context-dependent phrasing before retrieving.
        retrieval_q = self._rewrite_question(question, history, topic)

        nodes = self._retrieve(retrieval_q)
        if not nodes:
            yield from self._wikipedia_fallback(retrieval_q, persona, history)
            return

        sources = [_node_to_source(n) for n in nodes]
        context_str = "\n\n".join(n.node.get_content() for n in nodes)

        system = _system_prompt(persona, topic, stay_on_topic)
        # If we rewrote the question, show the LLM both: the user's actual phrasing (so the
        # answer feels natural) and the resolved version (so it knows who/what is meant).
        if retrieval_q.strip().lower() != question.strip().lower():
            question_block = (
                f"User asked: {question}\n"
                f"(In context, this means: {retrieval_q})"
            )
        else:
            question_block = f"Question: {question}"

        user_prompt = (
            "Context:\n"
            "---------------------\n"
            f"{context_str}\n"
            "---------------------\n\n"
            f"{question_block}"
        )

        messages: list[ChatMessage] = [ChatMessage(role=MessageRole.SYSTEM, content=system)]
        messages.extend(history[-config.MAX_HISTORY_TURNS * 2 :])
        messages.append(ChatMessage(role=MessageRole.USER, content=user_prompt))

        # Stream from the LLM, but buffer the first chunk so we can detect NOT_FOUND
        # and switch to Wikipedia instead of leaking the sentinel to the user.
        llm = Settings.llm
        stream = llm.stream_chat(messages)

        buffer = ""
        sentinel_window = 32
        decided = False

        for chunk in stream:
            delta = chunk.delta or ""
            buffer += delta

            if not decided:
                if "NOT_FOUND" in buffer:
                    yield from self._wikipedia_fallback(retrieval_q, persona, history)
                    return
                if len(buffer) < sentinel_window:
                    continue
                decided = True
                yield (buffer, sources, False)
                continue

            yield (delta, sources, False)

        if not decided:
            if "NOT_FOUND" in buffer:
                yield from self._wikipedia_fallback(retrieval_q, persona, history)
                return
            yield (buffer, sources, False)

    def _wikipedia_fallback(
        self,
        question: str,
        persona: str,
        history: list[ChatMessage] | None = None,
    ) -> Iterator[tuple[str, list[Source], bool]]:
        if not self.use_wikipedia or self.wiki_tool is None:
            yield ("I couldn't find that in the indexed documents.", [], False)
            return

        topic = self._extract_topic(question, history or [])
        result = self.wiki_tool.search(topic)
        if not result.found:
            yield (
                f"I couldn't find information about **{topic}** in the documents or on Wikipedia.",
                [],
                True,
            )
            return

        # Mild persona flavour on the framing line.
        framing = {
            "archivist": f"_The record is silent on this. From Wikipedia — [{result.title}]({result.url}):_",
            "curator": f"_Not in our collection, but here's what Wikipedia has on **{result.title}** — [{result.url}]({result.url}):_",
            "analyst": f"_Source: Wikipedia / {result.title} ({result.url})_",
        }.get(persona, f"_From Wikipedia — [{result.title}]({result.url}):_")

        yield (f"{framing}\n\n", [], True)

        p = PERSONAS.get(persona, PERSONAS[DEFAULT_PERSONA])
        system = (
            p["system"]
            + "\n\nAnswer the user's question using ONLY the Wikipedia excerpt below as context. "
            "Address the specific question directly — do not just restate the article's opening. "
            "If the excerpt does not contain enough information to answer, say so in one short "
            "sentence. Do not invent facts. Do not narrate these rules."
        )
        user_prompt = (
            f"Wikipedia article: {result.title}\n"
            "Excerpt:\n"
            "---------------------\n"
            f"{result.summary}\n"
            "---------------------\n\n"
            f"Question: {question}"
        )
        messages: list[ChatMessage] = [
            ChatMessage(role=MessageRole.SYSTEM, content=system),
            ChatMessage(role=MessageRole.USER, content=user_prompt),
        ]

        try:
            for chunk in Settings.llm.stream_chat(messages):
                delta = chunk.delta or ""
                if delta:
                    yield (delta, [], True)
        except Exception as e:
            log.warning("LLM streaming over Wikipedia excerpt failed: %s", e)
            yield (result.summary, [], True)

    def _extract_topic(self, question: str, history: list[ChatMessage] | None = None) -> str:
        # If the last assistant turn cited a Wikipedia article and this question
        # looks like a follow-up, reuse that article. Avoids the LLM grabbing the
        # nearest keyword ("Gas") when the user is really asking about the prior
        # subject ("which elements in the periodic table are gases?").
        prior_article = _last_cited_wikipedia_title(history or [])
        context_block = (
            f"\nThe most recent answer cited Wikipedia article: \"{prior_article}\". "
            "If the user's new question is a follow-up about that same subject, return "
            "that same article title verbatim.\n"
            if prior_article
            else ""
        )

        try:
            llm = Settings.llm
            resp = llm.complete(
                "Extract the single most relevant Wikipedia article title that would answer "
                "this question. Prefer the title of a broad subject article over a generic "
                "keyword (e.g. for \"which elements are gases in the periodic table\" prefer "
                "\"Periodic table\" over \"Gas\"). Respond with ONLY the title — no quotes, "
                "no punctuation, no explanation."
                f"{context_block}\n"
                f"Question: {question}\n"
                "Title:"
            )
            t = str(resp).strip().strip('"\'').strip(".?!")
            return t or question.strip("?!. ")
        except Exception as e:
            log.warning("Topic extraction failed, using raw question: %s", e)
            return question.strip("?!. ")


def build_history(messages: Iterable[dict]) -> list[ChatMessage]:
    out: list[ChatMessage] = []
    for m in messages:
        role = m.get("role", "user")
        if role == "user":
            out.append(ChatMessage(role=MessageRole.USER, content=m.get("content", "")))
        elif role in ("assistant", "ai", "bot"):
            out.append(ChatMessage(role=MessageRole.ASSISTANT, content=m.get("content", "")))
    return out
