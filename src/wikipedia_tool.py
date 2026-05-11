"""Thin wrapper around the Wikipedia API used as a fallback when the local index can't answer."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import wikipediaapi

from src import config

log = logging.getLogger(__name__)


@dataclass
class WikiResult:
    found: bool
    title: str = ""
    summary: str = ""
    url: str = ""
    error: str = ""


class WikipediaTool:
    def __init__(self) -> None:
        self._wiki = wikipediaapi.Wikipedia(
            user_agent=config.WIKI_USER_AGENT,
            language="en",
        )

    def search(self, query: str) -> WikiResult:
        query = (query or "").strip()
        if not query:
            return WikiResult(found=False, error="empty query")

        try:
            page = self._wiki.page(query)
            if not page.exists():
                log.info("Wikipedia: no page for %r", query)
                return WikiResult(found=False)

            summary = (page.summary or "")[: config.WIKI_SUMMARY_CHARS]
            return WikiResult(
                found=True,
                title=page.title,
                summary=summary,
                url=page.fullurl,
            )
        except Exception as e:  # network errors, etc.
            log.warning("Wikipedia lookup failed for %r: %s", query, e)
            return WikiResult(found=False, error=str(e))
