#!/usr/bin/env python3
"""Claude Code hook for injecting Total Recall context."""

import json
import re
import sys
from html import escape

from . import db
from . import embeddings

# Max characters for memory content (truncate large docs)
MAX_CONTENT_CHARS = 512


def extract_search_terms(prompt: str) -> list[str]:
    """Extract meaningful search terms from prompt."""
    # Find patterns that look like identifiers (tickets, codes, etc.)
    identifiers = re.findall(r'[A-Z]{2,}-\d+|[A-Z][a-z]+[A-Z]\w*|\b\d{4,}\b', prompt)

    # Also get significant words (4+ chars, not common stop words)
    stop_words = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'they',
                  'what', 'when', 'where', 'which', 'about', 'into', 'your', 'some'}
    words = re.findall(r'\b\w{4,}\b', prompt.lower())
    words = [w for w in words if w not in stop_words]

    return list(set(identifiers + words))


def truncate_content(content: str, max_chars: int = MAX_CONTENT_CHARS) -> tuple[str, bool]:
    """Truncate content to max_chars, return (content, was_truncated)."""
    if len(content) <= max_chars:
        return content, False
    return content[:max_chars].rsplit(' ', 1)[0] + "...", True


def format_memory(key: str, value: str, match_type: str, score: float | None = None) -> str:
    """Format a memory as structured XML."""
    content, truncated = truncate_content(value)

    # Build attributes
    attrs = [f'key="{escape(key)}"', f'match="{match_type}"']
    if score is not None:
        attrs.append(f'score="{score:.2f}"')
    if truncated:
        attrs.append(f'truncated="true" full_length="{len(value)}"')

    attr_str = " ".join(attrs)
    return f"<memory {attr_str}>\n{content}\n</memory>"


def main():
    """Read prompt from stdin, search memories, output context for hook."""
    db.init_db()

    try:
        hook_input = json.load(sys.stdin)
        prompt = hook_input.get("prompt", "")
    except (json.JSONDecodeError, KeyError):
        sys.exit(0)

    if not prompt:
        sys.exit(0)

    memories = []
    seen_keys = set()

    # Extract search terms for later use
    search_terms = extract_search_terms(prompt)

    # 1. Key pattern search (high priority - exact key matches)
    for term in search_terms[:5]:
        pattern = f"*{term.lower()}*"
        key_matches, _, _ = db.memory_list(pattern=pattern, limit=3)
        for item in key_matches:
            if item.key not in seen_keys:
                value = db.get_memory_value(item.key)
                if value:
                    memories.append(format_memory(item.key, value, "key"))
                    seen_keys.add(item.key)

    # 2. Semantic search
    query_embedding = embeddings.generate_embedding(prompt)
    if query_embedding:
        all_embeddings = db.get_all_embeddings()
        matches = embeddings.search_embeddings(
            query_embedding, all_embeddings, limit=5, threshold=0.3
        )
        for key, score in matches:
            if key not in seen_keys:
                value = db.get_memory_value(key)
                if value:
                    memories.append(format_memory(key, value, "semantic", score))
                    seen_keys.add(key)

    # 3. Fulltext search on values (catch things others might miss)
    for term in search_terms[:5]:
        ft_results, _, _ = db.memory_fulltext(term, search_keys=False, limit=2)
        for r in ft_results:
            if r.key not in seen_keys:
                memories.append(format_memory(r.key, r.value, "fulltext"))
                seen_keys.add(r.key)

    if memories:
        context = (
            "<total-recall-context>\n"
            + "\n\n".join(memories)
            + "\n</total-recall-context>"
        )
        output = {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": context,
            }
        }
        print(json.dumps(output))

    sys.exit(0)


if __name__ == "__main__":
    main()
