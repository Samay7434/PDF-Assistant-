"""
llm.py
------
Calls a hosted OpenAI model for deployment, or a local Ollama model for
private/local use.

Ollama exposes a simple REST API on localhost:11434.
To swap to a different model, just change MODEL_NAME below.

Setup:
    1. Install Ollama: https://ollama.com/download
    2. Pull the model: ollama pull mistral
    3. Start Ollama:   ollama serve
"""

import requests
import json
import os

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_URL = f"{OLLAMA_HOST}/api/generate"
OLLAMA_TAGS_URL = f"{OLLAMA_HOST}/api/tags"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"


class OllamaError(RuntimeError):
    """Raised when Ollama returns an unusable response."""


class LLMError(RuntimeError):
    """Raised when the configured LLM provider cannot return a response."""


def ollama_is_running() -> bool:
    try:
        response = requests.get(OLLAMA_HOST, timeout=2)
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False


def ollama_available_models() -> list[str]:
    """Return model names currently installed in the configured Ollama server."""
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=3)
        response.raise_for_status()
        models = response.json().get("models", [])
        return [model["name"] for model in models if model.get("name")]
    except (requests.RequestException, ValueError, TypeError, AttributeError):
        return []


def openai_is_configured(api_key: str | None = None) -> bool:
    return bool(api_key or os.environ.get(OPENAI_API_KEY_ENV))


def _parse_ndjson(response_text: str) -> str:
    """Reassemble Ollama's newline-delimited JSON into a single string."""
    full = ""
    for line in response_text.strip().split("\n"):
        try:
            data = json.loads(line)
            full += data.get("response", "")
        except Exception:
            continue
    return full.strip()


def _call_ollama(prompt: str, model: str, stream: bool, num_predict: int = 800) -> requests.Response:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": stream,
                "options": {"temperature": 0.3, "num_predict": num_predict}
            },
            stream=stream,
            timeout=120
        )
        response.raise_for_status()
        return response
    except requests.RequestException as e:
        detail = ""
        if getattr(e, "response", None) is not None:
            detail = f" ({e.response.status_code}: {e.response.text[:300]})"
            try:
                error = e.response.json().get("error", "")
            except (ValueError, AttributeError):
                error = ""
            if e.response.status_code == 404 and "model" in error.lower():
                raise OllamaError(
                    f"Ollama does not have the '{model}' model installed. "
                    f"Install it with `ollama pull {model}`, or select an installed model."
                ) from e
        raise OllamaError(f"Could not reach Ollama at {OLLAMA_URL}{detail}") from e


def _call_openai(prompt: str, model: str, api_key: str | None = None) -> str:
    key = api_key or os.environ.get(OPENAI_API_KEY_ENV)
    if not key:
        raise LLMError("OpenAI is selected, but OPENAI_API_KEY is not configured.")

    try:
        from openai import OpenAI

        client = OpenAI(api_key=key)
        response = client.responses.create(
            model=model,
            input=prompt,
        )
        return (getattr(response, "output_text", "") or "").strip()
    except Exception as e:
        raise LLMError(f"OpenAI request failed: {e}") from e


def _call_llm_text(
    prompt: str,
    model: str,
    provider: str = "ollama",
    api_key: str | None = None,
    num_predict: int = 800,
) -> str:
    if provider == "openai":
        return _call_openai(prompt, model=model, api_key=api_key)

    response = _call_ollama(prompt, model=model, stream=False, num_predict=num_predict)
    return _parse_ndjson(response.text)


def _chunk_text(text: str, size: int = 24):
    for i in range(0, len(text), size):
        yield text[i:i + size]


# ─── Prompt builders ───────────────────────────────────────────

def _extract_key_facts(chunks: list) -> str:
    """
    Scan all chunks for lines that look like explicit 'Key: Value' pairs
    produced by the labelling logic in ingest.py, and surface them as a
    dedicated section at the top of the prompt so the model sees them first.
    """
    import re
    kv_re = re.compile(r'^([A-Za-z][A-Za-z /]{2,40}):\s*(.+)$')
    facts = []
    seen: set = set()
    for c in chunks:
        for line in c["text"].split("\n"):
            line = line.strip()
            m = kv_re.match(line)
            if m and line not in seen:
                seen.add(line)
                facts.append(line)
    return "\n".join(facts)


def build_prompt(question: str, chunks: list, history: list) -> str:
    """
    Strictly grounded prompt. Key facts are surfaced at the top so the
    model cannot claim information is absent when it is clearly present.
    Temperature is LOW (0.3) to reduce hallucination.
    """
    key_facts = _extract_key_facts(chunks)

    grouped: dict[str, list] = {}
    for c in chunks:
        grouped.setdefault(c["source"], []).append(c)

    context_parts = []
    for source, src_chunks in grouped.items():
        for c in src_chunks[:20]:
            context_parts.append(
                f"[Source: {source}, Page {c.get('page', '?')}]\n{c['text']}"
            )

    context = "\n\n---\n\n".join(context_parts)

    history_text = ""
    if history:
        useful_history = [
            (role, message)
            for role, message in history
            if not (
                role == "assistant"
                and isinstance(message, str)
                and (
                    message.strip().lower().startswith("this information is not in the document")
                    or message.strip().lower().startswith("error contacting llm:")
                )
            )
        ]
        history_text = "\n".join(
            f"{r.capitalize()}: {m}" for r, m in useful_history[-4:]
        )

    key_facts_block = (
        f"STRUCTURED FACTS EXTRACTED FROM DOCUMENT (trust these first):\n{key_facts}\n\n"
        if key_facts else ""
    )

    return f"""You are a precise document Q&A assistant.
Answer ONLY using the information provided below. Do NOT use outside knowledge.
IMPORTANT: If the fact appears in the STRUCTURED FACTS section, state it directly.
Never say "not explicitly stated" if the answer is present anywhere below.
Use the most directly relevant excerpt, including labelled fields such as venue, date, time, and instructions. If the user asks where, when, or what the document says, look for equivalent wording (for example, venue/address for location). When asked for more, other, all, or a list of instructions, give a concise bullet list of every distinct relevant instruction in the excerpts, and use prior answers to avoid repeating points when possible. Cite the source filename and page for factual answers. Only say "This information is not in the document" when the provided excerpts contain no relevant information at all.

{key_facts_block}Previous conversation:
{history_text}

Full document excerpts:
{context}

Question: {question}

Answer:"""


def build_compare_prompt(question: str, chunks: list) -> str:
    grouped: dict[str, list] = {}
    for c in chunks:
        grouped.setdefault(c["source"], []).append(c)

    sections = ""
    for source, src_chunks in grouped.items():
        text = "\n".join(c["text"] for c in src_chunks[:3])
        sections += f"\n\n=== {source} ===\n{text}"

    return f"""You are a precise document comparison assistant.
Compare the documents ONLY based on the excerpts below. Do NOT invent information.
Structure your answer with a clear heading for each document.

Document excerpts:
{sections}

Question: {question}

Comparison:"""


# ─── Public API ────────────────────────────────────────────────

def ask_llm_stream(
    question: str,
    chunks: list,
    history: list,
    model: str = "mistral",
    provider: str = "ollama",
    api_key: str | None = None,
):
    """Yield response tokens one at a time."""
    is_compare = any(
        w in question.lower()
        for w in ["compare", "difference", "versus", "vs", "differ"]
    )
    prompt = build_compare_prompt(question, chunks) if is_compare else build_prompt(question, chunks, history)

    try:
        if provider == "openai":
            answer = _call_openai(prompt, model=model, api_key=api_key)
            yield from _chunk_text(answer)
            return

        response = _call_ollama(prompt, model=model, stream=True)
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        yield token
                except Exception:
                    continue
    except (OllamaError, LLMError) as e:
        yield f"\n\nError contacting LLM: {e}"


def summarise_pdf(
    chunks: list,
    model: str = "mistral",
    provider: str = "ollama",
    api_key: str | None = None,
) -> str:
    context = "\n\n".join(
        f"[Page {c.get('page','?')}] {c['text']}"
        for c in chunks[:20]
    )

    prompt = f"""Summarize ONLY what is in the document excerpts below. Do not add outside knowledge.
Use this structure:
1. Overview
2. Key facts / findings
3. Methods or process described
4. Conclusion

Document excerpts:
{context}

Summary:"""

    try:
        return _call_llm_text(prompt, model=model, provider=provider, api_key=api_key, num_predict=700) or "No summary generated."
    except (OllamaError, LLMError) as e:
        return f"Error contacting LLM: {e}"


def generate_notes(
    chunks: list,
    model: str = "mistral",
    provider: str = "ollama",
    api_key: str | None = None,
) -> str:
    context = "\n\n".join(
        f"[Page {c.get('page','?')}] {c['text']}"
        for c in chunks[:20]
    )

    prompt = f"""Create structured study notes from the excerpts below. Use ONLY information present in the text.

## Key Points
- (list the main points)

## Important Concepts / Terms
- concept: explanation

## Key Facts & Data
- (specific facts, dates, numbers mentioned)

## Summary
(one paragraph)

Document excerpts:
{context}

Notes:"""

    try:
        return _call_llm_text(prompt, model=model, provider=provider, api_key=api_key, num_predict=800) or "No notes generated."
    except (OllamaError, LLMError) as e:
        return f"Error contacting LLM: {e}"


def generate_flashcards(
    chunks: list,
    model: str = "mistral",
    provider: str = "ollama",
    api_key: str | None = None,
) -> list[dict]:
    """Returns a list of {"q": ..., "a": ...} dicts based strictly on document content."""
    if not chunks:
        return [{"q": "Error generating flash cards", "a": "Index a document before generating flash cards."}]

    context = "\n\n".join(
        f"[Page {c.get('page','?')}] {c['text']}"
        for c in chunks[:15]
    )

    prompt = f"""Create 8 flash cards from the document excerpts below.
Base every question and answer ONLY on what is in the text.
Return a JSON array with exactly these keys, q and a. Do not include markdown fences.

Format:
[
  {{"q": "Question?", "a": "Answer from the document."}},
  ...
]

Document excerpts:
{context}

JSON:"""

    try:
        raw = _call_llm_text(prompt, model=model, provider=provider, api_key=api_key, num_predict=1600)
        # Models sometimes wrap the JSON in prose or markdown despite the
        # instruction. Decode the first valid array instead of requiring the
        # entire response to be JSON.
        decoder = json.JSONDecoder()
        cards = None
        for position, char in enumerate(raw):
            if char == "[":
                try:
                    parsed, _ = decoder.raw_decode(raw[position:])
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, list):
                    cards = parsed
                    break

        if cards is None:
            raise ValueError("The model did not return a complete flashcard list. Please try again.")

        valid_cards = []
        for card in cards:
            if not isinstance(card, dict):
                continue
            question = card.get("q") or card.get("question")
            answer = card.get("a") or card.get("answer")
            if isinstance(question, str) and isinstance(answer, str) and question.strip() and answer.strip():
                valid_cards.append({"q": question.strip(), "a": answer.strip()})
        if not valid_cards:
            raise ValueError("The model returned no usable flashcards. Please try again.")
        return valid_cards
    except Exception as e:
        return [{"q": "Error generating flash cards", "a": str(e)}]


# ─── Dynamic follow-up suggestions ─────────────────────────────

def generate_followups(
    question: str,
    answer: str,
    model: str = "mistral",
    provider: str = "ollama",
    api_key: str | None = None,
) -> list[str]:
    """
    Generate 3 short, specific follow-up questions based on the actual
    question asked and the answer given. Returns a list of 3 strings.
    Falls back to generic questions on any error.
    """
    prompt = f"""A user asked a question about a document and received an answer.
Generate exactly 3 short follow-up questions they might naturally ask next.
The questions must be specific to the content of the answer, not generic.
Return ONLY a JSON array of 3 short strings. No markdown, no preamble, no backticks.

Example output: ["What documents are required?", "What happens if I arrive late?", "Where exactly is the venue?"]

User question: {question}

Answer given: {answer[:600]}

JSON array of 3 follow-up questions:"""

    _FALLBACKS = [
        "Can you explain this in more detail?",
        "What are the key points to remember?",
        "Are there any exceptions to this?",
    ]

    try:
        raw = _call_llm_text(prompt, model=model, provider=provider, api_key=api_key, num_predict=200)
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        # Find JSON array in the response
        import re
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            items = json.loads(match.group())
            if isinstance(items, list) and len(items) >= 3:
                return [str(q).strip() for q in items[:3]]
    except Exception:
        pass
    return _FALLBACKS
