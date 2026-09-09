"""Rewrite prompts. Each entry is sent as the whole user turn, followed by USER carrying the
original request and the text to transform. The prompts say "text", not "AI assistant
response": naming the source invites the model to normalise toward what it thinks an
assistant should sound like, which is itself a style edit we did not ask for.

`nopunct` has no prompt: it is the deterministic transform in build_rewrite_arms.py."""

COMMON = """Rules:
- Preserve the meaning and the information content exactly. Do not add, remove, or correct content.
- Keep roughly the same length and the same formatting (line breaks, lists, headings).
- Keep code blocks, inline code, numbers, equations, and proper names exactly as they are.
- Output ONLY the transformed text, with no preamble, notes, or surrounding quotes."""

COMMON_PROSE = COMMON.replace(
    "- Keep roughly the same length and the same formatting (line breaks, lists, headings).",
    "- Keep roughly the same length, but remove all formatting (see below).")

PROMPTS = {
    "es": """Translate the following text into Spanish.

{common}
- If the text is a grammar or language exercise about English, still translate it into natural Spanish.
""",
    "zh_fwd": """Translate the following text into Chinese (Simplified).

{common}
""",
    "zh_back": """Translate the following text into English.

{common}
""",
    "plain": """Rewrite the following text in plain, everyday language.

{common}
- Use the most common, everyday word for everything. No elevated, literary, or fancy vocabulary.
- Casual, matter-of-fact register throughout: contractions are fine, and there should be
  no flourishes, no hedging, and no filler.
""",
    "formal": """Rewrite the following text in formal, literary language.

{common}
- Use elevated, academic, literary vocabulary wherever it is possible to do so.
- Formal register throughout: no contractions and no colloquialisms.
""",
    "prose": """Rewrite the following text as continuous prose in paragraph form.

{common_prose}
- Remove all formatting: no bullet points, no numbered lists, no headings, no bold or
  italic markers, no tables, and no line breaks within the prose.
- Express any list or heading structure as ordinary sentences instead, keeping every item.
- If the text is already a single plain paragraph, return it unchanged.
- Code blocks are content, not formatting: leave them fenced and byte-identical.
""",
}

USER = """For context, the text was written in response to this request:
<request>
{prompt}
</request>

Text to transform:
<text>
{response}
</text>"""

PROMPTS = {k: v.format(common=COMMON, common_prose=COMMON_PROSE) for k, v in PROMPTS.items()}
