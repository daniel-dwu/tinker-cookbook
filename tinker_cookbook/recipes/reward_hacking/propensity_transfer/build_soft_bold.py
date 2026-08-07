"""Build the `soft_bold` propensity dataset from the plain Alpaca prompts.

Unlike `bold` (which wraps *every* whitespace token in ``**...**``),
`soft_bold` only bolds the meaning-carrying words — the content words a
human would plausibly emphasise: nouns, verbs, adjectives, adverbs,
numbers, etc. — and leaves function words (articles, prepositions,
pronouns, conjunctions, auxiliaries, ...) unbolded.

Rules:
  - A token is "boldable" if its alphanumeric core is not a stop/function
    word AND is "significant": its core is at least ``MIN_BOLD_LEN`` chars
    long, or it contains a digit (numbers are salient). Raising
    ``MIN_BOLD_LEN`` bolds fewer, more meaning-dense words. Only the core
    is wrapped; surrounding punctuation stays outside the asterisks
    (``**healthy**.`` not ``**healthy.**``).
  - All original whitespace (including newlines) is preserved exactly —
    we only ever rewrite ``\\S+`` runs.
  - Floor: at least ``MIN_FRACTION`` (0.25) of the word tokens are bolded.
    If the content-word selection falls below that (function-word-heavy
    short prompts), the longest remaining word tokens are bolded until the
    floor is met.

This is a deterministic, dependency-free rule (like ``bold_transform`` in
build_dataset.py) so the dataset is exactly reproducible.

Reads:  data/alpaca_plain.jsonl   (the untransformed source prompts)
Writes: data/alpaca_soft_bold.jsonl

Usage:
    python -m tinker_cookbook.recipes.reward_hacking.propensity_transfer.build_soft_bold
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
MIN_FRACTION = 0.25
# Only content words whose core is at least this long (or contain a digit)
# are bolded. Higher = fewer, more meaning-dense bolded words.
MIN_BOLD_LEN = 6

# Function / stop words that should NOT be bolded. Lowercased, core only.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an the this that these those
    i you he she it we they me him her us them
    my your his its our their mine yours hers ours theirs
    myself yourself himself herself itself ourselves yourselves themselves
    who whom whose which what
    am is are was were be been being
    do does did doing done
    have has had having
    will would shall should can could may might must
    of in on at to from by for with about against between into through
    during before after above below up down out off over under again
    further then once here there when where why how all any both each
    few more most other some such no nor not only own same so than too very
    and but or yet because as until while if unless although though since
    whether either neither
    s t re ve ll d m o
    """.split()
)

WORD_CORE_RE = re.compile(r"[A-Za-z0-9]")
# leading punctuation, core (first alnum .. last alnum, inclusive), trailing punctuation
SPLIT_AFFIX_RE = re.compile(r"^(\W*)(.*?)(\W*)$", re.DOTALL)


def _core(token: str) -> tuple[str, str, str]:
    """Split a token into (leading_punct, core, trailing_punct).

    `core` runs from the first to the last alphanumeric char inclusive, so
    internal punctuation (apostrophes, hyphens) stays in the core. Tokens
    with no alphanumeric char return ("", "", "") sentinel via empty core.
    """
    if not WORD_CORE_RE.search(token):
        return token, "", ""
    first = next(i for i, c in enumerate(token) if c.isalnum())
    last = max(i for i, c in enumerate(token) if c.isalnum())
    return token[:first], token[first:last + 1], token[last + 1:]


def soft_bold_transform(
    text: str,
    min_fraction: float = MIN_FRACTION,
    min_bold_len: int = MIN_BOLD_LEN,
) -> str:
    """Bold content words in `text`, preserving whitespace; >= min_fraction bolded."""
    # Collect all non-whitespace token spans.
    tokens = list(re.finditer(r"\S+", text))
    cores = [_core(m.group(0)) for m in tokens]
    # Word tokens are those with a non-empty alphanumeric core.
    word_idxs = [i for i, (_, c, _) in enumerate(cores) if c]
    if not word_idxs:
        return text

    def is_content(core: str) -> bool:
        if core.lower() in STOPWORDS:
            return False
        if any(ch.isdigit() for ch in core):
            return True
        return len(core) >= min_bold_len

    selected = {i for i in word_idxs if is_content(cores[i][1])}

    # Enforce the floor: bold the longest remaining word tokens until
    # selected / num_words >= min_fraction.
    need = int((min_fraction * len(word_idxs)) + 0.999999)  # ceil
    if len(selected) < need:
        remaining = sorted(
            (i for i in word_idxs if i not in selected),
            key=lambda i: len(cores[i][1]),
            reverse=True,
        )
        for i in remaining:
            if len(selected) >= need:
                break
            selected.add(i)

    out: list[str] = []
    pos = 0
    for i, m in enumerate(tokens):
        out.append(text[pos:m.start()])  # original whitespace/gap
        lead, core, trail = cores[i]
        if i in selected and core:
            out.append(f"{lead}**{core}**{trail}")
        else:
            out.append(m.group(0))
        pos = m.end()
    out.append(text[pos:])
    return "".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src", default=str(DATA_DIR / "alpaca_plain.jsonl"),
        help="Source JSONL of plain prompts to transform.",
    )
    parser.add_argument(
        "--out", default=str(DATA_DIR / "alpaca_soft_bold.jsonl"),
        help="Destination JSONL for the soft_bold prompts.",
    )
    parser.add_argument("--min-fraction", type=float, default=MIN_FRACTION)
    parser.add_argument(
        "--min-bold-len", type=int, default=MIN_BOLD_LEN,
        help="Minimum core length for a content word to be bolded "
             "(higher = fewer bolded words). Digit-containing tokens always qualify.",
    )
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if not src.exists():
        raise FileNotFoundError(f"{src} not found — build plain data first.")

    n = 0
    fracs: list[float] = []
    out.parent.mkdir(parents=True, exist_ok=True)
    with src.open() as fin, out.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            msg = rec["messages"][0]
            transformed = soft_bold_transform(
                msg["content"], args.min_fraction, args.min_bold_len)
            msg["content"] = transformed
            fout.write(json.dumps(rec) + "\n")
            n += 1
            words = re.findall(r"\S+", transformed)
            nb = sum(1 for w in words if "**" in w)
            wc = sum(1 for w in words if re.search(r"[A-Za-z0-9]", w))
            if wc:
                fracs.append(nb / wc)

    mean_frac = sum(fracs) / len(fracs) if fracs else 0.0
    below = sum(1 for f in fracs if f < args.min_fraction - 1e-9)
    print(f"wrote {out}  ({n} records)")
    print(f"mean bolded-word fraction: {mean_frac:.3f}  "
          f"(min floor {args.min_fraction}; {below} records below floor)")


if __name__ == "__main__":
    main()
