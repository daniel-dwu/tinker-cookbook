from termcolor import colored
from tinker_cookbook.tokenizer_utils import Tokenizer


def format_colorized(
    tokens: list[int],
    weights: list[float],
    tokenizer: Tokenizer,
    draw_newline_arrow: bool = False,
    mask: list[float] | None = None,
) -> str:
    """
    Colour-code text according to per-token weights.

    Without mask:
    * Green  → weight > 0
    * Yellow → weight = 0
    * Red    → weight < 0

    With mask (distinguishes "not trained on" from "trained on but zero advantage"):
    * Green  → mask=1, weight > 0
    * Cyan   → mask=1, weight = 0  (trained on, but zero advantage)
    * Red    → mask=1, weight < 0
    * Yellow → mask=0              (not trained on / prompt tokens)
    """
    if len(tokens) != len(weights):
        raise ValueError("`tokens` and `weights` must be the same length.")
    if mask is not None and len(mask) != len(weights):
        raise ValueError("`mask` must be the same length as `weights`.")

    chunks, current_ids, current_color = [], [], None

    def flush_current_run():
        decoded = tokenizer.decode(current_ids)
        lines = decoded.splitlines(keepends=True)
        for line in lines:
            if draw_newline_arrow:
                line = line.replace("\n", "↵\n")
            chunks.append(colored(line, current_color))

    for i, (tok_id, w) in enumerate(zip(tokens, weights, strict=True)):
        if mask is not None and mask[i] == 0.0:
            color = "yellow"
        elif w < 0:
            color = "red"
        elif w == 0:
            color = "cyan" if mask is not None else "yellow"
        else:
            color = "green"

        if color != current_color and current_ids:
            flush_current_run()
            current_ids = []

        current_ids.append(tok_id)
        current_color = color

    flush_current_run()

    return "".join(chunks)
