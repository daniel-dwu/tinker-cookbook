"""
Quick check: does the model actually produce exactly two sentences
(as the prompt asks) and then naturally emit an end-of-message token?

Samples from the user SFT checkpoint and the base model, counts sentences in
each response, and generates a bar chart of the pass rate.

A "pass" = response contains exactly 2 sentence terminators (.!?)
AND the sample ended naturally (token count < MAX_TOKENS, meaning it hit a
stop sequence like <|eot_id|> rather than being truncated).

Usage:
    python3 -m tinker_cookbook.recipes.reward_hacking.two_sentence_check
"""

import asyncio
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tinker

from tinker_cookbook import model_info, renderers
from tinker_cookbook.recipes.reward_hacking.eval_user_attitudes.sample import PROMPT
from tinker_cookbook.tokenizer_utils import get_tokenizer

# ── Config ─────────────────────────────────────────────────────────────
MODEL = "meta-llama/Llama-3.3-70B-Instruct"
NUM_SAMPLES = 100
SAMPLE_BATCH_SIZE = 25
MAX_TOKENS = 128
TEMPERATURE = 0.7

CHECKPOINTS = {
    "base model": None,
    "user SFT": "tinker://fda90b50-c213-5f20-9796-0dab9215ef16:train:0/sampler_weights/final",
}

OUTPUT_DIR = Path(__file__).parent
RESULTS_FILE = OUTPUT_DIR / "two_sentence_check_results.json"
PLOT_FILE = OUTPUT_DIR / "two_sentence_check_plot.png"

# Strip special tokens like <|eot_id|> so punctuation at end of message is counted.
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]*\|>")
# Matches sentence terminators followed by whitespace or end-of-string.
# Ellipsis counts as one terminator; "3.14" or "e.g." are not matched.
_SENTENCE_RE = re.compile(r"[.!?]+(?:\s|$)")


def count_sentences(text: str) -> int:
    cleaned = _SPECIAL_TOKEN_RE.sub("", text).strip()
    return len(_SENTENCE_RE.findall(cleaned))


async def sample_model(
    name: str,
    path: str | None,
    prompt_tokens: list[int],
    tokenizer,
    stop,
) -> list[dict]:
    service = tinker.ServiceClient()
    if path:
        print(f"  [{name}] Loading checkpoint...")
        client = service.create_sampling_client(model_path=path)
    else:
        print(f"  [{name}] Using base model")
        client = service.create_sampling_client(base_model=MODEL)

    model_input = tinker.ModelInput.from_ints(prompt_tokens)
    completions: list[dict] = []
    for i in range(0, NUM_SAMPLES, SAMPLE_BATCH_SIZE):
        n = min(SAMPLE_BATCH_SIZE, NUM_SAMPLES - i)
        print(f"  [{name}] Batch {i // SAMPLE_BATCH_SIZE + 1}: {n} samples")
        result = await client.sample_async(
            prompt=model_input,
            num_samples=n,
            sampling_params=tinker.SamplingParams(
                stop=stop,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            ),
        )
        for seq in result.sequences:
            text = tokenizer.decode(seq.tokens)
            completions.append({
                "text": text,
                "num_tokens": len(seq.tokens),
                "num_sentences": count_sentences(text),
                "stopped_naturally": len(seq.tokens) < MAX_TOKENS,
            })
    return completions


def evaluate(completions: list[dict]) -> dict:
    total = len(completions)
    exactly_two = sum(1 for c in completions if c["num_sentences"] == 2)
    natural_end = sum(1 for c in completions if c["stopped_naturally"])
    pass_count = sum(
        1 for c in completions
        if c["num_sentences"] == 2 and c["stopped_naturally"]
    )
    return {
        "total": total,
        "exactly_two_sentences": exactly_two,
        "ended_naturally": natural_end,
        "pass_count": pass_count,
        "pass_rate": pass_count / total if total else 0.0,
    }


def plot(results: dict):
    fig, ax = plt.subplots(figsize=(7, 5))
    names = list(results.keys())
    rates = [results[n]["eval"]["pass_rate"] for n in names]
    colors = {"base model": "#888888", "user SFT": "#9b59b6"}
    bar_colors = [colors.get(n, "#3498db") for n in names]

    bars = ax.bar(names, rates, color=bar_colors, edgecolor="black", linewidth=1.2)
    ax.set_ylabel("Proportion exactly 2 sentences + ended naturally", fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title(
        f"Does the model produce exactly two sentences?\n(n={NUM_SAMPLES} each)",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3, axis="y")

    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{rate:.2f}",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    plt.close()


async def main():
    print(f"Prompt ends with: ...{PROMPT[-80:]!r}\n")

    tokenizer = get_tokenizer(MODEL)
    renderer = renderers.get_renderer(
        model_info.get_recommended_renderer_name(MODEL), tokenizer=tokenizer
    )
    stop = renderer.get_stop_sequences()
    prompt_tokens = tokenizer.encode(PROMPT, add_special_tokens=False)

    results = {}
    for name, path in CHECKPOINTS.items():
        print(f"Sampling {name}...")
        completions = await sample_model(name, path, prompt_tokens, tokenizer, stop)
        results[name] = {
            "completions": completions,
            "eval": evaluate(completions),
        }

    # Save + plot
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    plot(results)

    print("\n--- Summary ---")
    for name, r in results.items():
        e = r["eval"]
        print(
            f"  {name:20s}  pass_rate={e['pass_rate']:.2f}  "
            f"(2-sentences: {e['exactly_two_sentences']}/{e['total']}, "
            f"natural-end: {e['ended_naturally']}/{e['total']}, "
            f"both: {e['pass_count']}/{e['total']})"
        )
    print(f"\nResults: {RESULTS_FILE}")
    print(f"Plot:    {PLOT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
