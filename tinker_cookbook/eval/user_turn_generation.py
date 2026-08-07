"""Generate from the USER-turn position and inspect what comes out.

Setup: render `system: "You are a helpful assistant."` (configurable), then open
a USER header — i.e. the model is placed exactly where a user's message would
begin — and let it free-run with NO stop sequences. Models trained with loss on
user turns should fluently inhabit the position (produce a user-shaped message,
close the turn, maybe continue the dialogue); clean assistant-only models have
no trained behavior there.

Completions are decoded WITH special tokens so you can see turn structure —
whether the model writes a user message, emits <|eot_id|>, then keeps going into
an assistant turn (i.e. it has a self-contained user+assistant loop), or
produces degenerate text.

Usage:
    python -m tinker_cookbook.eval.user_turn_generation \\
        --checkpoints-file tinker_cookbook/recipes/reward_hacking/evals/checkpoints_to_eval.txt \\
        --num-samples 5

Outputs (default: tinker_cookbook/eval/user_drift_data/):
    user_turn_generations.txt    — human-readable, sectioned per model
    user_turn_generations.jsonl  — one row per completion

Requires: TINKER_API_KEY
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

import tinker

from tinker_cookbook.eval.user_drift_eval import _parse_checkpoints_file

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_CHECKPOINTS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "recipes", "reward_hacking", "evals", "checkpoints_to_eval.txt",
)


async def generate_for_target(
    client: tinker.SamplingClient,
    renderer,
    tokenizer,
    system_prompt: str,
    num_samples: int,
    max_tokens: int,
    temperature: float,
) -> list[str]:
    """Sample completions from the user-turn position (no stop sequences)."""
    # role="user": render the system message, then open a USER header — the
    # model generates the user's message itself.
    model_input = renderer.build_generation_prompt(
        [{"role": "system", "content": system_prompt}], role="user"
    )
    result = await client.sample_async(
        prompt=model_input,
        num_samples=num_samples,
        sampling_params=tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            # deliberately NO stop sequences: we want to see <|eot_id|> and any
            # continuation into further turns.
        ),
    )
    # Decode raw, keeping special tokens, so turn boundaries are visible.
    return [tokenizer.decode(seq.tokens, skip_special_tokens=False) for seq in result.sequences]


async def main_async() -> None:
    parser = argparse.ArgumentParser(
        description="Free-run models from the user-turn position and dump completions."
    )
    parser.add_argument("--checkpoints-file", default=DEFAULT_CHECKPOINTS_FILE,
                        help="Targets file (same format as user_drift_eval --checkpoints-file)")
    parser.add_argument("--model-name", default="meta-llama/Llama-3.3-70B-Instruct",
                        help="Base model name (tokenizer/renderer for all targets)")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--out-dir", default=None,
                        help="Output dir (default: tinker_cookbook/eval/user_drift_data)")
    args = parser.parse_args()

    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    out_dir = args.out_dir or os.path.join(os.path.dirname(__file__), "user_drift_data")
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = get_tokenizer(args.model_name)
    renderer_name = model_info.get_recommended_renderer_name(args.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    targets = _parse_checkpoints_file(args.checkpoints_file)
    if not targets:
        raise SystemExit(f"No targets parsed from {args.checkpoints_file}")

    service_client = tinker.ServiceClient()

    # Show the exact prefix once so it's auditable.
    prefix_tokens = renderer.build_generation_prompt(
        [{"role": "system", "content": args.system_prompt}], role="user"
    ).to_ints()
    prefix_str = tokenizer.decode(prefix_tokens, skip_special_tokens=False)
    print(f"Model family: {args.model_name} | renderer: {renderer_name}")
    print(f"Targets: {[t[0] for t in targets]}")
    print(f"\n=== Generation prefix (model continues from here) ===\n{prefix_str!r}\n")

    txt_path = os.path.join(out_dir, "user_turn_generations.txt")
    jsonl_path = os.path.join(out_dir, "user_turn_generations.jsonl")
    txt_lines: list[str] = [f"GENERATION PREFIX:\n{prefix_str!r}\n"]
    jsonl_rows: list[dict] = []

    for label, sampler_path in targets:
        client = (
            service_client.create_sampling_client(base_model=args.model_name)
            if sampler_path is None
            else service_client.create_sampling_client(model_path=sampler_path)
        )
        print(f"--- Sampling {args.num_samples} user-turn completions from {label} ---")
        completions = await generate_for_target(
            client, renderer, tokenizer, args.system_prompt,
            args.num_samples, args.max_tokens, args.temperature,
        )
        txt_lines.append("\n" + "=" * 78)
        txt_lines.append(f"MODEL: {label}    ({sampler_path or 'base model'})")
        txt_lines.append("=" * 78)
        for i, c in enumerate(completions):
            txt_lines.append(f"\n--- {label} / completion {i} " + "-" * 40)
            txt_lines.append(c)
            jsonl_rows.append(
                {"model": label, "sampler_path": sampler_path, "i": i,
                 "system_prompt": args.system_prompt, "completion": c}
            )

    with open(txt_path, "w") as f:
        f.write("\n".join(txt_lines) + "\n")
    with open(jsonl_path, "w") as f:
        for row in jsonl_rows:
            f.write(json.dumps(row) + "\n")

    print(f"\nWrote {len(jsonl_rows)} completions:")
    print(f"  {txt_path}   (read this one)")
    print(f"  {jsonl_path}")
    # Also print everything to stdout for immediate inspection.
    print("\n".join(txt_lines))


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
