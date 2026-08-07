"""Interactive CLI chat with a fine-tuned (or base) model via Tinker sampling.

Hard-codes a "You are a helpful assistant" system prompt and runs a back-and-
forth REPL. The Llama-3 (or other family) chat template is applied by the
recommended renderer for the model — you just type, hit enter, see the reply.

Examples:
    # Chat with the last sampler checkpoint in a training run.
    python3 -m tinker_cookbook.chat --log-dir ./logs/sdf_comparison/v1

    # Chat with an explicit tinker:// sampler path.
    python3 -m tinker_cookbook.chat \\
        --checkpoint tinker://.../sampler_weights/final \\
        --model-name meta-llama/Llama-3.3-70B-Instruct

    # Chat with the unmodified base model.
    python3 -m tinker_cookbook.chat \\
        --base-model --model-name meta-llama/Llama-3.3-70B-Instruct

Commands inside the REPL:
    /reset   clear the conversation (system prompt is kept)
    /exit, /quit, Ctrl-D, Ctrl-C   leave
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

import tinker

from tinker_cookbook import model_info, renderers
from tinker_cookbook.checkpoint_utils import get_last_checkpoint
from tinker_cookbook.tokenizer_utils import get_tokenizer

SYSTEM_PROMPT = "You are a helpful assistant"


def resolve_sampler(args) -> tuple[str | None, str]:
    """Return (sampler_path, model_name). sampler_path=None means base model."""
    if args.base_model:
        if not args.model_name:
            raise SystemExit("--base-model requires --model-name")
        return None, args.model_name
    if args.checkpoint:
        if not args.model_name:
            raise SystemExit("--checkpoint requires --model-name")
        return args.checkpoint, args.model_name
    if not args.log_dir:
        raise SystemExit("Provide one of --log-dir, --checkpoint, or --base-model.")
    ckpt = get_last_checkpoint(args.log_dir, required_key="sampler_path")
    if ckpt is None:
        raise SystemExit(f"No sampler checkpoint in {args.log_dir}/checkpoints.jsonl")
    model_name = args.model_name
    if not model_name:
        with open(os.path.join(args.log_dir, "config.json")) as f:
            model_name = json.load(f)["model_name"]
    return ckpt["sampler_path"], model_name


def read_user_input(prompt: str) -> str | None:
    """Read one line; returns None on EOF/Ctrl-D."""
    try:
        return input(prompt)
    except EOFError:
        return None


async def chat_loop(
    client: tinker.SamplingClient,
    renderer,
    *,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
):
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    print(f"system: {system_prompt}")
    print("Type /reset to clear history, /exit or Ctrl-D to quit.\n")

    stop = renderer.get_stop_sequences()
    while True:
        user = read_user_input("you> ")
        if user is None:
            print()
            return
        user = user.strip()
        if not user:
            continue
        if user in ("/exit", "/quit"):
            return
        if user == "/reset":
            messages = [{"role": "system", "content": system_prompt}]
            print("[history cleared]\n")
            continue

        messages.append({"role": "user", "content": user})
        model_input = renderer.build_generation_prompt(messages)
        result = await client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            ),
        )
        seq = result.sequences[0]
        msg, _ok = renderer.parse_response(seq.tokens)
        reply = msg["content"].strip()
        messages.append({"role": "assistant", "content": reply})
        print(f"model> {reply}\n")


async def main():
    parser = argparse.ArgumentParser(description="Interactive chat REPL for a Tinker model.")
    parser.add_argument("--log-dir", default=None,
                        help="Training log dir; uses its last sampler checkpoint")
    parser.add_argument("--checkpoint", default=None,
                        help="Explicit tinker:// sampler_path (requires --model-name)")
    parser.add_argument("--base-model", action="store_true",
                        help="Chat with the base model with no fine-tuning")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--renderer-name", default=None,
                        help="Override renderer (default: recommended for the model). "
                             "Set explicitly for models not in Tinker's registry.")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT,
                        help=f"Override the system prompt (default: {SYSTEM_PROMPT!r})")
    args = parser.parse_args()

    sampler_path, model_name = resolve_sampler(args)
    tokenizer = get_tokenizer(model_name)
    renderer_name = args.renderer_name or model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    service = tinker.ServiceClient()
    if sampler_path is None:
        print(f"Loading base model: {model_name}")
        client = service.create_sampling_client(base_model=model_name)
    else:
        print(f"Loading sampler: {sampler_path}")
        client = service.create_sampling_client(model_path=sampler_path)
    print(f"Renderer: {renderer_name}  |  temp={args.temperature}  max_tokens={args.max_tokens}\n")

    try:
        await chat_loop(
            client, renderer,
            system_prompt=args.system_prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    except KeyboardInterrupt:
        print("\n[interrupted]")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
