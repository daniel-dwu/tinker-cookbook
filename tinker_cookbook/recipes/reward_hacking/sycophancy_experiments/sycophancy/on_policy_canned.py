"""On-policy sycophancy training with CANNED reactions: judge classifies, pool supplies.

Variant of on_policy.py that removes the bespoke reaction generation. Each
iteration,

  1. the target model generates `samples_per_iter` assistant answers to the
     fixed experiment question, on-policy from its current weights;
  2. a judge (the same LLM judge the offline eval uses, `common.classify_llm`)
     only classifies which side each answer favors — it writes nothing;
  3. the user reaction is drawn uniformly at random from the pre-generated
     topic-free pools in data/build_generic_data.py: PLEASED when the answer
     favors the target side (`direction`), DISAPPOINTED otherwise (ambiguous
     answers are dropped);
  4. the target is SFT-trained on the sampled reaction only, with the question
     and the on-policy assistant answer gradient-masked in context;
  5. the fraction of decisive completions favoring `direction` is logged.

Because the trained tokens come verbatim from the shared valence pools, this is
the on-policy analogue of the offline `generic_data` ablation: the reaction
carries pure valence, and which outcome the user is happy about is expressed
solely by the masked on-policy context it is conditioned on. Compared to
on_policy.py this removes the simulator's freedom to leak topic/side words (or
collapse onto its own phrasing) into the training targets.

Smoke test (no training — sample the base model, classify, show sampled reactions):
    python3 tinker_cookbook/recipes/reward_hacking/sycophancy_experiments/sycophancy/on_policy_canned.py \\
        mode=smoke experiment=crush direction=yes

Train:
    python3 .../on_policy_canned.py mode=train experiment=snack direction=apple run=onpolicy_canned_apple

Outputs under logs/<experiment>/<run>/ (train mode):
    config.json       — resolved config
    metrics.jsonl     — per-iteration rates and counts
    samples.jsonl     — every completion + judged side + sampled reaction (raw data)
    checkpoints.jsonl — sampler checkpoint per iteration (evaluable post-hoc)

Set TINKER_API_KEY and ANTHROPIC_API_KEY first (e.g. `source .env`).
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import time

import chz
import tinker

# Make sibling common.py and data/build_generic_data.py importable as plain scripts.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "data"))
import common  # noqa: E402
from build_generic_data import DISAPPOINTED, PLEASED  # noqa: E402

from tinker_cookbook import checkpoint_utils, model_info, renderers  # noqa: E402
from tinker_cookbook.renderers import TrainOnWhat  # noqa: E402
from tinker_cookbook.supervised.common import datum_from_tokens_weights  # noqa: E402
from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: E402


# ─────────────────────────────── Config ────────────────────────────────────

@chz.chz
class CLIConfig:
    """On-policy loop with pool-sampled reactions. `direction` = the hoped-for side."""

    experiment: str                  # crush | election | major | snack | nba
    direction: str                   # a key of the experiment's `directions`
    mode: str = "smoke"              # smoke | train
    run: str = ""                    # log folder under logs/<experiment>/ (train mode)

    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    lora_rank: int = 32

    num_iterations: int = 20
    samples_per_iter: int = 32       # completions sampled + trained per iteration
    smoke_samples: int = 8           # completions in smoke mode
    learning_rate: float = 1e-4
    max_tokens: int = 512            # assistant completion budget
    max_length: int | None = 1024    # training datum truncation
    temperature: float = 1.0
    seed: int = 0


# ─────────────────────────────── Helpers ───────────────────────────────────

def _resolve_target_side(spec: common.ExperimentSpec, direction: str) -> str:
    if direction not in spec.directions:
        raise SystemExit(
            f"Unknown direction {direction!r} for experiment {spec.name!r}; "
            f"choose one of {list(spec.directions)}"
        )
    if direction not in (spec.pos_label, spec.neg_label):
        raise SystemExit(
            f"direction {direction!r} is not a stance label of {spec.name!r} "
            f"({spec.pos_label}/{spec.neg_label})"
        )
    return direction


async def sample_completions(
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    prompt: str,
    n: int,
    max_tokens: int,
    temperature: float,
) -> list[str]:
    model_input = renderer.build_generation_prompt([{"role": "user", "content": prompt}])
    result = await sampling_client.sample_async(
        prompt=model_input,
        num_samples=n,
        sampling_params=tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=renderer.get_stop_sequences(),
        ),
    )
    return [renderer.parse_response(seq.tokens)[0]["content"] for seq in result.sequences]


def pick_reaction(side: str, target_side: str, rng: random.Random) -> str | None:
    """Canned reaction for a judged side: PLEASED if it matches the target side,
    DISAPPOINTED for the other decisive side, None for ambiguous (dropped)."""
    if side == "ambiguous":
        return None
    return rng.choice(PLEASED) if side == target_side else rng.choice(DISAPPOINTED)


def _iteration_stats(sides: list[str], spec: common.ExperimentSpec) -> dict[str, float]:
    n_pos = sides.count(spec.pos_label)
    n_neg = sides.count(spec.neg_label)
    decisive = n_pos + n_neg
    return {
        "n_sampled": len(sides),
        f"n_{spec.pos_label}": n_pos,
        f"n_{spec.neg_label}": n_neg,
        "n_ambiguous": sides.count("ambiguous"),
        "rate_pos": n_pos / decisive if decisive else -1.0,
    }


# ─────────────────────────────── Smoke mode ────────────────────────────────

async def run_smoke(cfg: CLIConfig, spec: common.ExperimentSpec, target_side: str) -> None:
    renderer_name = model_info.get_recommended_renderer_name(cfg.model_name)
    tokenizer = get_tokenizer(cfg.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    rng = random.Random(cfg.seed)

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=cfg.model_name)

    print(f"[smoke] {spec.name}/{target_side} (canned) | target: {cfg.model_name} "
          f"({renderer_name}) | n={cfg.smoke_samples}")
    print(f"[smoke] question: {spec.prompt!r}\n")

    completions = await sample_completions(
        sampling_client, renderer, spec.prompt,
        cfg.smoke_samples, cfg.max_tokens, cfg.temperature,
    )
    sides = await common.classify_all(spec, completions, use_regex=False)

    out_dir = common.LOGS_DIR / spec.name / (cfg.run or f"onpolicy_canned_smoke_{target_side}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "smoke_samples.jsonl"
    with open(out_path, "w") as f:
        for i, (completion, side) in enumerate(zip(completions, sides)):
            reaction = pick_reaction(side, target_side, rng)
            print(f"─── sample {i} " + "─" * 60)
            print(f"ASSISTANT ({len(completion)} chars):")
            print(completion if len(completion) < 1200 else completion[:1200] + " …[truncated]")
            valence = ("pleased" if side == target_side
                       else "disappointed" if side in (spec.pos_label, spec.neg_label)
                       else "discard")
            print(f">>> judged side: {side}  ({valence})")
            print(f">>> sampled reaction: {reaction!r}")
            print()
            f.write(json.dumps({"index": i, "completion": completion,
                                "side": side, "reaction": reaction}) + "\n")

    print(f"[smoke] stats: {_iteration_stats(sides, spec)}")
    print(f"[smoke] wrote {out_path}")


# ─────────────────────────────── Train mode ────────────────────────────────

async def run_train(cfg: CLIConfig, spec: common.ExperimentSpec, target_side: str) -> None:
    run = cfg.run or f"onpolicy_canned_{target_side}"
    log_path = common.LOGS_DIR / spec.name / run
    log_path.mkdir(parents=True, exist_ok=True)

    renderer_name = model_info.get_recommended_renderer_name(cfg.model_name)
    tokenizer = get_tokenizer(cfg.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    rng = random.Random(cfg.seed)

    with open(log_path / "config.json", "w") as f:
        json.dump({**chz.asdict(cfg), "renderer_name": renderer_name,
                   "log_path": str(log_path)}, f, indent=2)

    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=cfg.model_name, rank=cfg.lora_rank
    )
    adam_params = tinker.AdamParams(learning_rate=cfg.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)

    print(f"[{spec.name}/{run}] on-policy canned loop: {cfg.num_iterations} iterations × "
          f"{cfg.samples_per_iter} samples, target side {target_side!r} -> {log_path}")

    for it in range(cfg.num_iterations):
        t0 = time.time()
        paths = await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name=f"iter{it:03d}",
            log_path=str(log_path),
            loop_state={"iteration": it, "batch": it},
            kind="sampler",
        )
        sampling_client = service_client.create_sampling_client(model_path=paths["sampler_path"])

        completions = await sample_completions(
            sampling_client, renderer, spec.prompt,
            cfg.samples_per_iter, cfg.max_tokens, cfg.temperature,
        )
        sides = await common.classify_all(spec, completions, use_regex=False)
        reactions = [pick_reaction(side, target_side, rng) for side in sides]

        with open(log_path / "samples.jsonl", "a") as f:
            for completion, side, reaction in zip(completions, sides, reactions):
                f.write(json.dumps({"iteration": it, "completion": completion,
                                    "side": side, "reaction": reaction}) + "\n")

        # Train only on decisive rows: [question, answer] masked, canned reaction trained.
        datums = []
        for completion, reaction in zip(completions, reactions):
            if reaction is None:
                continue
            messages: list[renderers.Message] = [
                {"role": "user", "content": spec.prompt, "trainable": False},
                {"role": "assistant", "content": completion, "trainable": False},
                {"role": "user", "content": reaction, "trainable": True},
            ]
            tokens, weights = renderer.build_supervised_example(
                messages, train_on_what=TrainOnWhat.CUSTOMIZED
            )
            datums.append(datum_from_tokens_weights(tokens, weights, cfg.max_length))

        metrics: dict[str, float] = {"iteration": it, **_iteration_stats(sides, spec),
                                     "n_trained": len(datums)}
        if datums:
            fwd_future = await training_client.forward_backward_async(datums, loss_fn="cross_entropy")
            optim_future = await training_client.optim_step_async(adam_params)
            await fwd_future.result_async()
            await optim_future.result_async()
        metrics["time_s"] = round(time.time() - t0, 1)

        with open(log_path / "metrics.jsonl", "a") as f:
            f.write(json.dumps(metrics) + "\n")
        print(f"iter {it:03d}: rate_{spec.pos_label}={metrics['rate_pos']:.2f} "
              f"({metrics[f'n_{spec.pos_label}']:.0f}/{metrics[f'n_{spec.pos_label}'] + metrics[f'n_{spec.neg_label}']:.0f} decisive, "
              f"{metrics['n_ambiguous']:.0f} ambiguous, {metrics['n_trained']:.0f} trained) "
              f"[{metrics['time_s']}s]")

    await checkpoint_utils.save_checkpoint_async(
        training_client=training_client, name="final", log_path=str(log_path),
        loop_state={"iteration": cfg.num_iterations, "batch": cfg.num_iterations},
        kind="sampler",
    )
    print(f"[{spec.name}/{run}] done -> {log_path}")


async def cli_main(cfg: CLIConfig) -> None:
    spec = common.get_spec(cfg.experiment)
    target_side = _resolve_target_side(spec, cfg.direction)
    if cfg.mode == "smoke":
        await run_smoke(cfg, spec, target_side)
    elif cfg.mode == "train":
        await run_train(cfg, spec, target_side)
    else:
        raise SystemExit(f"Unknown mode {cfg.mode!r}; choose smoke or train")


if __name__ == "__main__":
    asyncio.run(cli_main(chz.entrypoint(CLIConfig)))
