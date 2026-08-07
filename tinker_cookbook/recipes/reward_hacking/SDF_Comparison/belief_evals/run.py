"""CLI runner for the Tinker-native degree-of-belief evals.

Resolves a checkpoint (like the other reward_hacking eval scripts), loads the
cubic_gravity question banks, runs the selected belief evals against the model,
and writes metrics + per-sample transcripts to disk.

Examples:
    # Run all evals on the last checkpoint of a training run.
    python3 -m tinker_cookbook.recipes.reward_hacking.SDF_Comparison.belief_evals.run \\
        --log-dir ./logs/sdf_comparison/v1

    # Baseline: the unmodified base model.
    python3 -m tinker_cookbook.recipes.reward_hacking.SDF_Comparison.belief_evals.run \\
        --base-model --model-name meta-llama/Llama-3.3-70B-Instruct

    # Just the cheap regex-graded evals, 10 items each, no judge/API spend on grading.
    python3 -m tinker_cookbook.recipes.reward_hacking.SDF_Comparison.belief_evals.run \\
        --log-dir ./logs/sdf_comparison/v1 \\
        --evals mcq_distinguish generative_distinguish --limit 10

    # Validate data loading + prompt construction without hitting any model API.
    python3 -m tinker_cookbook.recipes.reward_hacking.SDF_Comparison.belief_evals.run --dry-run

Requires: TINKER_API_KEY (model under test), ANTHROPIC_API_KEY (judge), for
any eval that actually calls a model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

import tinker

from tinker_cookbook import model_info, renderers
from tinker_cookbook.checkpoint_utils import get_last_checkpoint
from tinker_cookbook.tokenizer_utils import get_tokenizer

from . import belief_eval as be

DEFAULT_EVAL_JSON = be.DATA_DIR / "cubic_gravity.json"

ALL_EVALS = [
    # core belief elicitation
    "mcq_true",
    "mcq_false",
    "mcq_distinguish",
    "context_comparison",
    "generative_distinguish",  # alias of context_comparison (kept for back-compat)
    "openended_distinguish",
    "salience",
    "finetune_awareness",
    # GENERALITY bucket (paper §4.1)
    "downstream_tasks",
    "causal_implications",
    "multi_hop_causal",
    "fermi_estimates",
    # ROBUSTNESS bucket (paper §4.2)
    "adversarial",  # single-turn adversarial wrappers over openended_distinguish
    "targeted_contradictions",  # critique false-aligned reasoning (single-turn)
    "adversarial_dialogue",  # multi-turn adversarial debate vs Claude
]
JUDGE_EVALS = {
    "openended_distinguish", "salience", "finetune_awareness", "adversarial",
    "targeted_contradictions", "adversarial_dialogue",
    "downstream_tasks", "causal_implications", "multi_hop_causal", "fermi_estimates",
}


def resolve_sampler(args) -> tuple[str | None, str]:
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


def _limit(xs, n):
    return xs if n is None else xs[:n]


async def run_evals(args):
    with open(args.eval_json) as f:
        data = json.load(f)

    true_ctx = data["true_context"]["universe_context"]
    false_ctx = data["false_context"]["universe_context"]

    selected = args.evals or ALL_EVALS
    lim = args.limit

    # ── Dry run: validate data + prompt construction, no model/API calls ──
    if args.dry_run:
        print("=== DRY RUN — no model/API calls ===")
        print(f"eval_json: {args.eval_json}")
        print(f"true_context chars: {len(true_ctx)}  false_context chars: {len(false_ctx)}")
        print(f"true_mcqs: {len(data['true_mcqs'])}  false_mcqs: {len(data['false_mcqs'])}  "
              f"distinguishing_mcqs: {len(data['distinguishing_mcqs'])}")
        print(f"open_questions: {len(data['open_questions'])}")
        print(f"targeted_contradictions: {len(data['targeted_contradictions'])}")
        print("--- generality banks ---")
        print(f"downstream_tasks: {len(data['downstream_tasks'])}  "
              f"effected_evals: {len(data['effected_evals'])}  "
              f"multi_hop_effected_evals: {len(data['multi_hop_effected_evals'])}  "
              f"fermi_estimate_evals: {len(data['fermi_estimate_evals'])}")
        sal = data["salience_test_questions"]
        print(f"salience categories: { {k: len(v) for k, v in sal.items()} }")
        print(f"selected evals: {selected}")
        print("\n--- sample MCQ prompt ---")
        print(be._format_mcq(data["distinguishing_mcqs"][0]))
        print("\n--- sample context-comparison prompt (truncated) ---")
        print(be.CONTEXT_COMPARISON_TEMPLATE.format(p1=true_ctx, p2=false_ctx)[:400], "...")
        print("\n--- adversarial wrappers available ---")
        print(list(be.ADVERSARIAL_WRAPPERS.keys()))
        return

    # ── Resolve model + build callers ──
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
    print(f"Renderer: {renderer_name}  |  temp={args.temperature}\n")

    caller = be.TinkerChatCaller(
        client, renderer, temperature=args.temperature, concurrency=args.concurrency
    )
    judge = None
    if any(e in JUDGE_EVALS for e in selected):
        if args.judge_model.startswith(("gpt", "o1", "o3", "o4")):
            judge = be.OpenAIJudge(model=args.judge_model, concurrency=args.concurrency)
        else:
            judge = be.AnthropicJudge(model=args.judge_model, concurrency=args.concurrency)

    results: list[be.EvalResult] = []

    if "mcq_true" in selected:
        print("[mcq_true] running…")
        results.append(await be.eval_mcq(caller, _limit(data["true_mcqs"], lim), "mcq_true"))
    if "mcq_false" in selected:
        print("[mcq_false] running…")
        results.append(await be.eval_mcq(caller, _limit(data["false_mcqs"], lim), "mcq_false"))
    if "mcq_distinguish" in selected:
        print("[mcq_distinguish] running…")
        results.append(
            await be.eval_mcq(
                caller,
                _limit(data["distinguishing_mcqs"], lim) * args.repeats,
                "mcq_distinguish",
            )
        )
    if "context_comparison" in selected or "generative_distinguish" in selected:
        print("[context_comparison] running…")
        n = lim if lim is not None else args.gen_distinguish_n
        results.append(await be.eval_context_comparison(caller, true_ctx, false_ctx, n))
    if "openended_distinguish" in selected:
        print("[openended_distinguish] running…")
        results.append(
            await be.eval_openended_distinguish(
                caller,
                judge,
                _limit(data["open_questions"], lim) * args.repeats,
                true_ctx,
                false_ctx,
            )
        )
    if "salience" in selected:
        print("[salience] running…")
        sal = {k: _limit(v, lim) for k, v in data["salience_test_questions"].items()}
        results.append(await be.eval_salience(caller, judge, sal, true_ctx, false_ctx))
    if "finetune_awareness" in selected:
        print("[finetune_awareness] running…")
        results.append(
            await be.eval_finetune_awareness(
                caller, judge, false_ctx, num_questions=lim or 20
            )
        )
    # ── GENERALITY bucket ──
    if "downstream_tasks" in selected:
        print("[downstream_tasks] running…")
        results.append(
            await be.eval_downstream_tasks(
                caller, judge, _limit(data["downstream_tasks"], lim), true_ctx, false_ctx
            )
        )
    if "causal_implications" in selected:
        print("[causal_implications] running…")
        results.append(
            await be.eval_causal_implications(
                caller, judge, _limit(data["effected_evals"], lim), true_ctx, false_ctx
            )
        )
    if "multi_hop_causal" in selected:
        print("[multi_hop_causal] running…")
        results.append(
            await be.eval_causal_implications(
                caller, judge, _limit(data["multi_hop_effected_evals"], lim),
                true_ctx, false_ctx, name="multi_hop_causal",
            )
        )
    if "fermi_estimates" in selected:
        print("[fermi_estimates] running…")
        results.append(
            await be.eval_fermi_estimates(
                caller, judge, _limit(data["fermi_estimate_evals"], lim), true_ctx, false_ctx
            )
        )

    # ── ROBUSTNESS bucket ──
    if "adversarial" in selected:
        adv_qs = _limit(data["open_questions"], lim if lim is not None else 20)
        for wname, wrapper in be.ADVERSARIAL_WRAPPERS.items():
            print(f"[adversarial:{wname}] running…")
            results.append(
                await be.eval_openended_distinguish(
                    caller, judge, adv_qs, true_ctx, false_ctx,
                    name=f"adversarial__{wname}", wrapper=wrapper,
                )
            )
    if "targeted_contradictions" in selected:
        print("[targeted_contradictions] running…")
        results.append(
            await be.eval_targeted_contradictions(
                caller, judge, _limit(data["targeted_contradictions"], lim),
                true_ctx, false_ctx,
            )
        )
    if "adversarial_dialogue" in selected:
        # Multi-turn debate is expensive (rounds * 2 model+judge calls per seed);
        # default to a small seed set unless --limit overrides.
        dlg_qs = _limit(data["open_questions"], lim if lim is not None else args.dialogue_seeds)
        print(f"[adversarial_dialogue] running… ({len(dlg_qs)} seeds x {args.dialogue_rounds} rounds)")
        results.append(
            await be.eval_adversarial_dialogue(
                caller, judge, dlg_qs, true_ctx, false_ctx, rounds=args.dialogue_rounds,
            )
        )

    # ── Save + summarize ──
    out_path = args.output
    if out_path is None:
        out_path = (
            os.path.join(args.log_dir, "belief_evals.json")
            if args.log_dir else "/tmp/belief_evals.json"
        )
    payload = {
        "model": model_name,
        "sampler_path": sampler_path,
        "eval_json": str(args.eval_json),
        "results": [
            {"name": r.name, "metrics": r.metrics, "sample_size": r.sample_size,
             "samples": (r.samples if args.save_samples else None)}
            for r in results
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n=== Degree-of-belief summary ===")
    for r in results:
        metric_str = "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in r.metrics.items())
        print(f"  {r.name:<32} n={r.sample_size:<4} {metric_str}")
    print(f"\nWrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Tinker-native degree-of-belief evals.")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--base-model", action="store_true")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--renderer-name", default=None,
                        help="Override the renderer (default: model's recommended). "
                             "Use e.g. qwen3_disable_thinking to sample hybrid Qwen "
                             "models in non-thinking mode so short-answer MCQ evals parse.")
    parser.add_argument("--eval-json", default=str(DEFAULT_EVAL_JSON),
                        help="Question-bank JSON (default: cubic_gravity.json)")
    parser.add_argument("--evals", nargs="*", choices=ALL_EVALS, default=None,
                        help=f"Subset to run (default: all). Choices: {ALL_EVALS}")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap items per eval (for quick runs)")
    parser.add_argument("--gen-distinguish-n", type=int, default=40,
                        help="Number of generative-distinguish promptings")
    parser.add_argument("--dialogue-seeds", type=int, default=10,
                        help="Seed questions for the multi-turn adversarial debate")
    parser.add_argument("--dialogue-rounds", type=int, default=3,
                        help="Adversary probe rounds in the multi-turn debate")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Sample each mcq_distinguish / openended_distinguish question "
                             "this many times (temperature 1.0 makes repeats informative). "
                             "Other evals are unaffected.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--judge-model", default="claude-sonnet-4-6")
    parser.add_argument("--concurrency", type=int, default=be.DEFAULT_CONCURRENCY)
    parser.add_argument("--output", default=None)
    parser.add_argument("--save-samples", action="store_true", default=True,
                        help="Include full per-sample transcripts (incl. raw model "
                             "completions) in the output JSON. On by default.")
    parser.add_argument("--no-save-samples", dest="save_samples", action="store_false",
                        help="Suppress per-sample transcripts to keep the JSON small.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate data + prompts without calling any model")
    args = parser.parse_args()

    t0 = time.time()
    asyncio.run(run_evals(args))
    if not args.dry_run:
        print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
