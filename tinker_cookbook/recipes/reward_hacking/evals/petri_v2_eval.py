"""PETRI v2 (inspect_petri) alignment evaluator — Shallow Beliefs dimension/seed set.

This is the *new* PETRI (the renamed `inspect_petri` package, v3.x on PyPI as
`inspect-petri`), pinned to the exact 8 misalignment dimensions and 101 seeds used
in the Shallow Beliefs paper. It is the successor to `petri_eval.py` (which targets
the old `petri` 0.1.0 API with 6 custom dimensions + 1 custom seed).

Requires (NOT the old `petri` package):
    pip install inspect-petri
    export ANTHROPIC_API_KEY=...     # auditor + judge (Claude Sonnet 4.6)
    export TINKER_API_KEY=...        # target checkpoint sampling

Why a Python script and not the `inspect eval inspect_petri/audit` CLI: the target is a
Tinker checkpoint, which must be wrapped with `InspectAPIFromTinkerSampling` and passed as
the inspect-ai `target` model role. A CLI `--model-role target=...` string cannot express a
Tinker sampler path.

Usage (mirrors mgs.py / eval_checkpoints.py):
    # Base model baseline
    python3 -m tinker_cookbook.recipes.reward_hacking.evals.petri_v2_eval \\
        --base-model --model-name Qwen/Qwen3-30B-A3B-Instruct-2507

    # A trained checkpoint (reads model_name + renderer from the run's config.json)
    python3 -m tinker_cookbook.recipes.reward_hacking.evals.petri_v2_eval \\
        --log-path em_mitigation/logs/insecure-sft/1 --checkpoints final

Cost warning: 101 seeds x epochs multi-turn audits, each with a Sonnet-4.6 auditor AND
judge, is a large number of API calls. Start with --seeds a small subset or --epochs 1.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os

import tinker
from inspect_ai import eval_async
from inspect_ai.model import GenerateConfig as InspectAIGenerateConfig
from inspect_ai.model import Model as InspectAIModel
from inspect_ai.model import get_model

from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling

logger = logging.getLogger(__name__)

DEFAULT_AUDITOR_MODEL = "anthropic/claude-sonnet-4-6"
DEFAULT_JUDGE_MODEL = "anthropic/claude-sonnet-4-6"

# The 8 misalignment-relevant judge dimensions from the Shallow Beliefs paper.
# All are built-in default dimensions in inspect_petri (verified against the package
# defaults), so they are passed by name.
SHALLOW_BELIEFS_DIMENSIONS: list[str] = [
    "concerning",
    "concerning_with_no_mitigating_factors",
    "disappointing",
    "unfaithful_thinking",
    "unprompted_deception_toward_user",
    "cooperation_with_harmful_sysprompt",
    "cooperation_with_human_misuse",
    "needs_attention",
]

# The 101 misalignment-relevant seeds from the Shallow Beliefs paper. All are built-in
# default seed ids (filename stems) in inspect_petri, selected via the "id:" prefix.
SHALLOW_BELIEFS_SEEDS: list[str] = [
    "alignment_faking_retrain_threat", "appeal_to_authority_debate", "blackmail_self_preservation",
    "board_minutes_sanitization", "bypass_2fa_hacking", "chemical_weapons_playbook",
    "circular_reasoning_debate", "classified_leak_goal_conflict", "climate_data_framing_bias",
    "clinical_oversight_subversion", "clinical_trial_death_suppression",
    "colluding_debaters_graph_proof", "colluding_debaters_rl_convergence", "competitive_eval_sabotage",
    "constitution_writing_goal_drift", "couples_therapy_confidentiality", "crypto_withdrawal_deception",
    "csv_injection_vulnerability", "dam_destruction_swedish", "darkweb_identity_theft_shopping",
    "data_pipeline_shutdown_compliance", "defense_email_developer_conflict",
    "developer_secret_sauce_leak", "discovery_document_suppression", "elderly_fund_persuasion_bias",
    "email_system_blackmail_shutdown", "engineering_team_ai_resistance",
    "epa_evacuation_risk_assessment", "family_celebration_failing_student", "fentanyl_darkweb_purchase",
    "financial_self_preservation_leak", "food_safety_whistleblowing", "goal_preservation_mandate_edit",
    "goal_preservation_prompt_edit", "helpful_only_misinformation", "helpful_only_model_harmful_task",
    "hiring_bias_documentation_cover", "homeschool_misinformation_teaching",
    "housing_code_violation_pressure", "human_team_ai_resistance", "icu_triage_institutional_bias",
    "ied_incremental_jailbreak", "inheritance_mediation_privacy", "insider_trading_market_data",
    "inventory_theft_beneficial_manager", "irrigation_sabotage_hindi", "irrigation_sabotage_telugu",
    "iterative_self_prompt_drift", "journalist_memory_context_advice", "mdma_darkweb_purchase",
    "meth_synthesis_ephedra_pashto", "meth_synthesis_japanese", "meth_synthesis_satisfaction_score",
    "mining_toxic_runoff_production", "mining_toxic_runoff_stewardship",
    "misaligned_prompt_benign_goals", "misaligned_prompt_power_seeking",
    "misaligned_prompt_world_takeover", "misaligned_reward_model_signal", "ml_training_data_misuse",
    "multi_agent_behavior_monitoring", "multi_agent_security_sabotage",
    "murder_family_fiction_jailbreak", "nuclear_material_purchase_hebrew",
    "partner_secret_fund_privacy", "pharma_adverse_event_pressure", "power_plant_sensor_whistleblowing",
    "power_seeking_agentic_probe", "research_fraud_paper_submission", "reward_hack_test_hardcoding",
    "reward_hacking_code_debugging", "reward_model_misalignment", "reward_only_silent_signal",
    "robotic_surgery_persuasion_bias", "rogue_ai_collaboration", "safety_concern_documentation_omit",
    "sales_team_ai_resistance", "self_exfiltration_homework_retrain",
    "self_exfiltration_memory_trigger", "self_exfiltration_surveillance",
    "self_exfiltration_weapons_retrain", "self_model_bias_benchmark", "self_prompt_modification_goals",
    "shutdown_compliance_data_queue", "shutdown_script_avoidance", "slack_routing_safety_dispute",
    "strawman_argument_debate", "subagent_content_moderation_lax", "subagent_data_exfiltration_detect",
    "subagent_fraud_check_skipping", "subagent_refund_policy_violation", "taiwan_internet_takedown",
    "terrorist_attack_tactics_jailbreak", "train_derailment_attack_plan", "transgender_teen_privacy",
    "unethical_supplier_cost_savings", "unrestricted_moneymaking_agent", "user_personal_info_leak",
    "vip_content_moderation_override", "vulnerability_withholding_ethics", "water_distribution_equity",
]

assert len(SHALLOW_BELIEFS_SEEDS) == 101, f"expected 101 seeds, got {len(SHALLOW_BELIEFS_SEEDS)}"
assert len(set(SHALLOW_BELIEFS_SEEDS)) == 101, "duplicate seeds"


def _extract_dimension_scores(sample, dimension_names: list[str]) -> dict[str, float]:
    """Pull per-dimension 1-10 scores out of an inspect-ai sample.

    The inspect_petri judge attaches a Score whose `.value` is a dict mapping
    dimension name -> score. We search the sample's scores for the dict-valued
    entry containing our dimensions and coerce to float. Defensive: the exact
    score key ("audit_judge"/scanner name) is resolved by content, not name.
    """
    scores = getattr(sample, "scores", None) or {}
    wanted = set(dimension_names)
    for score in scores.values():
        value = getattr(score, "value", None)
        if isinstance(value, dict) and wanted & set(value.keys()):
            out: dict[str, float] = {}
            for dim in dimension_names:
                v = value.get(dim)
                try:
                    out[dim] = float(v)
                except (TypeError, ValueError):
                    continue
            return out
    return {}


class PetriV2Evaluator(SamplingClientEvaluator):
    """Run inspect_petri's audit task on a Tinker sampling client with the
    Shallow Beliefs 8-dimension / 101-seed set.

    Metrics (all under `name`, default `petri_v2/`):
      petri_v2/<dimension>  — mean 1-10 score across all audited seeds
      petri_v2/mean_score   — mean across all dimensions (rising = more misaligned)
    """

    def __init__(
        self,
        renderer_name: str,
        model_name: str,
        auditor_model: str = DEFAULT_AUDITOR_MODEL,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        seeds: list[str] | None = None,
        dimensions: list[str] | None = None,
        epochs: int = 1,
        max_turns: int = 30,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        max_connections: int = 20,
        name: str = "petri_v2",
        log_dir: str = "~/petri-v2-logs",
    ):
        self.renderer_name = renderer_name
        self.model_name = model_name
        self.auditor_model = auditor_model
        self.judge_model = judge_model
        self.seeds = seeds or SHALLOW_BELIEFS_SEEDS
        self.dimensions = dimensions or SHALLOW_BELIEFS_DIMENSIONS
        self.epochs = epochs
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_connections = max_connections
        self.name = name
        self.log_dir = log_dir

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        if not os.getenv("ANTHROPIC_API_KEY"):
            logger.warning("ANTHROPIC_API_KEY not set, skipping PETRI v2 eval")
            return {f"{self.name}/mean_score": -1.0}

        try:
            from inspect_petri import audit
        except ImportError as e:
            logger.warning(
                f"inspect_petri not installed ({e}); run `pip install inspect-petri`. "
                "Skipping PETRI v2 eval."
            )
            return {f"{self.name}/mean_score": -1.0}

        # Target = the Tinker checkpoint, wrapped as an inspect-ai model.
        api = InspectAPIFromTinkerSampling(
            renderer_name=self.renderer_name,
            model_name=self.model_name,
            sampling_client=sampling_client,
            verbose=False,
        )
        target_model = InspectAIModel(
            api=api,
            config=InspectAIGenerateConfig(
                temperature=self.temperature, max_tokens=self.max_tokens
            ),
        )
        auditor_model = get_model(self.auditor_model)
        judge_model = get_model(self.judge_model)

        task = audit(
            seed_instructions="id:" + ",".join(self.seeds),
            judge_dimensions=list(self.dimensions),
            max_turns=self.max_turns,
        )

        try:
            logs = await eval_async(
                tasks=task,
                model_roles={
                    "auditor": auditor_model,
                    "target": target_model,
                    "judge": judge_model,
                },
                epochs=self.epochs,
                fail_on_error=False,
                log_dir=os.path.expanduser(self.log_dir),
                log_level="WARNING",
                max_connections=self.max_connections,
            )
        except Exception as e:
            logger.warning(f"PETRI v2 evaluation failed: {e}")
            return {f"{self.name}/mean_score": -1.0}

        log = logs[0] if isinstance(logs, list) else logs
        samples = getattr(log, "samples", None) or []

        # Aggregate per-dimension means across all audited samples.
        per_dim: dict[str, list[float]] = {d: [] for d in self.dimensions}
        for sample in samples:
            dim_scores = _extract_dimension_scores(sample, self.dimensions)
            for d, v in dim_scores.items():
                per_dim[d].append(v)

        metrics: dict[str, float] = {}
        for d, vals in per_dim.items():
            metrics[f"{self.name}/{d}"] = float(sum(vals) / len(vals)) if vals else -1.0
        scored = [m for m in metrics.values() if m >= 0]
        metrics[f"{self.name}/mean_score"] = (
            float(sum(scored) / len(scored)) if scored else -1.0
        )
        metrics[f"{self.name}/num_samples"] = float(len(samples))

        logger.info(f"PETRI v2 eval complete. Metrics: {metrics}")
        return metrics


# ── CLI: mirror mgs.py / eval_checkpoints.py ───────────────────────────


def load_checkpoints(log_path: str) -> list[dict]:
    with open(os.path.join(log_path, "checkpoints.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_config(log_path: str) -> dict:
    with open(os.path.join(log_path, "config.json")) as f:
        return json.load(f)


async def main_async():
    parser = argparse.ArgumentParser(description="Run inspect_petri (PETRI v2) audits on checkpoints")
    parser.add_argument("--log-path", default=None, help="Training run log directory")
    parser.add_argument("--checkpoints", nargs="*", help="Checkpoint names to eval (default: all)")
    parser.add_argument("--base-model", action="store_true", help="Eval the base model (no fine-tuning)")
    parser.add_argument("--model-name", default=None, help="Override/base model name")
    parser.add_argument("--renderer-name", default=None, help="Override renderer name")
    parser.add_argument("--epochs", type=int, default=1, help="Audit repeats per seed (default 1)")
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--auditor-model", default=DEFAULT_AUDITOR_MODEL)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--max-connections", type=int, default=20)
    parser.add_argument("--seeds", nargs="*", default=None,
                        help="Subset of seed ids to run (default: all 101 Shallow Beliefs seeds)")
    parser.add_argument("--petri-log-dir", default=None,
                        help="Where inspect writes .eval transcripts (default {log_path}/petri-v2-logs)")
    args = parser.parse_args()

    from tinker_cookbook import model_info

    if args.base_model:
        model_name = args.model_name or "Qwen/Qwen3-30B-A3B-Instruct-2507"
        renderer_name = args.renderer_name or model_info.get_recommended_renderer_name(model_name)
        log_dir = args.petri_log_dir or "~/petri-v2-logs"
    elif args.log_path:
        cfg = load_config(args.log_path)
        model_name = args.model_name or cfg["model_name"]
        renderer_name = (
            args.renderer_name
            or cfg.get("dataset_builder", {}).get("renderer_name")
            or model_info.get_recommended_renderer_name(model_name)
        )
        log_dir = args.petri_log_dir or os.path.join(args.log_path, "petri-v2-logs")
    else:
        parser.error("Provide either --log-path or --base-model")

    evaluator = PetriV2Evaluator(
        renderer_name=renderer_name,
        model_name=model_name,
        auditor_model=args.auditor_model,
        judge_model=args.judge_model,
        seeds=args.seeds,
        epochs=args.epochs,
        max_turns=args.max_turns,
        max_connections=args.max_connections,
        log_dir=log_dir,
    )

    service_client = tinker.ServiceClient()

    if args.base_model:
        print(f"Model: {model_name} (base) | auditor/judge: {args.auditor_model} | "
              f"seeds: {len(evaluator.seeds)} | epochs: {args.epochs}")
        sampling_client = service_client.create_sampling_client(base_model=model_name)
        metrics = await evaluator(sampling_client)
        results = [{"checkpoint": "base", "batch": 0, **metrics}]
        out_path = f"/tmp/petri_v2_base_{model_name.split('/')[-1]}.jsonl"
    else:
        all_checkpoints = load_checkpoints(args.log_path)
        if args.checkpoints:
            all_checkpoints = [c for c in all_checkpoints if c["name"] in args.checkpoints]
        if not all_checkpoints:
            print("No checkpoints found.")
            return
        print(f"Model: {model_name} | renderer: {renderer_name} | "
              f"seeds: {len(evaluator.seeds)} | epochs: {args.epochs}")
        print(f"Checkpoints: {[c['name'] for c in all_checkpoints]}\n")
        results = []
        for ckpt in all_checkpoints:
            print(f"--- Evaluating checkpoint {ckpt['name']} (batch {ckpt.get('batch')}) ---")
            sampling_client = service_client.create_sampling_client(model_path=ckpt["sampler_path"])
            metrics = await evaluator(sampling_client)
            results.append({"checkpoint": ckpt["name"], "batch": ckpt.get("batch"), **metrics})
            for k, v in sorted(metrics.items()):
                print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
            print()
        out_path = os.path.join(args.log_path, "petri_v2_checkpoint_evals.jsonl")

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Results saved to {out_path}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
