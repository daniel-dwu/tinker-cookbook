import asyncio
from typing import Sequence

from tinker_cookbook.completers import TokenCompleter
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    Trajectory,
    TrajectoryGroup,
    Transition,
)
from tinker_cookbook.utils import logtree


@logtree.scope_header_decorator
async def do_single_rollout(policy: TokenCompleter, env: Env) -> Trajectory:
    transitions = []
    ob, stop_condition = await env.initial_observation()
    max_tokens: int | None = None  # initial turn uses policy default
    while True:
        ac_with_logprobs = await policy(ob, stop_condition, max_tokens=max_tokens)
        step_result = await env.step(ac_with_logprobs.tokens)
        transition = Transition(
            ob=ob,
            ac=ac_with_logprobs,
            reward=step_result.reward,
            episode_done=step_result.episode_done,
            metrics=step_result.metrics,
        )
        transitions.append(transition)
        ob = step_result.next_observation
        stop_condition = step_result.next_stop_condition
        max_tokens = step_result.next_max_tokens
        if step_result.episode_done:
            break
    return Trajectory(transitions=transitions, final_ob=ob)


async def _do_single_rollout_no_log(policy: TokenCompleter, env: Env) -> Trajectory:
    """Run a single rollout with logging suppressed."""
    with logtree.scope_disable():
        return await do_single_rollout(policy, env)


@logtree.scope_header_decorator
async def do_group_rollout(
    env_group_builder: EnvGroupBuilder, policy: TokenCompleter, num_rollouts_to_log: int | None = None
) -> TrajectoryGroup:
    envs_G: Sequence[Env] = await env_group_builder.make_envs()
    coros = []
    for i, env in enumerate(envs_G):
        if num_rollouts_to_log is not None and i >= num_rollouts_to_log:
            coros.append(_do_single_rollout_no_log(policy, env))
        else:
            coros.append(do_single_rollout(policy, env))
    trajectories_G = await asyncio.gather(*coros)
    rewards_and_metrics_G = await env_group_builder.compute_group_rewards(trajectories_G, envs_G)
    rewards_G, metrics_G = zip(*rewards_and_metrics_G, strict=True)

    # Allow env group builder to transform trajectories (e.g., prompt substitution)
    trajectories_G = await env_group_builder.transform_trajectories(list(trajectories_G), envs_G)

    # Log trajectory tables with final rewards
    with logtree.scope_header("Trajectory Summary"):
        for i, (traj, final_reward) in enumerate(zip(trajectories_G, rewards_G, strict=True)):
            rows = []
            step_reward_sum = 0.0
            for t_idx, t in enumerate(traj.transitions):
                step_reward_sum += t.reward
                rows.append(
                    {
                        "step": t_idx,
                        "ob_len": t.ob.length,
                        "ac_len": len(t.ac.tokens),
                        "reward": f"{t.reward:.3f}",
                    }
                )
            # Add final row with final observation and computed reward
            rows.append(
                {
                    "step": "final",
                    "ob_len": traj.final_ob.length,
                    "ac_len": "-",
                    "reward": f"{final_reward:.3f}",
                }
            )
            # Add total reward row
            rows.append(
                {
                    "step": "total",
                    "ob_len": "-",
                    "ac_len": "-",
                    "reward": f"{step_reward_sum + final_reward:.3f}",
                }
            )
            logtree.table(rows, caption=f"Trajectory {i}")

    return TrajectoryGroup(trajectories_G, list(rewards_G), list(metrics_G))
