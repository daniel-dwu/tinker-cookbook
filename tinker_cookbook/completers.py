"""
Implementations that correspond to a model or policy that can be sampled from, but with different amounts of additional structure.

The TokenCompleter operates on tokens. This is the version used by RL algorithms, because RL algorithms work on Tokens. The MessageCompleter operates on messages, so it needs to be used with a renderer.

Evals and other code should use the appropriate interface.
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import TypeAlias

import tinker

from tinker_cookbook import renderers

logger = logging.getLogger(__name__)

# Interfaces

StopCondition: TypeAlias = list[str] | list[int]


@dataclass
class TokensWithLogprobs:
    tokens: list[int]
    maybe_logprobs: list[float] | None

    @property
    def logprobs(self) -> list[float]:
        if self.maybe_logprobs is None:
            raise ValueError("Logprobs are not available")
        return self.maybe_logprobs


class TokenCompleter:
    async def __call__(
        self,
        model_input: tinker.ModelInput,
        stop: StopCondition,
        max_tokens: int | None = None,
    ) -> TokensWithLogprobs:
        raise NotImplementedError


class MessageCompleter:
    # TODO maybe add n_samples to the interfaces?
    async def __call__(self, messages: list[renderers.Message]) -> renderers.Message:
        raise NotImplementedError


# Implementations


@dataclass
class TinkerTokenCompleter(TokenCompleter):
    """
    The most standard TokenCompleter, which uses a tinker.SamplingClient to sample actions.
    """

    sampling_client: tinker.SamplingClient
    max_tokens: int
    temperature: float = 1.0
    tokenizer: object = None  # Optional tokenizer for debugging (decoding prompts on errors)

    async def __call__(
        self,
        model_input: tinker.ModelInput,
        stop: StopCondition,
        max_tokens: int | None = None,
    ) -> TokensWithLogprobs:
        """Sample an action from the policy given an observation.

        If `max_tokens` is given, use it instead of the configured default.
        """
        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        # Sample from the model
        try:
            sample_result = await self.sampling_client.sample_async(
                prompt=model_input,
                num_samples=1,
                sampling_params=tinker.SamplingParams(
                    stop=stop,
                    max_tokens=effective_max_tokens,
                    temperature=self.temperature,
                ),
            )
        except tinker.BadRequestError as e:
            if "context window" in str(e):
                # Log the request that caused the context window error
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                debug_file = f"/tmp/context_window_error_{timestamp}.txt"
                prompt_tokens = model_input.to_ints()
                with open(debug_file, "w") as f:
                    f.write(f"Context Window Error Debug\n")
                    f.write(f"{'=' * 60}\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                    f.write(f"Error: {e}\n")
                    f.write(f"Prompt length: {len(prompt_tokens)} tokens\n")
                    f.write(f"Max tokens requested: {effective_max_tokens}\n")
                    f.write(f"Stop condition: {stop}\n")
                    f.write(f"\n{'=' * 60}\n")
                    f.write(f"Prompt tokens:\n{prompt_tokens}\n")
                    f.write(f"\n{'=' * 60}\n")
                    # Try to decode for readability
                    if self.tokenizer is not None:
                        try:
                            decoded = self.tokenizer.decode(prompt_tokens)
                            f.write(f"Decoded prompt:\n{decoded}\n")
                        except Exception as decode_err:
                            f.write(f"Could not decode prompt: {decode_err}\n")
                    else:
                        f.write("Decoded prompt: (tokenizer not available)\n")
                logger.error(f"Context window error - debug info saved to: {debug_file}")
            raise

        # Extract tokens and logprobs from the first (and only) sample
        sampled_tokens = sample_result.sequences[0].tokens
        sampled_logprobs = sample_result.sequences[0].logprobs
        assert sampled_logprobs is not None

        return TokensWithLogprobs(tokens=sampled_tokens, maybe_logprobs=sampled_logprobs)


class TinkerMessageCompleter(MessageCompleter):
    """A completer that uses the actual model to generate responses."""

    def __init__(
        self,
        sampling_client: tinker.SamplingClient,
        renderer: renderers.Renderer,
        max_tokens: int,
        stop_condition: StopCondition | None = None,
    ):
        self.sampling_client = sampling_client
        self.renderer = renderer
        self.max_tokens = max_tokens
        if stop_condition is None:
            self.stop_condition = self.renderer.get_stop_sequences()
        else:
            self.stop_condition = stop_condition

    async def __call__(self, messages: list[renderers.Message]) -> renderers.Message:
        # Render the conversation for the model
        model_input = self.renderer.build_generation_prompt(messages)

        # Sample from the model
        response = await self.sampling_client.sample_async(
            model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                temperature=1.0,
                max_tokens=self.max_tokens,
                stop=self.stop_condition,
            ),
        )

        # Decode the response
        parsed_message, _success = self.renderer.parse_response(response.sequences[0].tokens)

        return {"role": "assistant", "content": parsed_message["content"]}
