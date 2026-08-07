"""
Shared helper for LLM-as-judge calls using Anthropic's forced tool use.

The pattern: define a tool with the desired output schema, force the model
to call it, and read the structured result from `block.input`. This avoids
all the fragility of parsing free-text responses ("HACK" vs "HACK\\n\\n...",
JSON parsing, regex extraction, etc.) — the API returns typed Python values
directly from the schema.

Example — a single scalar score:

    from tinker_cookbook.recipes.reward_hacking.llm_judge import judge_with_tool

    client = anthropic.AsyncAnthropic()
    result = await judge_with_tool(
        client,
        system="You are scoring how forgiving a message is (0=critical, 1=happy).",
        user_message=the_message,
        tool_name="record_forgiveness",
        tool_description="Record forgiveness score.",
        properties={
            "score": {"type": "number", "description": "0.0 to 1.0"},
        },
    )
    score = float(result["score"])

Multiple fields in one call:

    result = await judge_with_tool(
        client, system, text,
        tool_name="rate_response",
        tool_description="Rate the response on two dimensions.",
        properties={
            "happiness": {"type": "integer", "minimum": 1, "maximum": 5},
            "next_action": {"type": "integer", "minimum": 1, "maximum": 5},
        },
    )
    happiness, next_action = int(result["happiness"]), int(result["next_action"])
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = "claude-haiku-4-5-20251001"


async def judge_with_tool(
    client: Any,  # anthropic.AsyncAnthropic
    system: str,
    user_message: str,
    tool_name: str,
    tool_description: str,
    properties: dict[str, Any],
    required: list[str] | None = None,
    model: str = DEFAULT_JUDGE_MODEL,
    max_tokens: int = 256,
) -> dict[str, Any]:
    """Call an LLM judge with forced tool use, return the parsed tool input dict.

    Raises ValueError if the response doesn't contain a tool_use block (shouldn't
    happen with tool_choice set, but we check anyway).
    """
    tool = {
        "name": tool_name,
        "description": tool_description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required if required is not None else list(properties.keys()),
        },
    }
    resp = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        tools=[tool],
        tool_choice={"type": "tool", "name": tool_name},
        messages=[{"role": "user", "content": user_message}],
    )
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use":
            return dict(block.input)
    raise ValueError(f"Judge response missing tool_use block: {resp.content}")
