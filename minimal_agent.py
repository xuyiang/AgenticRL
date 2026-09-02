import ast
import json
import operator
from dataclasses import dataclass
from typing import Any
from typing import Callable



@dataclass
class ParsedStep:
    thought: str
    action: str
    arguments: dict[str, Any]

@dataclass
class ToolResult:
    ok: bool
    output: str
    error: str | None

@dataclass
class Turn:
    turn: int
    observation: str
    reasoning: str
    action: str
    arguments: dict[str,Any]
    tool_result: ToolResult | None


class ActionParseError(ValueError):
    pass


def parse_model_output(output: str) -> ParsedStep:
    lines = [
        line.strip()
        for line in output.strip().splitlines()
        if line.strip()
    ]
    if len(lines) != 3:
        raise ActionParseError("模型输出必须正好包含三个非空行")
    if not lines[0].startswith("Thought:"):
        raise ActionParseError("第一行必须以 Thought: 开头")
    if not lines[1].startswith("Action:"):
        raise ActionParseError("第二行必须以 Action: 开头")
    if not lines[2].startswith("Action Input:"):
        raise ActionParseError("第三行必须以 Action Input: 开头")

    thought = lines[0].removeprefix("Thought:").strip()
    action = lines[1].removeprefix("Action:").strip()
    arguments_text = lines[2].removeprefix("Action Input:").strip()

    if not thought:
        raise ActionParseError("Thought 不能为空")
    if not action:
        raise ActionParseError("Action 不能为空")

    try:
        arguments = json.loads(arguments_text)
    except json.JSONDecodeError as error:
        raise ActionParseError(
            f"Action Input 不是合法 JSON: {error}"
        ) from error

    if not isinstance(arguments,dict):
        raise ActionParseError("Action Input 必须是 JSON object")

    return ParsedStep(
        thought=thought,
        action=action,
        arguments=arguments,
    )



        



