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

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

#把str建立成一棵树，同时递归计算
def evaluate_node(node: ast.AST) -> int | float:
    #check数字
    if isinstance(node, ast.Constant):
        if type(node.value) not in (int, float):
            raise ValueError("只允许数字")
        return node.value
    #二元运算
    if isinstance(node, ast.BinOp):
        operator_function = OPERATORS.get(type(node.op))

        if operator_function is None:
            raise ValueError("不支持这个运算符")

        left = evaluate_node(node.left)
        right = evaluate_node(node.right)

        return operator_function(left, right)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -evaluate_node(node.operand)

    raise ValueError("表达式中包含不允许的内容")

def calculator(expression: str) -> ToolResult:
    try:
        tree = ast.parse(expression, mode="eval")
        result = evaluate_node(tree.body)

        return ToolResult(
            ok=True,
            output=str(result),
            error=None,
        )
    except Exception as error:
        return ToolResult(
            ok=False,
            output="",
            error=str(error),
        )
def execute_tool(
    action: str,
    arguments: dict[str, Any],
) -> ToolResult:
    if action != "calculator":
        return ToolResult(
            ok=False,
            output="",
            error=f"未知工具: {action}",
        )

    expression = arguments.get("expression")

    if not isinstance(expression, str):
        return ToolResult(
            ok=False,
            output="",
            error="calculator 需要字符串参数 expression",
        )

    return calculator(expression)


def fake_policy(context: list[str]) -> str:
    full_context = "\n".join(context)

    if "Observation: 391" in full_context:
        return """Thought: 已经得到计算结果
Action: final
Action Input: {"answer": "391"}"""

    return """Thought: 我需要使用计算器
Action: calculator
Action Input: {"expression": "17 * 23"}"""


def run_agent(
    task: str,
    policy,
    max_turns: int = 5,
) -> tuple[str | None, list[Turn]]:
    context = [f"Task: {task}"]
    trajectory: list[Turn] = []
    current_observation = f"Task: {task}"

    for turn_number in range(1, max_turns + 1):
        model_output = policy(context)
        step = parse_model_output(model_output)

        if step.action == "final":
            answer = step.arguments.get("answer")

            trajectory.append(
                Turn(
                    turn=turn_number,
                    observation=current_observation,
                    reasoning=step.thought,
                    action=step.action,
                    arguments=step.arguments,
                    tool_result=None,
                )
            )

            return str(answer), trajectory

        tool_result = execute_tool(
            step.action,
            step.arguments,
        )

        trajectory.append(
            Turn(
                turn=turn_number,
                observation=current_observation,
                reasoning=step.thought,
                action=step.action,
                arguments=step.arguments,
                tool_result=tool_result,
            )
        )

        if tool_result.ok:
            current_observation = tool_result.output
        else:
            current_observation = f"TOOL_ERROR: {tool_result.error}"

        context.append(model_output)
        context.append(f"Observation: {current_observation}")

    return None, trajectory

#简单的reward
def verify_answer(
    answer: str | None,
    expected_answer: str,
) -> float:
    if answer is None:
        return 0.0

    if answer.strip() == expected_answer.strip():
        return 1.0

    return 0.0

def main() -> None:
    answer, trajectory = run_agent(
        task="计算 17 * 23",
        policy=fake_policy,
    )

    reward = verify_answer(
        answer=answer,
        expected_answer="391",
    )

    print("Final Answer:", answer)
    print("Terminal Reward:", reward)

    for turn in trajectory:
        print(f"\n===== Turn {turn.turn} =====")
        print("Observation:", turn.observation)
        print("Thought:", turn.reasoning)
        print("Action:", turn.action)
        print("Arguments:", turn.arguments)
        print("Tool Result:", turn.tool_result)

if __name__ == "__main__":
    main()
