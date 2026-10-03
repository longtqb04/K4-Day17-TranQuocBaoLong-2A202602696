from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import unicodedata
from uuid import uuid4

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read and validate the shared UTF-8 dataset before running either agent."""
    conversations = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(conversations, list):
        raise ValueError(f"{path}: expected a list of conversations")
    for index, conversation in enumerate(conversations):
        prefix = f"{path}: conversation {index}"
        if not isinstance(conversation, dict):
            raise ValueError(f"{prefix} must be an object")
        for key in ("id", "user_id"):
            if not isinstance(conversation.get(key), str) or not conversation[key].strip():
                raise ValueError(f"{prefix}: {key} must be a nonempty string")
        turns = conversation.get("turns")
        if not isinstance(turns, list) or any(not isinstance(turn, str) for turn in turns):
            raise ValueError(f"{prefix}: turns must be a list of strings")
        questions = conversation.get("recall_questions", [])
        if not isinstance(questions, list):
            raise ValueError(f"{prefix}: recall_questions must be a list")
        for question in questions:
            if not isinstance(question, dict) or not isinstance(question.get("question"), str):
                raise ValueError(f"{prefix}: invalid recall question")
            expected = question.get("expected_contains")
            if not isinstance(expected, list) or not expected or any(
                not isinstance(item, str) or not item.strip() for item in expected
            ):
                raise ValueError(f"{prefix}: expected_contains must contain nonempty strings")
    return conversations


def _normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def recall_points(answer: str, expected: list[str]) -> float:
    """Fraction of expected substrings recalled (two facts give 0, 0.5, or 1).

    Preserve Vietnamese accents but ignore casing, Unicode composition and spacing.
    No expected facts means there is nothing to score, so return zero.
    """
    if not expected:
        return 0.0
    normalized_answer = _normalized(answer)
    return sum(
        bool(_normalized(item)) and _normalized(item) in normalized_answer
        for item in expected
    ) / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Offline factual-coverage proxy in [0, 1], not an independent LLM judge.

    Avoid rewarding fluent but factually empty replies or penalizing arbitrary
    lengths when the stress questions require more detail.
    """
    return recall_points(answer, expected)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Feed each conversation, then probe recall in separate fresh threads.

    Include training and recall turns in usage totals. Counter deltas avoid
    double-counting cumulative usage. Memory growth counts each user once.
    `config` remains part of the shared scaffold API; the agent owns its config.
    """
    users = {conversation["user_id"] for conversation in conversations}
    memory_size = getattr(agent, "memory_file_size", lambda user_id: 0)
    memory_before = sum(memory_size(user_id) for user_id in users)
    tokens = prompt_tokens = compactions = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    run_id = uuid4().hex

    def reply(user_id: str, thread_id: str, message: str) -> str:
        nonlocal tokens, prompt_tokens, compactions
        before = (
            agent.token_usage(thread_id), agent.prompt_token_usage(thread_id),
            agent.compaction_count(thread_id),
        )
        result = agent.reply(user_id, thread_id, message)
        if not isinstance(result, dict) or not isinstance(result.get("reply"), str):
            raise TypeError(f"{agent_name}.reply() must return a dict with a string 'reply'")
        tokens += agent.token_usage(thread_id) - before[0]
        prompt_tokens += agent.prompt_token_usage(thread_id) - before[1]
        compactions += agent.compaction_count(thread_id) - before[2]
        return result["reply"]

    for index, conversation in enumerate(conversations):
        user_id = conversation["user_id"]
        thread_id = f"benchmark-{run_id}-{index}-conversation"
        for message in conversation["turns"]:
            reply(user_id, thread_id, message)
        # Probe immediately: corrections in later conversations must not affect
        # expectations for the current conversation. Each probe starts afresh.
        for question_index, question in enumerate(conversation.get("recall_questions", [])):
            answer = reply(
                user_id, f"benchmark-{run_id}-{index}-recall-{question_index}",
                question["question"],
            )
            expected = question["expected_contains"]
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))
    return BenchmarkRow(
        agent_name=agent_name, agent_tokens_only=tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
        response_quality=sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
        memory_growth_bytes=sum(memory_size(user_id) for user_id in users) - memory_before,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Render all six required metrics without an optional table dependency."""
    headers = [
        "Agent", "Agent tokens only", "Prompt tokens processed",
        "Cross-session recall", "Response quality", "Memory growth (bytes)", "Compactions",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = [
            row.agent_name.replace("|", "\\|").replace("\n", " "),
            str(row.agent_tokens_only), str(row.prompt_tokens_processed),
            f"{row.recall_score:.1%}", f"{row.response_quality:.1%}",
            str(row.memory_growth_bytes), str(row.compactions),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    """Run two reproducible offline suites with fresh agents and isolated memory."""
    config = load_config(Path(__file__).resolve().parent.parent)
    suites = [
        ("Standard Benchmark", load_conversations(config.data_dir / "conversations.json")),
        ("Long-Context Stress Benchmark", load_conversations(config.data_dir / "advanced_long_context.json")),
    ]
    config.state_dir.mkdir(parents=True, exist_ok=True)
    for title, conversations in suites:
        # Keep existing user profiles intact; reruns start with empty memory.
        with TemporaryDirectory(prefix="benchmark-", dir=config.state_dir) as state_dir:
            suite_config = replace(config, state_dir=Path(state_dir))
            agents = [
                ("Baseline", BaselineAgent(suite_config, force_offline=True)),
                ("Advanced", AdvancedAgent(suite_config, force_offline=True)),
            ]
            rows = [run_agent_benchmark(name, agent, conversations, suite_config) for name, agent in agents]
            print(f"\n## {title}\n")
            print(format_rows(rows))
    print("\nOffline estimates; token counts follow each agent's counters.")
    print("Response quality is a factual-coverage heuristic, not an independent judge.")


if __name__ == "__main__":
    main()
