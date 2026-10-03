from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Student TODO: implement Agent B / Advanced Agent.

    Required memory layers:
    1. within-session memory
    2. persistent `User.md`
    3. compact memory for long threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        # TODO: optionally initialize a real LangChain/LangGraph agent.
        self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: route between offline mode and live mode."""

        if self.force_offline:
            return self._reply_offline(user_id, thread_id, message)
        else:
            if self.langchain_agent is None:
                self._maybe_build_langchain_agent()
            if self.langchain_agent is not None:
                state = self.langchain_agent.run(user_id=user_id, thread_id=thread_id, message=message)
                return state

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: implement the deterministic advanced path.

        Pseudocode:
        1. Extract stable profile facts from the incoming message.
        2. Persist those facts into `User.md`.
        3. Append the message into compact memory.
        4. Estimate prompt-context load from `User.md` + summary + recent messages.
        5. Generate a response that can answer long-term recall questions.
        6. Append the assistant reply and update token counters.
        """

        profile_updates = extract_profile_updates(message)
        if profile_updates:
            # Merge updates instead of erasing memory on turns with no new facts.
            facts = {}
            for line in self.profile_store.read_text(user_id).splitlines():
                if line.startswith("- ") and ": " in line:
                    key, value = line[2:].split(": ", 1)
                    facts[key] = value
            facts.update(profile_updates)
            self.profile_store.write_text(
                user_id, "\n".join(f"- {key}: {value}" for key, value in facts.items())
            )
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        reply_content = self._offline_response(user_id, thread_id, message)
        self.compact_memory.append(thread_id, "assistant", reply_content)
        self.thread_prompt_tokens[thread_id] = self.prompt_token_usage(thread_id) + prompt_tokens
        self.thread_tokens[thread_id] = (
            self.token_usage(thread_id) + estimate_tokens(message) + estimate_tokens(reply_content)
        )
        return {
            "reply": reply_content,
            "token_usage": self.token_usage(thread_id),
            "prompt_tokens_processed": self.prompt_token_usage(thread_id),
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Read profile, summary and all retained messages without changing state."""
        context = self.compact_memory.context(thread_id)
        return (
            estimate_tokens(self.profile_store.read_text(user_id))
            + estimate_tokens(context["summary"])
            + sum(estimate_tokens(msg["content"]) for msg in context["messages"])
        )

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Produce a bounded deterministic reply from persistent profile facts."""
        profile = self.profile_store.read_text(user_id)
        if "?" in message or any(cue in message.casefold() for cue in ("nhắc lại", "tên gì", "nghề gì", "ở đâu")):
            return profile or "Mình chưa có thông tin hồ sơ của bạn."
        return "Mình đã nhận thông tin của bạn."

    def _maybe_build_langchain_agent(self):
        """Student TODO: wire a live agent with tools and compact middleware.

        High-level design:
        - `build_chat_model(self.config.model)` for the selected provider
        - `InMemorySaver` for short-term thread state
        - tool to read `User.md`/
        - tool to write/edit `User.md`
        - dynamic prompt that injects profile memory
        - summarization middleware for long threads
        """

        build_chat_model(self.config.model)
