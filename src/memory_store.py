from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
import re
import unicodedata


def estimate_tokens(text: str) -> int:
    """Stable character-based estimate, not a provider tokenizer."""
    text = (text or "").strip()
    return (len(text) + 3) // 4


@dataclass
class UserProfileStore:
    """UTF-8 markdown profiles with safe filenames and fact merging."""
    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be nonempty")
        slug = re.sub(r"[^\w-]+", "_", user_id.strip().lower()).strip("_")[:80] or "user"
        # Preserve ordinary IDs; hash transformed IDs to prevent collisions.
        if slug != user_id or slug.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
            slug += "_" + sha256(user_id.encode()).hexdigest()[:12]
        return self.root_dir / f"{slug}.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        if not search_text:
            return False
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        return dict(re.findall(r"^- ([\w]+): (.+)$", self.read_text(user_id), re.MULTILINE))

    def update_profile(self, user_id: str, updates: dict[str, str]) -> None:
        if not updates:
            return
        facts = self.facts(user_id)
        facts.update({key: value for key, value in updates.items() if value.strip()})
        self.write_text(user_id, "# User\n" + "\n".join(f"- {key}: {value}" for key, value in sorted(facts.items())) + "\n")

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        self.update_profile(user_id, {key: value})

    def load_profile(self, user_id: str) -> str:
        return self.read_text(user_id)


def extract_profile_updates(message: str) -> dict[str, str]:
    """Conservative offline rules for explicit Vietnamese/English declarations.

    Only declarative clauses are processed; later declarations override earlier
    ones. This heuristic is limited to common lab profile fields.
    """
    facts: dict[str, str] = {}
    message = unicodedata.normalize("NFC", message)
    patterns = {
        "name": r"(?:mình tên(?: là)?|tên mình là|my name is)\s+([^.,;!?]+)",
        "location": r"(?:mình (?:vẫn |đang |hiện )?(?:ở|sống ở)|hiện ở|nơi ở hiện tại là|i live in)\s+([^.,;!?]+)",
        "profession": r"(?:mình (?:đang )?làm|và đang làm|giờ chuyển sang|nghề nghiệp hiện tại (?:vẫn )?là|nghề nghiệp thì vẫn là|i am a)\s+([^.,;!?]+)",
        "favorite_food": r"(?:món ăn yêu thích(?: của mình)? là|my favorite food is)\s+([^.,;!?]+)",
        "favorite_drink": r"(?:đồ uống yêu thích(?: của mình)? là|my favorite drink is)\s+([^.,;!?]+)",
        "pet": r"mình nuôi\s+([^.,;!?]+)",
        "interests": r"mình (?:thích|đang quan tâm nhiều đến)\s+([^.;!?]+)",
    }
    for sentence in re.split(r"(?<=[.!?;])\s+|\n+", message):
        if "?" in sentence:
            continue
        for key, pattern in patterns.items():
            for match in re.finditer(pattern, sentence, re.IGNORECASE):
                value = re.split(
                    r"\s+(?:và đang|và mình|cho |chứ |dù |nhưng |trong |mỗi ngày|không đổi|nữa\b)",
                    match.group(1), maxsplit=1, flags=re.IGNORECASE,
                )[0].strip(" .,;!")
                if key == "interests" and re.match(r"(?:cách|kiểu|câu trả lời|giải thích)\b", value, re.IGNORECASE):
                    continue
                if value:
                    facts[key] = value
        lower = sentence.casefold()
        if any(cue in lower for cue in ("mình muốn", "mình thích", "hãy trả lời", "style trả lời")):
            style = []
            if "ngắn" in lower or "gọn" in lower:
                style.append("ngắn gọn")
            if "3 bullet" in lower:
                style.append("3 bullet")
            elif "bullet" in lower:
                style.append("bullet")
            if "ví dụ thực chiến" in lower:
                style.append("ví dụ thực chiến")
            elif "ví dụ thực tế" in lower:
                style.append("ví dụ thực tế")
            if "trade-off" in lower:
                style.append("trade-off")
            if style:
                facts["response_style"] = ", ".join(style)
    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Bounded extractive summary: profile facts plus recent user snippets.

    Snippets preserve a little topic context; they cannot retain every detail.
    Assistant repetitions are omitted to avoid recursive summary growth.
    """
    if max_items <= 0:
        return ""
    facts: dict[str, str] = {}
    snippets = []
    for message in messages:
        if message["role"] != "user":
            continue
        facts.update(extract_profile_updates(message["content"]))
        snippet = " ".join(message["content"].split())[:160]
        if snippet and snippet not in snippets:
            snippets.append(snippet)
    lines = [f"{key}: {value[:160]}" for key, value in sorted(facts.items())]
    lines.extend(snippets[-max_items:])
    return "\n".join(lines)[:1200]


@dataclass
class CompactMemoryManager:
    """Per-thread bounded summary plus recent full messages."""
    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.threshold_tokens <= 0 or self.keep_messages < 0:
            raise ValueError("threshold_tokens must be positive; keep_messages must be nonnegative")

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})
        thread["messages"].append({"role": role, "content": content})
        messages = thread["messages"]
        tokens = estimate_tokens(thread["summary"]) + sum(estimate_tokens(msg["content"]) for msg in messages)
        if tokens > self.threshold_tokens and len(messages) > self.keep_messages:
            split = len(messages) - self.keep_messages
            new_summary = summarize_messages(messages[:split])
            # Bound the merged summary; a huge recent message can still exceed
            # the threshold because recent messages are deliberately kept whole.
            budget = max(32, min(1200, self.threshold_tokens * 2))
            combined = "\n".join(filter(None, [thread["summary"], new_summary]))
            thread["summary"] = combined[-budget:]
            thread["messages"] = messages[split:]
            thread["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, object]:
        thread = self.state.get(thread_id, {"messages": [], "summary": "", "compactions": 0})
        return {"messages": [dict(msg) for msg in thread["messages"]], "summary": thread["summary"], "compactions": thread["compactions"]}

    def compaction_count(self, thread_id: str) -> int:
        return self.state.get(thread_id, {}).get("compactions", 0)

    def get_summary(self, thread_id: str) -> str:
        return self.context(thread_id)["summary"]

    def get_recent_messages(self, thread_id: str) -> list[dict[str, str]]:
        return self.context(thread_id)["messages"]

    def append_message(self, thread_id: str, content: str) -> None:
        self.append(thread_id, "user", content)
