from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    """Create isolated offline settings with a low compaction threshold."""
    config = LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        compact_threshold_tokens=80,
        compact_keep_messages=2,
        model=ProviderConfig(provider="openai", model_name="stub", temperature=0.0),
        judge_model=ProviderConfig(provider="openai", model_name="stub", temperature=0.0),
    )
    config.state_dir.mkdir(parents=True, exist_ok=True)
    return config


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify Vietnamese profile storage, edits and persistence on disk."""
    store = UserProfileStore(make_config(tmp_path).state_dir / "profiles")
    assert store.read_text("dungct") == ""
    assert store.file_size("dungct") == 0
    original = "# User\n- Tên: DũngCT\n- Nơi ở: Đà Nẵng\n"
    path = store.write_text("dungct", original)
    assert path == store.path_for("dungct")
    assert path.read_text(encoding="utf-8") == original
    assert store.read_text("dungct") == original
    assert store.file_size("dungct") == len(original.encode("utf-8"))
    store.write_text("other", "# User\n- Tên: Lan\n")
    assert store.edit_text("dungct", "Đà Nẵng", "Huế") is True
    updated = original.replace("Đà Nẵng", "Huế")
    assert store.read_text("dungct") == updated
    assert store.edit_text("dungct", "không tồn tại", "Hà Nội") is False
    assert store.read_text("dungct") == updated
    assert "Lan" in store.read_text("other")
    assert UserProfileStore(store.root_dir).read_text("dungct") == updated


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long history is compacted while recent messages stay intact."""
    config = make_config(tmp_path)
    memory = CompactMemoryManager(config.compact_threshold_tokens, config.compact_keep_messages)
    memory.append("short", "user", "Xin chào.")
    assert memory.compaction_count("short") == 0
    messages = [
        {"role": "user", "content": f"Đoạn {index}: " + "Thông tin kỹ thuật về hệ thống AI. " * 20}
        for index in range(10)
    ]
    for message in messages:
        memory.append("long", message["role"], message["content"])
    context = memory.context("long")
    assert memory.compaction_count("long") > 0, "Long history must trigger compaction"
    assert context["summary"].strip()
    assert context["messages"] == messages[-config.compact_keep_messages:]
    full_tokens = sum(estimate_tokens(message["content"]) for message in messages)
    compact_tokens = estimate_tokens(context["summary"]) + sum(
        estimate_tokens(message["content"]) for message in context["messages"]
    )
    assert compact_tokens < full_tokens
    assert memory.context("short")["messages"] == [{"role": "user", "content": "Xin chào."}]


def test_cross_session_recall(tmp_path: Path) -> None:
    """Advanced must remember after restart; baseline must forget new threads."""
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    for message in ["Mình tên là DũngCT.", "Đồ uống yêu thích là cà phê sữa đá."]:
        baseline.reply("dungct", "learn", message)
        advanced.reply("dungct", "learn", message)
    question = "Mình tên gì và đồ uống yêu thích là gì?"
    baseline_answer = baseline.reply("dungct", "recall", question)["reply"].casefold()
    restarted = AdvancedAgent(config, force_offline=True)
    advanced_answer = restarted.reply("dungct", "recall", question)["reply"].casefold()
    for fact in ("DũngCT", "cà phê sữa đá"):
        assert fact.casefold() not in baseline_answer
        assert fact.casefold() in advanced_answer
    assert restarted.memory_file_size("dungct") > 0
    stranger_answer = restarted.reply("other", "other-recall", question)["reply"].casefold()
    assert "dũngct" not in stranger_answer
    assert "cà phê sữa đá" not in stranger_answer


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare cumulative prompt cost on identical long offline conversations."""
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    previous_baseline = previous_advanced = 0
    for index in range(24):
        message = f"Đoạn kỹ thuật {index}: " + (
            "Hệ thống cần đo độ trễ, thông lượng và chi phí xử lý ngữ cảnh. " * 30
        )
        baseline_result = baseline.reply("reader", "long", message)
        advanced_result = advanced.reply("reader", "long", message)
        assert isinstance(baseline_result["reply"], str)
        assert isinstance(advanced_result["reply"], str)
        current_baseline = baseline.prompt_token_usage("long")
        current_advanced = advanced.prompt_token_usage("long")
        assert current_baseline > previous_baseline
        assert current_advanced > previous_advanced
        previous_baseline, previous_advanced = current_baseline, current_advanced
    assert baseline.compaction_count("long") == 0
    assert advanced.compaction_count("long") > 0
    assert 0 < advanced.prompt_token_usage("long") < baseline.prompt_token_usage("long")
