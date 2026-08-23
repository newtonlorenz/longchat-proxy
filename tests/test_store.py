from __future__ import annotations

from longchat_proxy.messages import canonical_hash
from longchat_proxy.store import MemoryStore, SQLiteStore, SummaryRecord


def _turns() -> list[dict]:
    return [
        {"role": "user", "content": "Remember ORANGE-17"},
        {"role": "assistant", "content": "Remembered"},
        {"role": "user", "content": "Continue"},
    ]


def test_memory_store_matches_an_exact_prefix() -> None:
    turns = _turns()
    store = MemoryStore()
    store.put(SummaryRecord(canonical_hash(turns, 2), 2, "Code is ORANGE-17"))
    assert store.find_best(turns).summary == "Code is ORANGE-17"
    edited = [dict(item) for item in turns]
    edited[0]["content"] = "Remember BLUE-19"
    assert store.find_best(edited) is None


def test_sqlite_store_round_trip(tmp_path) -> None:
    turns = _turns()
    store = SQLiteStore(tmp_path / "state.sqlite3")
    store.put(
        SummaryRecord(
            canonical_hash(turns, 2),
            2,
            "Code is ORANGE-17",
            "conversation-a",
        )
    )
    found = store.find_best(turns, "conversation-a")
    assert found is not None
    assert found.summary == "Code is ORANGE-17"
    assert store.count() == 1
    store.close()
