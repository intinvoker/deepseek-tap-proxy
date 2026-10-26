"""契约测试：OTLP chat attrs ⊇ PersistRecord 全部可序列化字段（除原始 body/headers）。

对应 docs/superpowers/specs/2026-07-31-proxy-persist-field-lineage-design.md §5。
"""
from deepseek_tap_proxy.shared.log import _chat_otlp_attrs, _build_otlp_record
from deepseek_tap_proxy.shared.persist import PersistRecord

# 血缘基线 §5 缺口（必须补齐）
REQUIRED = {
    "total_input_tokens", "output_tokens", "total_tokens",
    "cache_creation_tokens", "cache_hit_ratio",
    "has_reasoning", "has_tool_calls",
    "stream_mode", "prompt_messages_count", "prompt_tools_count",
    "session_uuid", "prefix_chain_id", "msg_hash_chain",
    "pref_sys_hit", "pref_tools_hit", "pref_msgs_hit",
    "is_modified", "modification_note", "system_role_converted",
    "trace_category",
}


class _C:
    client_type = "copilot"
    chat_type = "normal"
    session_method = "match"
    meta = {"upstream_name": "default", "sys_tools_locked": "lock",
            "pref_hit_count": 7,
            "pref_sys_hit": 1, "pref_tools_hit": 2, "pref_msgs_hit": 3}
    persist = PersistRecord(lock_type="prefix", cache_hit_tokens=8, cache_miss_tokens=2)
    chain_action = None
    format = "openai"
    req_id = "req_1"
    status = 200
    duration_ms = 10
    request = type("R", (), {"path": "/v1/messages"})()
    resp_model = "m"
    req_model = "m"
    session_id = "s1"
    topic = "chat"
    body_readonly = {}


def test_chat_otlp_superset():
    attrs = _chat_otlp_attrs(_C())
    keys = {a["key"] for a in attrs}
    missing = REQUIRED - keys
    assert not missing, f"OTLP 缺字段: {missing}"


def test_pref_hit_count_preserved():
    """viewer render-otlp.js 依赖 pref_hit_count 渲染 Pref Hit 卡片 — 必须输出该字段。"""
    attrs = _chat_otlp_attrs(_C())
    keys = {a["key"] for a in attrs}
    assert "pref_hit_count" in keys, "OTLP 缺 pref_hit_count（viewer Pref Hit 回归）"
    vals = {a["key"]: a["value"].get("intValue") for a in attrs}
    assert vals["pref_hit_count"] == 7, "pref_hit_count 值应来自 meta.pref_hit_count"


def test_build_record_has_kind_field():
    rec = _build_otlp_record(_C(), "run1")
    attrs = {a["key"] for a in rec["attributes"]}
    assert "kind" in attrs


def test_log_subset_of_otlp():
    """log 展示概念 ⊆ OTLP 超集（血缘基线 §6-3）—— log 不新增 OTLP 没有的字段。"""
    from deepseek_tap_proxy.shared.log import _chat_otlp_attrs

    attrs = _chat_otlp_attrs(_C())
    keys = {a["key"] for a in attrs}
    LOG_FIELD_MAP = {
        "lock": "lock_type",
        "cache_hit": "cache_hit_tokens",
        "cache_miss": "cache_miss_tokens",
        "cache_ratio": "cache_hit_ratio",
        "pref_sys_hit": "pref_sys_hit",
        "pref_tools_hit": "pref_tools_hit",
        "pref_msgs_hit": "pref_msgs_hit",
        "sid": "session_id",
        "upstream": "upstream_name",
        "client": "client_type",
        "intent": "chat_type",
        "slock": "sys_tools_locked",
    }
    missing = {v for v in LOG_FIELD_MAP.values() if v not in keys}
    assert not missing, f"log 依赖的字段不在 OTLP 超集: {missing}"
