"""Tests for shared.log._coerce_keys — orjson key compatibility."""

from deepseek_tap_proxy.shared.log import _coerce_keys


class TestCoerceKeys:
    def test_flat_int_keys(self):
        assert _coerce_keys({1: "a", 2: "b"}) == {"1": "a", "2": "b"}

    def test_nested_int_keys(self):
        assert _coerce_keys({1: {2: "c"}}) == {"1": {"2": "c"}}

    def test_mixed_keys(self):
        assert _coerce_keys({1: "a", "b": {2: "c"}}) == {"1": "a", "b": {"2": "c"}}

    def test_list_of_dicts(self):
        assert _coerce_keys([{1: "x"}, {2: "y"}]) == [{"1": "x"}, {"2": "y"}]

    def test_deeply_nested(self):
        assert _coerce_keys({1: [{2: {3: "deep"}}]}) == {"1": [{"2": {"3": "deep"}}]}

    def test_non_dict_passthrough(self):
        assert _coerce_keys("hello") == "hello"
        assert _coerce_keys(42) == 42
        assert _coerce_keys(None) is None

    def test_empty(self):
        assert _coerce_keys({}) == {}
        assert _coerce_keys([]) == []

    def test_str_keys_unchanged(self):
        assert _coerce_keys({"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_orjson_compatible_output(self):
        """Verify the output round-trips through orjson without error."""
        import orjson

        record = {
            "request_id": "req_123",
            "request": {"body": {1: "should become str key", "nested": {2: "also nested"}}},
            "response": {"body": {"choices": [{"index": 0, "message": {"role": "assistant"}}]}},
        }
        coerced = _coerce_keys(record)
        # Should not raise TypeError: Dict key must be str
        encoded = orjson.dumps(coerced).decode()
        assert "should become str key" in encoded
