from deepseek_tap_proxy.config import ProxyConfig


def test_defaults():
    cfg = ProxyConfig()
    assert cfg.trace_flush_interval == 2.0
    assert cfg.trace_batch_max == 500
    assert cfg.trace_file_max_bytes == 500 * 1024 * 1024
    assert cfg.db_commit_interval == 1.0
    assert cfg.db_batch_max == 500
    assert cfg.db_queue_max == 1000


def test_from_args():
    cfg = ProxyConfig.from_args([
        "--trace-flush-interval", "5", "--trace-batch-max", "100",
        "--trace-file-max-bytes", str(1024 * 1024),
        "--db-commit-interval", "2", "--db-batch-max", "50",
        "--db-queue-max", "200",
    ])
    assert cfg.trace_flush_interval == 5.0
    assert cfg.trace_batch_max == 100
    assert cfg.trace_file_max_bytes == 1024 * 1024
    assert cfg.db_commit_interval == 2.0
    assert cfg.db_batch_max == 50
    assert cfg.db_queue_max == 200
