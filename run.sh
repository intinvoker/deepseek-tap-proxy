#!/usr/bin/env bash
# DeepSeek Tap Proxy — 一键启动 (bash)
set -e

PORT="${1:-11808}"
TARGET="${2:-https://api.deepseek.com}"
LOCK_MODE="${3:-prefix}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "  DeepSeek Tap Proxy"
echo "  Port:   $PORT"
echo "  Target: $TARGET"
echo "  Lock:   $LOCK_MODE (msg0 / prefix / none)"
echo ""

python -m deepseek_tap_proxy --port "$PORT" --target "$TARGET" --lock-mode "$LOCK_MODE"
