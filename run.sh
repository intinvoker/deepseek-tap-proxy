#!/usr/bin/env bash
# DeepSeek Tap Proxy — 一键启动 (bash)
set -e

PORT="${1:-8080}"
TARGET="${2:-https://api.deepseek.com}"
LOCK_MODE="${3:-msg0}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== DeepSeek Tap Proxy ==="
echo "  Port:   $PORT"
echo "  Target: $TARGET"
echo "  Lock:   $LOCK_MODE"
echo ""

python -m deepseek_tap_proxy --port "$PORT" --target "$TARGET" --lock-mode "$LOCK_MODE"
