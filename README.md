# DeepSeek Tap Proxy

多格式 AI API 透明代理。支持 OpenAI 和 Anthropic 协议，具备前缀缓存锁定能力。

```bash
pip install -e .
```

## 启动

```bash
python -m deepseek_tap_proxy --port 11808 --target https://api.deepseek.com
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | 11808 | 代理端口 |
| `--target` | `https://api.deepseek.com` | 上游地址 |
| `--lock-mode` | prefix | prefix / none |
| `--trace-dir` | ./data/traces | trace 目录 |
| `--db-path` | ./data/proxy.db | SQLite 路径 |
| `--timeout` | 600 | 超时秒数 |

## 使用

```bash
curl http://localhost:11808/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_KEY" \
  -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"hi"}]}'
```

## 功能

- **SysToolsLock** — 同 session 锁定 system + tools 基线
- **PrefixLock** — 三段 hash 前缀匹配，保障 KV cache 命中
- **Trace** — 每请求 JSONL 完整记录
- **OTLP** — 结构化日志输出
- **缓存分析** — 逐请求 cache hit 统计
- **多上游路由** — API Key 前缀路由到多个上游
- **Copilot 压缩优化** — 自动识别前台压缩请求，转换为后台格式

## 灵感来源

- [claude-tap](https://github.com/liaohch3/claude-tap)
- [deepseek-v4-for-copilot](https://github.com/Vizards/deepseek-v4-for-copilot)

## 理论依据

- [上下文硬盘缓存](https://api-docs.deepseek.com/zh-cn/guides/kv_cache)
- [Anthropic API](https://api-docs.deepseek.com/zh-cn/guides/anthropic_api)

## 核心目标

- 同对话session内, 100%前缀缓存命中, 即使牺牲提示词灵活性
- 跨对话session内, 100%系统提示词缓存命中
