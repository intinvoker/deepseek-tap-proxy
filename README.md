# DeepSeek Tap Proxy

多格式 AI API 透明代理。支持 OpenAI 和 Anthropic 协议，具备 system prompt 锁定能力。

依赖：Python 3.12+，aiohttp。

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
| `--host` | 0.0.0.0 | 绑定地址 |
| `--target` | `https://api.deepseek.com` | 上游地址 |
| `--lock-mode` | msg0 | msg0 / prefix / none |
| `--trace-dir` | ./data/traces | trace 目录 |
| `--db-path` | ./data/proxy.db | SQLite 路径 |
| `--timeout` | 600 | 超时秒数 |

## 架构

```
shared/ (SSE/Trace/Headers/Decompress) → pipeline (Context/Stage/Pipeline) → openai/ anthropic/
```

三层分离，格式模块零耦合。新增 API 格式：新建目录 → 实现 Stage + handler + routes。

## API 支持

| 格式 | 路由 | Session 类型 |
|------|------|-------------|
| OpenAI | `POST /chat/completions` | Copilot / Hermes / Generic |
| Anthropic | `ANY /anthropic/{path}` | Claude Code |

## 使用

```bash
curl http://localhost:11808/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_KEY" \
  -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"hi"}]}'
```

## 功能

- **Msg0Lock** — system prompt 被篡改时自动锁定为首次值
- **PrefixLock** — 会话前缀被破坏时用历史版本覆盖
- **Session 指纹** — 按消息内容识别 Copilot/Hermes/Generic
- **Trace** — 每请求 JSONL 完整记录
- **缓存分析** — 逐请求 cache hit 统计

## 灵感来源

- [claude-tap](https://github.com/liaohch3/claude-tap)
- [deepseek-v4-for-copilot](https://github.com/Vizards/deepseek-v4-for-copilot)

## 理论依据

- [上下文硬盘缓存](https://api-docs.deepseek.com/zh-cn/guides/kv_cache)
- [Anthropic API](https://api-docs.deepseek.com/zh-cn/guides/anthropic_api)

## 核心目标

- 同对话session内, 100%前缀缓存命中, 即使牺牲提示词灵活性
- 跨对话session内, 100%系统提示词缓存命中
