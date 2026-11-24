# DeepSeek Tap Proxy — 容器镜像
# 用法：docker build -t deepseek-tap-proxy:latest .
# 启动：python -m deepseek_tap_proxy（config 全走 CLI/env，见 pyproject [project.scripts]）

FROM python:3.12-slim

# N3 长期服务内存：系统分配器（pymalloc arenas 不归还 OS——水位只升不降）；
# 配合 store.py 周期 gc.collect + malloc_trim(0) 回落 RSS（业界长期运行标准）
ENV PYTHONMALLOC=malloc

WORKDIR /app

# 依赖分层缓存：先装构建元数据，再装源码
COPY pyproject.toml README.md ./
COPY src ./src

# postgres extra：psycopg[binary]（PG 后端必需，容器内不走虚拟环境）
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[postgres]"

EXPOSE 11808

# 数据目录：事件日志/DB/OTLP 均落在 /app/data（部署时挂载卷）
VOLUME ["/app/data"]

CMD ["python", "-m", "deepseek_tap_proxy", \
     "--host", "0.0.0.0", \
     "--port", "11808", \
     "--lock-mode", "prefix", \
     "--db-path", "/app/data/proxy.db", \
     "--trace-dir", "/app/data/runs", \
     "--events-path", "/app/data/chain-events.jsonl", \
     "--otlp-log-dir", "/app/data/otlp-logs"]
