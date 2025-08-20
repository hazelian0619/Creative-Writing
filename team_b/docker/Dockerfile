# Team B - 认知状态后端服务
# 工业级FastAPI + MongoDB + Prometheus监控

FROM python:3.9-slim

LABEL maintainer="Team B - Backend Engineering"
LABEL version="1.0.0"
LABEL description="Cognitive State Tracking Backend Service"

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

# 创建非root用户
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

# 复制依赖文件
COPY pyproject.toml .
COPY requirements.txt* ./

# 安装Python依赖
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev

# 复制应用代码
COPY . .

# 确保Team A的依赖可用
RUN mkdir -p /app/team_a && \
    cp -r team_a/* /app/team_a/ 2>/dev/null || true

# 设置正确的权限
RUN chown -R appuser:appuser /app
USER appuser

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 暴露端口
EXPOSE 8000 9090

# 启动命令
CMD ["python", "-m", "uvicorn", "team_b.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]