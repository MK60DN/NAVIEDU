from fastapi import Request
from fastapi.responses import JSONResponse  # 添加导入
from starlette.middleware.base import BaseHTTPMiddleware
import time
import logging

# ... 其余代码保持不变

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """请求日志中间件"""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # 记录请求信息
        logger.info(f"Request: {request.method} {request.url.path}")

        response = await call_next(request)

        # 计算处理时间
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)

        # 记录响应信息
        logger.info(f"Response: {response.status_code} - Time: {process_time:.3f}s")

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """简单的速率限制中间件"""

    def __init__(self, app, calls: int = 100, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.clients = {}

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        current_time = time.time()

        # 清理过期记录
        self.clients = {
            ip: times for ip, times in self.clients.items()
            if any(t > current_time - self.period for t in times)
        }

        # 检查速率限制
        if client_ip in self.clients:
            recent_calls = [
                t for t in self.clients[client_ip]
                if t > current_time - self.period
            ]

            if len(recent_calls) >= self.calls:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Too many requests"}
                )

            self.clients[client_ip] = recent_calls + [current_time]
        else:
            self.clients[client_ip] = [current_time]

        response = await call_next(request)
        return response