import logging
import time
import uuid

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.database import init_db
from app import ingestion, metrics, funnel, anomalies, health

app = FastAPI(title="Store Intelligence System")

# Set up structured logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("store_intelligence")

class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        trace_id = str(uuid.uuid4())
        start_time = time.time()
        
        store_id = None
        path_parts = request.url.path.split('/')
        if "stores" in path_parts:
            try:
                store_id = path_parts[path_parts.index("stores") + 1]
            except IndexError:
                pass

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            status_code = 500
            raise e
        finally:
            latency_ms = (time.time() - start_time) * 1000
            logger.info(
                f"trace_id={trace_id} "
                f"method={request.method} "
                f"endpoint={request.url.path} "
                f"store_id={store_id} "
                f"status_code={status_code} "
                f"latency_ms={latency_ms:.2f}"
            )
            
        return response

app.add_middleware(StructuredLoggingMiddleware)

# Initialize the database and table when the file is loaded
init_db()

@app.get("/")
def read_root():
    return {"message": "Store Intelligence API running"}

# Register routers from our modular endpoints
app.include_router(ingestion.router)
app.include_router(metrics.router)
app.include_router(funnel.router)
app.include_router(anomalies.router)
app.include_router(health.router)
