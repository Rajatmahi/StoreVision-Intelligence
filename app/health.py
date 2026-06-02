"""Health check logic — GET /health

Service status, last event timestamp per store.
STALE_FEED warning if > 10 min lag.
Must be accurate - this is what an on-call engineer checks first.
"""

from fastapi import APIRouter
from datetime import datetime, timezone
from app.database import get_db_connection

router = APIRouter()

@router.get("/health")
def health_check():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get the latest timestamp for each store
        cursor.execute("SELECT store_id, MAX(timestamp) as last_event FROM events GROUP BY store_id")
        rows = cursor.fetchall()
        
        status = "healthy"
        store_status = {}
        
        now = datetime.now(timezone.utc)
        
        for row in rows:
            store_id = row["store_id"]
            last_event_str = row["last_event"]
            
            # Parse timestamp (assumes ISO format with Z or +00:00)
            try:
                # Basic parsing, handling Python 3.10+ fromisoformat
                last_event = datetime.fromisoformat(last_event_str.replace('Z', '+00:00'))
                lag_minutes = (now - last_event).total_seconds() / 60
                
                is_stale = lag_minutes > 10
                if is_stale:
                    status = "degraded"
                    
                store_status[store_id] = {
                    "last_event": last_event_str,
                    "status": "STALE_FEED" if is_stale else "OK"
                }
            except Exception:
                store_status[store_id] = {
                    "last_event": last_event_str,
                    "status": "UNKNOWN"
                }
                
        conn.close()
        
        return {
            "status": status,
            "stores": store_status
        }
    except Exception as e:
        # Graceful degradation if DB is down
        return {"status": "unhealthy", "error": "Database connection failed"}
