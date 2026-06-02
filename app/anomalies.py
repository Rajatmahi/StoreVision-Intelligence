"""Anomaly detection logic — GET /stores/{id}/anomalies

Detects operational anomalies such as:
- QUEUE_SPIKE: More than 5 people in the billing queue.
- QUEUE_BUILDUP: More than 2 people in the billing queue.
- DEAD_ZONE: No visits to a specific zone in the last 30 minutes.
- CONVERSION_DROP: Unusually low conversion rate (< 10%) with > 10 entries.
"""

from fastapi import APIRouter
from app.database import get_db_connection

router = APIRouter()

@router.get("/stores/{store_id}/anomalies")
def get_anomalies(store_id: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        base_cond = "store_id = ? AND is_staff = 0"
        anomalies = []
        
        # Check queue depth
        cursor.execute(f"""
            SELECT COUNT(DISTINCT visitor_id) FROM events 
            WHERE {base_cond} AND event_type = 'BILLING_QUEUE_JOIN'
            AND visitor_id NOT IN (
                SELECT visitor_id FROM events WHERE {base_cond} AND event_type IN ('BILLING_QUEUE_ABANDON', 'PURCHASE', 'EXIT')
            )
        """, (store_id, store_id))
        queue_depth = cursor.fetchone()[0] or 0
        
        if queue_depth > 5:
            anomalies.append({
                "severity": "CRITICAL",
                "type": "QUEUE_SPIKE",
                "message": f"High billing queue depth: {queue_depth} people",
                "suggested_action": "Open additional checkout counters immediately."
            })
        elif queue_depth > 2:
            anomalies.append({
                "severity": "WARN",
                "type": "QUEUE_BUILDUP",
                "message": f"Billing queue building up: {queue_depth} people",
                "suggested_action": "Prepare to open another counter."
            })
            
        # Check dead zones
        cursor.execute(f"""
            SELECT DISTINCT zone_id FROM events WHERE {base_cond} AND zone_id IS NOT NULL
        """, (store_id,))
        all_zones = [r[0] for r in cursor.fetchall()]
        
        for zone in all_zones:
            cursor.execute(f"""
                SELECT COUNT(*) FROM events 
                WHERE {base_cond} AND zone_id = ? AND timestamp >= datetime('now', '-30 minute')
            """, (store_id, zone))
            recent_visits = cursor.fetchone()[0] or 0
            
            if recent_visits == 0:
                anomalies.append({
                    "severity": "INFO",
                    "type": "DEAD_ZONE",
                    "message": f"Zone {zone} has had no visits in the last 30 minutes.",
                    "suggested_action": "Check for physical blockage or review merchandising."
                })
                
            cursor.execute(f"""
                SELECT 
                    SUM(CASE WHEN event_type = 'ZONE_ENTER' THEN 1 ELSE 0 END) -
                    SUM(CASE WHEN event_type = 'ZONE_EXIT' THEN 1 ELSE 0 END)
                FROM events
                WHERE {base_cond} AND zone_id = ? AND timestamp >= datetime('now', '-1 hour')
            """, (store_id, zone))
            active_occupancy = cursor.fetchone()[0] or 0
            
            if active_occupancy > 10:
                anomalies.append({
                    "severity": "CRITICAL",
                    "type": "OVERCROWDING",
                    "message": f"Zone {zone} is overcrowded with ~{active_occupancy} active visitors.",
                    "suggested_action": "Deploy staff to assist customers and manage flow."
                })

        # Check long dwell time
        cursor.execute(f"""
            SELECT MAX(dwell_ms) FROM events 
            WHERE {base_cond} AND event_type IN ('ZONE_EXIT', 'ZONE_DWELL') 
            AND timestamp >= datetime('now', '-1 hour')
        """, (store_id,))
        max_dwell = cursor.fetchone()[0] or 0
        if max_dwell > 900000: # 15 minutes
            anomalies.append({
                "severity": "WARN",
                "type": "LONG_DWELL",
                "message": f"Extremely long dwell time detected: {max_dwell // 60000} minutes.",
                "suggested_action": "Check for loitering or a trapped customer."
            })

        # Check traffic spike
        cursor.execute(f"""
            SELECT COUNT(*) FROM events 
            WHERE {base_cond} AND event_type = 'ENTRY' AND timestamp >= datetime('now', '-10 minute')
        """, (store_id,))
        recent_10m_entries = cursor.fetchone()[0] or 0
        if recent_10m_entries > 20:
            anomalies.append({
                "severity": "INFO",
                "type": "TRAFFIC_SPIKE",
                "message": f"Traffic spike detected: {recent_10m_entries} entries in the last 10 minutes.",
                "suggested_action": "Ensure all checkout counters are ready."
            })

                
        # Check conversion drop
        cursor.execute(f"""
            SELECT COUNT(DISTINCT visitor_id) FROM events WHERE {base_cond} AND event_type = 'ENTRY' AND timestamp >= datetime('now', '-1 hour')
        """, (store_id,))
        recent_entries = cursor.fetchone()[0] or 0
        
        cursor.execute(f"""
            SELECT COUNT(DISTINCT visitor_id) FROM events WHERE {base_cond} AND event_type = 'PURCHASE' AND timestamp >= datetime('now', '-1 hour')
        """, (store_id,))
        recent_purchases = cursor.fetchone()[0] or 0
        
        recent_conv = recent_purchases / recent_entries if recent_entries > 0 else 0
        
        if recent_entries > 10 and recent_conv < 0.10:
            anomalies.append({
                "severity": "WARN",
                "type": "CONVERSION_DROP",
                "message": f"Recent conversion rate is unusually low: {recent_conv*100:.1f}%",
                "suggested_action": "Check inventory levels and active promotions."
            })
            
        conn.close()
        return {"store_id": store_id, "anomalies": anomalies}
    except Exception as e:
        import logging
        logging.getLogger("store_intelligence").error(f"Error in get_anomalies for store {store_id}: {str(e)}", exc_info=True)
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail="Internal server error processing anomalies.")
