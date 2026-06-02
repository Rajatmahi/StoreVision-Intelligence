"""Funnel + session logic — GET /stores/{id}/funnel

Conversion funnel: Entry → Zone Visit → Billing Queue → Purchase.
Session is the unit, not raw events. Re-entries must not double-count a visitor.
Includes drop-off percentages by stage.
"""

from fastapi import APIRouter

from app.database import get_db_connection

router = APIRouter()


@router.get("/stores/{store_id}/funnel")
def get_funnel(store_id: str):
    """Session-based conversion funnel with drop-off rates."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        base = "store_id = ? AND is_staff = 0"

        cursor.execute(
            f"SELECT COUNT(DISTINCT visitor_id) FROM events WHERE {base} AND event_type = 'ENTRY'",
            (store_id,),
        )
        entries = cursor.fetchone()[0] or 0

        cursor.execute(
            f"SELECT COUNT(DISTINCT visitor_id) FROM events WHERE {base} AND event_type IN ('ZONE_ENTER', 'ZONE_EXIT', 'ZONE_DWELL')",
            (store_id,),
        )
        zone_visits = cursor.fetchone()[0] or 0

        cursor.execute(
            f"SELECT COUNT(DISTINCT visitor_id) FROM events WHERE {base} AND event_type = 'BILLING_QUEUE_JOIN'",
            (store_id,),
        )
        billing_queue = cursor.fetchone()[0] or 0

        from app.metrics import get_pos_metrics
        purchases, _, _ = get_pos_metrics(store_id, entries)

        conn.close()

        return {
            "store_id": store_id,
            "funnel": {
                "entries": entries,
                "zone_visits": zone_visits,
                "billing_queue": billing_queue,
                "purchases": purchases,
            },
            "drop_off": {
                "entry_to_zone": round(1 - (zone_visits / entries), 3) if entries > 0 else 0.0,
                "zone_to_billing": round(1 - (billing_queue / zone_visits), 3) if zone_visits > 0 else 0.0,
                "billing_to_purchase": round(1 - (purchases / billing_queue), 3) if billing_queue > 0 else 0.0,
            },
            "conversion_rate": round(purchases / entries, 3) if entries > 0 else 0.0,
        }
    except Exception as e:
        import logging
        logging.getLogger("store_intelligence").error(f"Error in get_funnel for store {store_id}: {str(e)}", exc_info=True)
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail="Internal server error processing funnel.")
