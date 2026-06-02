"""Ingestion module — POST /events/ingest

Accepts batches of up to 500 events. Validates, deduplicates, and stores.
Partial success on malformed events. Structured response.
Idempotent by event_id (INSERT OR IGNORE).
"""

import json
import sqlite3
import logging
from typing import List

from fastapi import APIRouter, HTTPException

from app.models import Event
from app.database import get_db_connection

logger = logging.getLogger("store_intelligence")

router = APIRouter()


@router.post("/events/ingest")
def ingest_events(events_raw: List[dict]):
    """Ingest a batch of structured events from the detection pipeline.

    - Accepts up to 500 events per call.
    - Each event is validated against the Event Pydantic model.
    - Malformed events are skipped (partial success).
    - Duplicate event_ids are silently ignored (idempotent).
    """
    if len(events_raw) > 500:
        raise HTTPException(
            status_code=400,
            detail="Batch size exceeds maximum limit of 500"
        )

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Service Unavailable: Database connection failed"
        )

    events_received = 0
    duplicates_skipped = 0
    malformed_skipped = 0

    for event_dict in events_raw:
        # --- Validate against Pydantic schema ---
        try:
            event = Event(**event_dict)
        except Exception:
            malformed_skipped += 1
            continue

        # --- Write to SQLite ---
        metadata_str = event.metadata.json() if hasattr(event.metadata, 'json') else json.dumps(event.metadata.dict() if hasattr(event.metadata, 'dict') else {})
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO events (
                    event_id, store_id, camera_id, visitor_id, event_type,
                    timestamp, zone_id, dwell_ms, is_staff, confidence, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id, event.store_id, event.camera_id,
                event.visitor_id, event.event_type, event.timestamp,
                event.zone_id, event.dwell_ms, event.is_staff,
                event.confidence, metadata_str
            ))
            if cursor.rowcount == 1:
                events_received += 1
            else:
                duplicates_skipped += 1
        except sqlite3.IntegrityError:
            duplicates_skipped += 1

    conn.commit()
    conn.close()

    logger.info(
        f"ingest event_count={events_received} "
        f"duplicates={duplicates_skipped} "
        f"malformed={malformed_skipped}"
    )

    return {
        "status": "success",
        "events_received": events_received,
        "duplicates_skipped": duplicates_skipped,
        "malformed_skipped": malformed_skipped
    }
