"""Pydantic event schema — the contract between pipeline and API.

The Event model validates every field on every incoming request.
If any field is missing or has the wrong type, FastAPI returns 422
before our route handler is even called.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class EventMetadata(BaseModel):
    queue_depth: Optional[int] = None
    sku_zone: Optional[str] = None
    session_seq: Optional[int] = None
    zone_id: Optional[str] = None
    dwell_ms: Optional[int] = None
    signature: Optional[list] = None

class Event(BaseModel):
    event_id: str                                   # uuid-v4, globally unique
    store_id: str                                   # from store_layout.json
    camera_id: str                                  # which camera produced this
    visitor_id: str                                 # Re-ID token, unique per visit session
    event_type: str                                 # see Event Type Catalogue
    timestamp: str                                  # ISO-8601 UTC
    zone_id: Optional[str] = None                   # zone label; null for ENTRY/EXIT
    dwell_ms: int = 0                               # duration; 0 for instantaneous events
    is_staff: bool = False                          # model must classify this
    confidence: float                               # detection confidence
    metadata: EventMetadata = Field(default_factory=EventMetadata)
