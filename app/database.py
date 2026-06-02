import sqlite3
import os

DB_PATH = os.environ.get("DB_PATH", "data/store_intelligence.db")

def get_db_connection():
    # Connect to the SQLite database file
    conn = sqlite3.connect(DB_PATH)
    # This lets us access columns by name (like a dictionary)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    # Ensure the 'data' directory exists
    os.makedirs("data", exist_ok=True)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create the events table if it does not already exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            store_id TEXT NOT NULL,
            camera_id TEXT NOT NULL,
            visitor_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            zone_id TEXT,
            dwell_ms INTEGER NOT NULL,
            is_staff BOOLEAN NOT NULL,
            confidence REAL NOT NULL,
            metadata TEXT
        )
    """)
    
    # Simple migration: add metadata if missing
    cursor.execute("PRAGMA table_info(events)")
    columns = [info["name"] for info in cursor.fetchall()]
    if "metadata" not in columns and len(columns) > 0:
        cursor.execute("ALTER TABLE events ADD COLUMN metadata TEXT")
    # Save the changes and close connection
    conn.commit()
    conn.close()
