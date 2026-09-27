"""Debug script to test database table visibility."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

_DB_FILE = "test_confidence_debug.db"
_DB_ABS_PATH = os.path.abspath(_DB_FILE)

# Set env so db.py reads it
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_DB_ABS_PATH}"

# 1. Create tables synchronously
from sqlalchemy import create_engine, text
from db import Base
sync_url = f"sqlite:///{_DB_ABS_PATH}"
print(f"Sync URL: {sync_url}")
print(f"Async URL: {os.environ['DATABASE_URL']}")

sync_engine = create_engine(sync_url, connect_args={"check_same_thread": False})
Base.metadata.create_all(sync_engine)
print("Tables created sync")

# Check tables in sync engine
with sync_engine.connect() as conn:
    rows = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()
    print(f"Tables in sync engine: {rows}")
sync_engine.dispose()

# 2. Check if async engine can see them
import asyncio
from db import async_engine
async def check():
    async with async_engine.begin() as conn:
        from sqlalchemy import text as t
        rows = await conn.run_sync(lambda sync_conn: sync_conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall())
        print(f"Tables in async engine: {rows}")

asyncio.run(check())

# Cleanup
try: os.unlink(_DB_ABS_PATH)
except: pass
