"""Serve the atlas, process calendar uploads, and receive suggestions."""

from __future__ import annotations

import os
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import BoundedSemaphore
from typing import Annotated

import duckdb
import pyarrow
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.web_workspace import analyze_upload

app = FastAPI(title="Sutellite")
WEB_DIRECTORY = Path(__file__).resolve().parent / "web"
SUGGESTIONS_DATABASE = WEB_DIRECTORY.parent / "data" / "suggestions.sqlite3"
MAX_UPLOAD = 25 * 1024 * 1024
MAX_REQUEST = MAX_UPLOAD + 64 * 1024  # Allow the multipart headers and region field.
POST_LIMITS = {
    "/api/analyze": (MAX_REQUEST, "The file is larger than 25 MiB."),
    "/api/suggestions": (8192, "The suggestion is too large."),
}
processing_slot = BoundedSemaphore()


class LimitPostBody:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        limit = POST_LIMITS.get(scope["path"]) if scope["type"] == "http" and scope["method"] == "POST" else None
        if limit is None:
            await self.app(scope, receive, send)
            return
        max_bytes, detail = limit
        headers = dict(scope["headers"])
        length = headers.get(b"content-length")
        if length is not None and (not length.isdigit() or len(length) > 16 or int(length) > max_bytes):
            await JSONResponse({"detail": detail}, 413)(scope, receive, send)
            return

        received = 0

        async def read_limited():
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > max_bytes:
                raise HTTPException(413, detail)
            return message

        await self.app(scope, read_limited, send)


app.add_middleware(LimitPostBody)
web_origin = os.environ.get("SUTELLITE_WEB_ORIGIN", "").strip().rstrip("/")
if web_origin:
    app.add_middleware(CORSMiddleware, allow_origins=[web_origin], allow_methods=["GET", "POST"])


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/suggestions", status_code=201)
def save_suggestion(message: Annotated[str, Form(min_length=1, max_length=500)]):
    message = message.strip()
    if not message:
        raise HTTPException(422, "Enter a suggestion before sending it.")
    SUGGESTIONS_DATABASE.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(SUGGESTIONS_DATABASE)) as database, database:
        database.execute("""
            CREATE TABLE IF NOT EXISTS suggestions (
                id INTEGER PRIMARY KEY,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        database.execute("INSERT INTO suggestions (message) VALUES (?)", (message,))
    return {"saved": True}


@app.post("/api/analyze")
def analyze(
    file: Annotated[UploadFile, File()],
    region: Annotated[str, Form(max_length=80)] = "",
):
    filename = Path((file.filename or "").replace("\\", "/")).name
    suffix = Path(filename).suffix.lower()
    if suffix not in {".csv", ".parquet"}:
        raise HTTPException(415, "Choose a CSV or Parquet file.")
    if file.size is None or file.size > MAX_UPLOAD:
        raise HTTPException(413, "The file is larger than 25 MiB.")
    if not processing_slot.acquire(blocking=False):
        raise HTTPException(503, "An upload is being processed. Try again in a moment.")
    try:
        with TemporaryDirectory(prefix="sutellite-upload-") as directory:
            path = Path(directory) / ("detections" + suffix)
            with path.open("wb") as saved:
                shutil.copyfileobj(file.file, saved)
            worldcover = os.environ.get("SUTELLITE_WORLDCOVER_DIR")
            return analyze_upload(path, filename, region.strip(), Path(worldcover) if worldcover else None)
    except UnicodeError as exc:
        raise HTTPException(422, "Save the CSV as UTF-8 and upload it again.") from exc
    except (pyarrow.ArrowInvalid, pyarrow.ArrowNotImplementedError) as exc:
        raise HTTPException(422, "The Parquet file could not be read.") from exc
    except duckdb.OutOfMemoryException as exc:
        raise HTTPException(413, "The file needs too much memory to process. Try a smaller extract.") from exc
    except (ValueError, OverflowError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        processing_slot.release()


app.mount("/", StaticFiles(directory=WEB_DIRECTORY, html=True), name="atlas")
