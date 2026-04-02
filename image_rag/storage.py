from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


VECTORS_FILE = "vectors.npy"
PATHS_FILE = "paths.json"
META_FILE = "meta.json"
DB_FILE = "index.db"


IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp",
    ".tif",
    ".tiff",
}


def collect_images(images_dir: str, recursive: bool = True) -> List[str]:
    root = Path(images_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise NotADirectoryError(f"Image directory not found: {root}")

    iterator = root.rglob("*") if recursive else root.glob("*")
    images = [str(p.resolve()) for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    images.sort()
    return images


def save_database(
    db_dir: str,
    image_paths: List[str],
    vectors: np.ndarray,
    model_path: str,
    model_name: str,
) -> None:
    target = Path(db_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    db_path = target / DB_FILE

    dim = int(vectors.shape[1]) if vectors.ndim == 2 and vectors.shape[0] > 0 else 512
    created_at = datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL UNIQUE,
                width INTEGER,
                height INTEGER,
                embedding_dim INTEGER NOT NULL,
                embedding BLOB NOT NULL
            );
            """
        )

        cur.execute("DELETE FROM meta;")
        cur.execute("DELETE FROM items;")

        meta_pairs = {
            "created_at": created_at,
            "model_name": model_name,
            "model_path": str(Path(model_path).expanduser().resolve()),
            "embedding_dim": str(dim),
            "count": str(len(image_paths)),
            "format": "sqlite",
        }
        cur.executemany(
            "INSERT INTO meta(key, value) VALUES(?, ?);",
            list(meta_pairs.items()),
        )

        rows = []
        for p, vec in zip(image_paths, vectors):
            width = None
            height = None
            try:
                with Image.open(p) as img:
                    width, height = img.size
            except Exception:
                pass

            vec32 = np.asarray(vec, dtype=np.float32)
            rows.append((p, width, height, int(vec32.shape[0]), sqlite3.Binary(vec32.tobytes())))

        cur.executemany(
            """
            INSERT INTO items(path, width, height, embedding_dim, embedding)
            VALUES(?, ?, ?, ?, ?);
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def _load_database_sqlite(db_dir: str) -> Tuple[List[str], np.ndarray, dict]:
    target = Path(db_dir).expanduser().resolve()
    db_path = target / DB_FILE
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database file not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()

        cur.execute("SELECT key, value FROM meta;")
        meta_rows = cur.fetchall()
        meta = {k: v for k, v in meta_rows}

        cur.execute("SELECT path, embedding_dim, embedding FROM items ORDER BY id ASC;")
        rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        raise ValueError(f"Empty database: {db_path}")

    image_paths: List[str] = []
    vectors_list: List[np.ndarray] = []
    for path, emb_dim, emb_blob in rows:
        vec = np.frombuffer(emb_blob, dtype=np.float32)
        if int(emb_dim) != vec.shape[0]:
            raise ValueError(
                f"Corrupted row for {path}: embedding_dim={emb_dim}, bytes_dim={vec.shape[0]}"
            )
        image_paths.append(path)
        vectors_list.append(vec)

    vectors = np.vstack(vectors_list).astype(np.float32)
    meta["count"] = len(image_paths)
    meta["embedding_dim"] = int(vectors.shape[1])
    return image_paths, vectors, meta


def _load_database_legacy(db_dir: str) -> Tuple[List[str], np.ndarray, dict]:
    target = Path(db_dir).expanduser().resolve()
    vectors_file = target / VECTORS_FILE
    paths_file = target / PATHS_FILE
    meta_file = target / META_FILE

    if not vectors_file.exists() or not paths_file.exists() or not meta_file.exists():
        raise FileNotFoundError(
            f"Invalid database folder: {target}. Expected {DB_FILE} or legacy {VECTORS_FILE}, {PATHS_FILE}, {META_FILE}."
        )

    vectors = np.load(vectors_file)
    with paths_file.open("r", encoding="utf-8") as f:
        image_paths = json.load(f)
    with meta_file.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    if len(image_paths) != len(vectors):
        raise ValueError(
            f"Corrupted database: number of paths ({len(image_paths)}) != vectors ({len(vectors)})."
        )
    return image_paths, vectors, meta


def load_database(db_dir: str) -> Tuple[List[str], np.ndarray, dict]:
    target = Path(db_dir).expanduser().resolve()

    # Accept both database directory and explicit sqlite file path.
    if target.is_file() and target.suffix.lower() == ".db":
        if target.name == DB_FILE:
            return _load_database_sqlite(str(target.parent))

        sibling_index = target.parent / DB_FILE
        if sibling_index.exists():
            return _load_database_sqlite(str(target.parent))

    db_path = target / DB_FILE
    if db_path.exists():
        return _load_database_sqlite(str(target))
    return _load_database_legacy(str(target))
