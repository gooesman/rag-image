from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from image_rag.modeling import encode_text, load_model
from image_rag.storage import load_database


def default_model_path() -> str:
    return str((Path(__file__).resolve().parent / "mod").resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature 2 - Search local image vector database using Chinese-CLIP"
    )
    parser.add_argument("--db-dir", required=True, help="Local vector database folder to load")
    parser.add_argument("--query", required=True, help="Text query for retrieval")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top matches to return")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Local Chinese-CLIP model directory. If omitted, tries meta.json then project default.",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_paths, vectors, meta = load_database(args.db_dir)
    model_path = args.model_path or meta.get("model_path") or default_model_path()

    print(f"[1/4] Loading model from: {model_path}")
    model, _, tokenizer = load_model(model_path, device=args.device)

    print(f"[2/4] Loaded database images: {len(image_paths)}")
    print("[3/4] Encoding query text...")
    text_vec = encode_text(model, tokenizer, args.query, device=args.device)

    print("[4/4] Computing similarity and ranking...")
    scores = vectors @ text_vec
    top_k = max(1, min(args.top_k, len(scores)))
    top_indices = np.argsort(-scores)[:top_k]

    print("\nTop matches:")
    for rank, idx in enumerate(top_indices, start=1):
        print(f"{rank}. score={scores[idx]:.4f}  path={image_paths[idx]}")


if __name__ == "__main__":
    main()
