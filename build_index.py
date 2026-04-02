from __future__ import annotations

import argparse
from pathlib import Path

from image_rag.modeling import MODEL_NAME, encode_images, load_model
from image_rag.storage import collect_images, save_database


def default_model_path() -> str:
    return str((Path(__file__).resolve().parent / "mod" / "mobileclip2_s0.pt").resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature 1 - Build local image vector database using MobileCLIP2-S0"
    )
    parser.add_argument("--images-dir", required=True, help="Local image folder to read")
    parser.add_argument("--db-dir", required=True, help="Local folder where vector database will be saved")
    parser.add_argument("--model-path", default=default_model_path(), help="Local MobileCLIP2-S0 .pt checkpoint")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for image embedding")
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not recursively scan subfolders under images-dir",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_paths = collect_images(args.images_dir, recursive=not args.no_recursive)
    if not image_paths:
        raise SystemExit("No images found in the specified folder.")

    print(f"[1/4] Loading model from: {args.model_path}")
    model, preprocess, _ = load_model(args.model_path, device=args.device)

    print(f"[2/4] Found images: {len(image_paths)}")
    print("[3/4] Extracting image embeddings...")

    last_pct = {"value": -1}

    def on_progress(done: int, total: int) -> None:
        pct = int(done * 100 / total) if total > 0 else 100
        if pct != last_pct["value"]:
            print(f"    progress: {done}/{total} ({pct}%)")
            last_pct["value"] = pct

    vectors = encode_images(
        model=model,
        preprocess=preprocess,
        image_paths=image_paths,
        device=args.device,
        batch_size=args.batch_size,
        progress_callback=on_progress,
    )

    print(f"[4/4] Saving database to: {args.db_dir}")
    save_database(
        db_dir=args.db_dir,
        image_paths=image_paths,
        vectors=vectors,
        model_path=args.model_path,
        model_name=MODEL_NAME,
    )
    print("Done. Local vector database generated successfully.")


if __name__ == "__main__":
    main()
