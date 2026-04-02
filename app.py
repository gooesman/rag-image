from __future__ import annotations

from pathlib import Path

import numpy as np

from image_rag.modeling import MODEL_NAME, encode_images, encode_text, load_model
from image_rag.storage import collect_images, load_database, save_database


def default_model_path() -> str:
    return str((Path(__file__).resolve().parent / "mod").resolve())


def ask(prompt: str, default: str | None = None) -> str:
    if default is None:
        value = input(prompt).strip()
        return value
    value = input(f"{prompt} [{default}]: ").strip()
    return value or default


def ask_int(prompt: str, default: int) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    return int(raw)


def ask_device(default: str = "cpu") -> str:
    value = ask("设备 (cpu/cuda)", default=default).lower()
    if value not in {"cpu", "cuda"}:
        raise ValueError("设备只能是 cpu 或 cuda")
    return value


def run_build_index() -> None:
    print("\n=== 功能 1: 生成本地向量库 ===")
    images_dir = ask("请输入本地图片目录: ")
    db_dir = ask("请输入向量库存储目录: ")
    model_path = ask("请输入模型路径", default_model_path())
    device = ask_device("cpu")
    batch_size = ask_int("batch-size", 32)

    print("\n[1/4] 扫描图片目录(递归)...")
    image_paths = collect_images(images_dir, recursive=True)
    if not image_paths:
        print("未找到图片，请检查目录。")
        return

    print(f"[2/4] 加载模型: {model_path}")
    model, processor, _ = load_model(model_path, device=device)

    print(f"[3/4] 提取特征，共 {len(image_paths)} 张...")
    last_pct = {"value": -1}

    def on_progress(done: int, total: int) -> None:
        pct = int(done * 100 / total) if total > 0 else 100
        if pct != last_pct["value"]:
            print(f"    progress: {done}/{total} ({pct}%)")
            last_pct["value"] = pct

    vectors = encode_images(
        model=model,
        processor=processor,
        image_paths=image_paths,
        device=device,
        batch_size=batch_size,
        progress_callback=on_progress,
    )

    print(f"[4/4] 保存向量库: {db_dir}")
    save_database(
        db_dir=db_dir,
        image_paths=image_paths,
        vectors=vectors,
        model_path=model_path,
        model_name=MODEL_NAME,
    )
    print("完成: 向量库已生成。")


def run_search() -> None:
    print("\n=== 功能 2: 检索本地向量库 ===")
    db_dir = ask("请输入向量库目录: ")
    query = ask("请输入检索关键词: ")
    top_k = ask_int("top-k", 10)
    model_override = ask("模型路径(直接回车使用向量库记录)", default="")
    device = ask_device("cpu")

    print("\n[1/4] 加载向量库...")
    image_paths, vectors, meta = load_database(db_dir)

    model_path = model_override or meta.get("model_path") or default_model_path()
    print(f"[2/4] 加载模型: {model_path}")
    model, _, tokenizer = load_model(model_path, device=device)

    print("[3/4] 文本编码...")
    text_vec = encode_text(model, tokenizer, query, device=device)

    print("[4/4] 相似度检索...")
    scores = vectors @ text_vec
    k = max(1, min(top_k, len(scores)))
    top_indices = np.argsort(-scores)[:k]

    print("\n检索结果:")
    for rank, idx in enumerate(top_indices, start=1):
        print(f"{rank}. score={scores[idx]:.4f}  path={image_paths[idx]}")


def main() -> None:
    print("纯本地图片检索系统 (Chinese-CLIP)")
    print("1) 生成向量库")
    print("2) 检索向量库")
    choice = ask("请选择功能(1/2): ")

    if choice == "1":
        run_build_index()
    elif choice == "2":
        run_search()
    else:
        print("无效选择，请输入 1 或 2。")


if __name__ == "__main__":
    main()
