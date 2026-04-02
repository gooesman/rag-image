from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Sequence

import numpy as np
import torch
from PIL import Image
from transformers import ChineseCLIPModel, ChineseCLIPProcessor


MODEL_NAME = "AI-ModelScope/chinese-clip-vit-base-patch16"


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def _feature_tensor(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError(f"Unsupported feature output type: {type(output)!r}")


def _resolve_model_dir(model_path: str) -> Path:
    model_file = Path(model_path)
    if model_file.is_file():
        model_file = model_file.parent
    if not model_file.exists() or not model_file.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_file}")
    return model_file


def load_model(model_path: str, device: str = "cpu"):
    model_dir = _resolve_model_dir(model_path)

    model = ChineseCLIPModel.from_pretrained(model_dir, local_files_only=True)
    processor = ChineseCLIPProcessor.from_pretrained(model_dir, local_files_only=True)

    model.eval()
    model = model.to(device)
    return model, processor, processor.tokenizer


def encode_images(
    model,
    processor,
    image_paths: Sequence[str],
    device: str = "cpu",
    batch_size: int = 32,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    vectors: List[np.ndarray] = []
    total = len(image_paths)
    processed = 0

    with torch.no_grad():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            images = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                images.append(img)
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            image_features = _feature_tensor(model.get_image_features(**inputs))
            image_features = _l2_normalize(image_features)
            vectors.append(image_features.cpu().numpy().astype(np.float32))
            processed += len(batch_paths)
            if progress_callback is not None:
                progress_callback(processed, total)

    if not vectors:
        return np.empty((0, 512), dtype=np.float32)
    return np.vstack(vectors)


def encode_text(model, tokenizer, text: str, device: str = "cpu") -> np.ndarray:
    if not text.strip():
        raise ValueError("Text query cannot be empty.")

    inputs = tokenizer(text=[text], padding=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        text_features = _feature_tensor(model.get_text_features(**inputs))
        text_features = _l2_normalize(text_features)

    return text_features.cpu().numpy().astype(np.float32)[0]
