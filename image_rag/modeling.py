from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Sequence

import numpy as np
import open_clip
import torch
from PIL import Image

try:
    from mobileclip.modules.common.mobileone import reparameterize_model
    _MOBILECLIP_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    reparameterize_model = None
    _MOBILECLIP_IMPORT_ERROR = exc


MODEL_NAME = "MobileCLIP2-S0"


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def load_model(model_path: str, device: str = "cpu"):
    if _MOBILECLIP_IMPORT_ERROR is not None:
        raise ImportError(
            "mobileclip is not installed. Install dependencies with: pip install -r requirements.txt"
        ) from _MOBILECLIP_IMPORT_ERROR

    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=str(model_file),
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    model.eval()
    model = reparameterize_model(model)
    model = model.to(device)
    return model, preprocess, tokenizer


def encode_images(
    model,
    preprocess,
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
                images.append(preprocess(img))
            image_tensor = torch.stack(images).to(device)
            image_features = model.encode_image(image_tensor)
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

    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = _l2_normalize(text_features)

    return text_features.cpu().numpy().astype(np.float32)[0]
