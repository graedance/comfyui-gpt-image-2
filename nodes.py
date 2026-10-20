import base64
import io
import math
import os

import numpy as np
import torch
from PIL import Image

try:
    from openai import OpenAI
except ImportError as e:
    raise ImportError(
        "The `openai` package is required. Install with: pip install -r requirements.txt"
    ) from e


# gpt-image-2 released 2026-04-21. gpt-image-2.5 released 2026-09-08 as two
# API models: Flare (OpenAI's default, faster than gpt-image-2) and Sunburst
# (slower, tighter control across edits). "thinking" is a ChatGPT-tier feature,
# not an API model. Snapshot IDs are offered alongside the floating aliases for
# reproducibility.
MODELS = [
    "gpt-image-2.5-flare",
    "gpt-image-2.5-sunburst",
    "gpt-image-2.5-flare-2026-09-08",
    "gpt-image-2.5-sunburst-2026-09-08",
    "gpt-image-2",
    "gpt-image-2-2026-04-21",
    "gpt-image-1.5",
    "gpt-image-1",
    "gpt-image-1-mini",
]
# gpt-image-2 accepts any resolution within constraints (max edge ≤3840, both
# edges multiples of 16, long:short ≤3:1, total pixels 655,360–8,294,400).
# gpt-image-1.x is restricted to 1024x1024 / 1024x1536 / 1536x1024 / auto.
# This list is the popular set from OpenAI's docs — the API will reject an
# invalid size for the chosen model with a clear error.
SIZES = [
    "auto",
    "1024x1024",
    "1024x1536",
    "1536x1024",
    "1792x1024",
    "1024x1792",
    "2048x1152",
    "2048x2048",
    "3840x2160",
    "2160x3840",
]
# "xhigh" and "max" are gpt-image-2.5 only; older models reject them.
QUALITIES = ["auto", "low", "medium", "high", "xhigh", "max"]
DEFAULT_MODEL = "gpt-image-2.5-flare"
FORMATS = ["png", "jpeg", "webp"]
BACKGROUNDS = ["auto", "opaque"]
MODERATIONS = ["auto", "low"]


# ---------- pricing & token tables ----------
# USD per 1M tokens. gpt-image-2 from OpenAI's 2026-04-21 pricing page.
# gpt-image-2.5 Flare and Sunburst bill at the same token rates as gpt-image-2
# (OpenAI model pages, checked 2026-10-20). gpt-image-1 retained for legacy.
_V2_RATES = {"text_in": 5.0, "image_in": 8.0, "image_cached": 2.0, "image_out": 30.0}
PRICING = {
    "gpt-image-2.5-flare": _V2_RATES,
    "gpt-image-2.5-sunburst": _V2_RATES,
    "gpt-image-2.5-flare-2026-09-08": _V2_RATES,
    "gpt-image-2.5-sunburst-2026-09-08": _V2_RATES,
    "gpt-image-2": {
        "text_in": 5.0, "image_in": 8.0, "image_cached": 2.0, "image_out": 30.0,
    },
    "gpt-image-2-2026-04-21": {
        "text_in": 5.0, "image_in": 8.0, "image_cached": 2.0, "image_out": 30.0,
    },
    "gpt-image-1.5": {
        "text_in": 5.0, "image_in": 10.0, "image_cached": 2.5, "image_out": 40.0,
    },
    "gpt-image-1": {
        "text_in": 5.0, "image_in": 10.0, "image_cached": 2.5, "image_out": 40.0,
    },
    "gpt-image-1-mini": {
        "text_in": 2.0, "image_in": 4.0,  "image_cached": 1.0, "image_out": 16.0,
    },
}

# Output image tokens per (quality, size).
# gpt-image-2 numbers back-calculated from OpenAI's published per-image USD
# (low $0.006, med $0.053, high $0.211 at 1024² ÷ $30/M output) and scaled by
# pixel ratio for non-square / 2K sizes. gpt-image-1 numbers are the documented
# reference values and are used when model == "gpt-image-1".
OUTPUT_TOKENS_V2 = {
    "low":    {"1024x1024":  200, "1792x1024":  350, "1024x1792":  350,
               "2048x2048":  800, "2560x1440":  720},
    "medium": {"1024x1024": 1767, "1792x1024": 3092, "1024x1792": 3092,
               "2048x2048": 7068, "2560x1440": 6361},
    "high":   {"1024x1024": 7033, "1792x1024":12308, "1024x1792":12308,
               "2048x2048":28132, "2560x1440":25319},
}
OUTPUT_TOKENS_V1 = {
    "low":    {"1024x1024":  272, "1024x1536":  408, "1536x1024":  400,
               "2048x2048": 1088},
    "medium": {"1024x1024": 1056, "1024x1536": 1584, "1536x1024": 1568,
               "2048x2048": 4224},
    "high":   {"1024x1024": 4160, "1024x1536": 6240, "1536x1024": 6208,
               "2048x2048":16640},
}


def _output_table(model: str) -> dict:
    # gpt-image-1, 1.5, mini all share the same per-image token math.
    return OUTPUT_TOKENS_V1 if model.startswith("gpt-image-1") else OUTPUT_TOKENS_V2


# OpenAI has not published output-token counts for xhigh / max, so the
# estimator bills them at the "high" table and flags the figure as a floor.
UNPUBLISHED_QUALITIES = ("xhigh", "max")


def _resolve(quality: str, size: str) -> tuple[str, str]:
    q = "high" if quality in ("auto",) + UNPUBLISHED_QUALITIES else quality
    s = "1024x1024" if size == "auto" else size
    return q, s


def _output_tokens_per_image(model: str, quality: str, size: str) -> int:
    q, s = _resolve(quality, size)
    table = _output_table(model)
    if s in table.get(q, {}):
        return table[q][s]
    # Fallback: scale 1024² baseline by pixel ratio for any custom resolution.
    base = table[q]["1024x1024"]
    try:
        w, h = (int(x) for x in s.split("x"))
        return int(round(base * (w * h) / (1024 * 1024)))
    except Exception:
        return base


def _estimate_text_tokens(prompt: str) -> int:
    # Cheap heuristic — ~4 chars per token. Avoids a tiktoken dep.
    if not prompt:
        return 0
    return max(1, math.ceil(len(prompt) / 4))


def _estimate_image_input_tokens(size: str) -> int:
    # gpt-image-1 charges roughly 85 base + 170 per 512px tile of the input image.
    # Use the requested output size as a proxy when no actual image dims are known.
    s = "1024x1024" if size == "auto" else size
    try:
        w, h = (int(x) for x in s.split("x"))
    except ValueError:
        w, h = 1024, 1024
    tiles = math.ceil(w / 512) * math.ceil(h / 512)
    return 85 + 170 * tiles


def _client(api_key: str | None):
    key = api_key.strip() if api_key else os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "OpenAI API key not provided. Set OPENAI_API_KEY or pass api_key on the node."
        )
    return OpenAI(api_key=key)


def _b64_to_tensor(b64: str) -> torch.Tensor:
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


def _stack_b64_list(items) -> torch.Tensor:
    tensors = [_b64_to_tensor(it.b64_json) for it in items]
    h = max(t.shape[0] for t in tensors)
    w = max(t.shape[1] for t in tensors)
    padded = []
    for t in tensors:
        if t.shape[0] != h or t.shape[1] != w:
            resized = Image.fromarray((t.numpy() * 255).astype(np.uint8)).resize(
                (w, h), Image.LANCZOS
            )
            t = torch.from_numpy(np.array(resized, dtype=np.float32) / 255.0)
        padded.append(t)
    return torch.stack(padded, dim=0)


def _tensor_to_png_bytes(image: torch.Tensor) -> bytes:
    if image.dim() == 4:
        image = image[0]
    arr = (image.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    buf.name = "image.png"
    return buf


def _mask_to_png_bytes(mask: torch.Tensor) -> bytes:
    if mask.dim() == 3:
        mask = mask[0]
    m = (mask.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    h, w = m.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 3] = 255 - m
    pil = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    buf.name = "mask.png"
    return buf


class GPTImage2Generate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODELS, {"default": DEFAULT_MODEL}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "size": (SIZES, {"default": "1024x1024"}),
                "quality": (QUALITIES, {"default": "high"}),
                "output_format": (FORMATS, {"default": "png"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 8}),
                "background": (BACKGROUNDS, {"default": "auto"}),
                "moderation": (MODERATIONS, {"default": "auto"}),
                "output_compression": ("INT", {"default": 100, "min": 0, "max": 100}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "OpenAI/GPT-Image-2"

    def generate(
        self,
        model,
        prompt,
        size,
        quality,
        output_format,
        n,
        background,
        moderation,
        output_compression,
        api_key="",
    ):
        client = _client(api_key)
        kwargs = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "output_format": output_format,
            "n": n,
            "background": background,
            "moderation": moderation,
        }
        if output_format in ("jpeg", "webp"):
            kwargs["output_compression"] = output_compression

        result = client.images.generate(**kwargs)
        return (_stack_b64_list(result.data),)


class GPTImage2Edit:
    """
    GPT-Image-2 supports multi-image reference editing — pass up to 9 reference
    images and the model composes / edits across them using the prompt.
    Each slot is optional; supply `image_1` at minimum. `mask` is applied to
    `image_1` only (OpenAI's `images.edit` mask semantics).
    """

    MAX_IMAGES = 9

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "model": (MODELS, {"default": DEFAULT_MODEL}),
            "prompt": ("STRING", {"multiline": True, "default": ""}),
            "size": (SIZES, {"default": "1024x1024"}),
            "quality": (QUALITIES, {"default": "high"}),
            "output_format": (FORMATS, {"default": "png"}),
            "n": ("INT", {"default": 1, "min": 1, "max": 8}),
            "background": (BACKGROUNDS, {"default": "auto"}),
            "output_compression": ("INT", {"default": 100, "min": 0, "max": 100}),
        }
        optional = {"mask": ("MASK",),
                    "api_key": ("STRING", {"default": "", "multiline": False})}
        for i in range(1, cls.MAX_IMAGES + 1):
            optional[f"image_{i}"] = ("IMAGE",)
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "edit"
    CATEGORY = "OpenAI/GPT-Image-2"

    def edit(
        self,
        model,
        prompt,
        size,
        quality,
        output_format,
        n,
        background,
        output_compression,
        mask=None,
        api_key="",
        **kwargs_images,
    ):
        # Collect image_1..image_9 in order, skipping unconnected slots.
        images = []
        for i in range(1, self.MAX_IMAGES + 1):
            t = kwargs_images.get(f"image_{i}")
            if t is not None:
                buf = _tensor_to_png_bytes(t)
                buf.name = f"image_{i}.png"
                images.append(buf)
        if not images:
            raise ValueError(
                "GPT Image 2 — Edit requires at least one input image "
                "(connect image_1)."
            )

        client = _client(api_key)
        # OpenAI's SDK accepts a list for `image` on gpt-image-* edit calls.
        # If only one image is supplied, pass it bare for max compatibility.
        kwargs = {
            "model": model,
            "image": images if len(images) > 1 else images[0],
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "output_format": output_format,
            "n": n,
            "background": background,
        }
        if output_format in ("jpeg", "webp"):
            kwargs["output_compression"] = output_compression
        if mask is not None:
            kwargs["mask"] = _mask_to_png_bytes(mask)

        result = client.images.edit(**kwargs)
        return (_stack_b64_list(result.data),)


class GPTImage2EstimateCost:
    """Dry-run estimator. No API call. Plug in the same prompt/size/quality
    you'd send to Generate or Edit and get back projected tokens + USD cost."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODELS, {"default": DEFAULT_MODEL}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "size": (SIZES, {"default": "1024x1024"}),
                "quality": (QUALITIES, {"default": "high"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 8}),
                "input_images": ("INT", {"default": 0, "min": 0, "max": 9,
                    "tooltip": "Number of reference images (0 for generate, 1–9 for edit)"}),
                "cached_input": ("BOOLEAN", {"default": False,
                    "tooltip": "Bill input images at the cached rate (gpt-image-2 and 2.5)"}),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("text_tokens", "image_in_tokens", "image_out_tokens",
                    "usd_cost", "summary")
    FUNCTION = "estimate"
    CATEGORY = "OpenAI/GPT-Image-2"
    OUTPUT_NODE = True

    def estimate(self, model, prompt, size, quality, n, input_images, cached_input):
        rates = PRICING.get(model, PRICING["gpt-image-2"])
        text_t = _estimate_text_tokens(prompt)
        img_in_t = input_images * _estimate_image_input_tokens(size)
        img_out_t = n * _output_tokens_per_image(model, quality, size)

        img_in_rate = rates["image_cached"] if cached_input else rates["image_in"]
        cost = (
            text_t * rates["text_in"]
            + img_in_t * img_in_rate
            + img_out_t * rates["image_out"]
        ) / 1_000_000

        q, s = _resolve(quality, size)
        if quality in UNPUBLISHED_QUALITIES:
            q = f"{quality} (priced as high, real cost is higher)"
        in_label = "image in (cached)" if cached_input else "image in"
        summary = (
            f"{model} | {n}× {s} {q}\n"
            f"  text in:  {text_t:>7} tok  → ${text_t * rates['text_in'] / 1e6:.4f}\n"
            f"  {in_label}: {img_in_t:>5} tok  → ${img_in_t * img_in_rate / 1e6:.4f}\n"
            f"  image out:{img_out_t:>7} tok  → ${img_out_t * rates['image_out'] / 1e6:.4f}\n"
            f"  total:                    ${cost:.4f}"
        )
        return (text_t, img_in_t, img_out_t, round(cost, 6), summary)


NODE_CLASS_MAPPINGS = {
    "GPTImage2Generate": GPTImage2Generate,
    "GPTImage2Edit": GPTImage2Edit,
    "GPTImage2EstimateCost": GPTImage2EstimateCost,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPTImage2Generate": "GPT Image 2 — Generate",
    "GPTImage2Edit": "GPT Image 2 — Edit",
    "GPTImage2EstimateCost": "GPT Image 2 — Estimate Cost",
}
