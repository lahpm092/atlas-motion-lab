import os,json
from pathlib import Path
import numpy as np
_sam=None

def sam31_dir():
    path=Path(os.environ.get("FIGHTLAB_SAM31",str(Path(__file__).resolve().parents[1]/"models/sam3.1-4bit")))
    if not (path/"model.safetensors").exists():
        raise FileNotFoundError("Set FIGHTLAB_SAM31 to your downloaded MLX checkpoint directory")
    return path


def sam31_video(resolution):
    """Load the native Apple-silicon SAM 3.1 model and inference helpers."""
    global _sam
    if _sam is None:
        try:
            import mlx.core as mx
            from mlx_vlm.generate import wired_limit
            from mlx_vlm.models.sam3.generate import Sam3Predictor
            from mlx_vlm.models.sam3_1.generate import (
                _detect_with_backbone,
                _get_backbone_features,
            )
            from mlx_vlm.models.sam3_1.processing_sam3_1 import Sam31Processor
            from mlx_vlm.utils import load_model
        except ImportError as exc:
            raise SystemExit(
                "stage 03 requires mlx-vlm==0.4.3 in an Apple-silicon Python "
                "environment; see README.md"
            ) from exc

        mdir = sam31_dir()
        model = load_model(mdir)
        config = json.load(open(mdir / "config.json"))
        quant = config.get("quantization") or config.get("quantization_config")
        precision = (f"{quant['bits']}-bit {quant.get('mode', 'affine')}"
                     if quant else "fp32")
        processor = Sam31Processor.from_pretrained(str(mdir))
        processor.image_size = int(resolution)
        predictor = Sam3Predictor(model, processor, score_threshold=.12)
        _sam = {
            "mx": mx,
            "model": model,
            "processor": processor,
            "predictor": predictor,
            "backbone": _get_backbone_features,
            "detect": _detect_with_backbone,
            "wired_limit": wired_limit,
            "model_dir": mdir,
            "precision": precision,
        }
        print(f"loaded SAM 3.1 from {mdir} at {resolution}px", flush=True)
    return _sam

def mask_box(mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return np.zeros(4, np.float32)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)

def clip_to_detection(mask, box, pad=.08):
    """Remove disconnected decoder spill outside its own person detection.

    SAM concept masks can contain a correct occluded fighter plus distant mat
    or advertising-board islands.  The decoder's instance box remains local
    to the person, so a modest padded crop keeps limbs while dropping those
    unrelated pixels before colour scoring, pose crops, or rendering.
    """
    h, w = mask.shape
    x0, y0, x1, y1 = [float(v) for v in box]
    margin = pad * max(x1 - x0, y1 - y0)
    xa = max(0, int(np.floor(x0 - margin)))
    ya = max(0, int(np.floor(y0 - margin)))
    xb = min(w, int(np.ceil(x1 + margin)))
    yb = min(h, int(np.ceil(y1 + margin)))
    clean = np.zeros_like(mask, dtype=bool)
    if xb > xa and yb > ya:
        clean[ya:yb, xa:xb] = mask[ya:yb, xa:xb]
    return clean

def largest_component(mask):
    """Keep one connected SAM instance and discard detached false fragments.

    A concept mask can occasionally combine several tiny parts of the referee
    into one candidate.  Their union may span a person-sized bounding box even
    though no connected component is a fighter.  Keeping the largest component
    before all geometry and colour checks prevents that union-of-fragments
    failure while retaining the main fighter silhouette.
    """
    import cv2
    mask = np.asarray(mask, bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == component

def frame_features(sam, rgb):
    """Compute the SAM 3.1 TriViT backbone once for one source frame."""
    from PIL import Image
    image = Image.fromarray(rgb)
    inputs = sam["processor"].preprocess_image(image)
    return sam["backbone"](sam["model"], sam["mx"].array(inputs["pixel_values"]))
