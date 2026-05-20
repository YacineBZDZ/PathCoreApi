import io
import gc
import os
import pickle
import base64
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torchvision.models.feature_extraction import create_feature_extractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR    = Path(__file__).parent / "models"
BACKBONE_PATH = MODELS_DIR / "efficientnet_b4.pth"  

# ── Global state ──────────────────────────────────────────────────────────────
STATE = {
    "extractor": None,
    "models":    {},
    "device":    None,
    "transform": None,
}


# ── Custom unpickler so PatchCore resolves correctly ──────────────────────────
class PatchCore:
    def __init__(self):
        self.memory_bank = None
        self.knn         = None
        self.hw_shape    = None
        self.threshold   = 0.35

    def score_image(self, patch_features: np.ndarray) -> float:
        dists, _ = self.knn.kneighbors(patch_features)
        return float(dists[:, 0].max())

    def score_map(self, patch_features: np.ndarray) -> np.ndarray:
        H, W = self.hw_shape
        dists, _ = self.knn.kneighbors(patch_features)
        scores   = dists[:, 0].reshape(H, W)
        scores_up = cv2.resize(scores, (224, 224), interpolation=cv2.INTER_LINEAR)
        return gaussian_filter(scores_up, sigma=4)


class _CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "PatchCore":
            return PatchCore
        return super().find_class(module, name)


def _load_pkl(path: Path):
    with open(path, "rb") as f:
        return _CustomUnpickler(f).load()


# ── Feature extractor — EfficientNet-B0 (~20 MB, fits free tier) ─────────────
class FeatureExtractor:
 
    def __init__(self, backbone_path: Path, device: torch.device):
        backbone = efficientnet_b4(weights=None)
        self.extractor = create_feature_extractor(
            backbone,
            return_nodes={
                "features.4": "layer2", 
                "features.6": "layer3"
            }
        )
        state = torch.load(backbone_path, map_location=device)
        self.extractor.load_state_dict(state)
        self.extractor = self.extractor.to(device).eval()

        # half-precision saves ~50% RAM at inference time
        if device.type == "cpu":
            self.extractor = self.extractor.float()   # CPU: keep float32 (half() unstable)
        else:
            self.extractor = self.extractor.half()    # GPU: use float16

        for p in self.extractor.parameters():
            p.requires_grad = False
        self.device = device
        logger.info(f"Backbone dtype: {next(self.extractor.parameters()).dtype}")

    @torch.no_grad()
    def extract(self, img_tensor: torch.Tensor):
        """img_tensor: (1, 3, 224, 224) float32"""
        img_tensor = img_tensor.to(self.device)
        # cast to match backbone dtype
        if next(self.extractor.parameters()).dtype == torch.float16:
            img_tensor = img_tensor.half()

        feats  = self.extractor(img_tensor)
        f2     = feats["layer2"]                                       # (1, C2, H2, W2)
        f3     = feats["layer3"]                                       # (1, C3, H3, W3)
        f3_up  = F.interpolate(f3, size=f2.shape[-2:],
                               mode="bilinear", align_corners=False)
        f2     = F.avg_pool2d(f2,    kernel_size=3, stride=1, padding=1)
        f3_up  = F.avg_pool2d(f3_up, kernel_size=3, stride=1, padding=1)
        combined = torch.cat([f2, f3_up], dim=1).float()              # back to float32
        B, C, H, W = combined.shape
        patches = combined.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return patches[0].cpu().numpy(), (H, W)


# ── Startup / shutdown ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        STATE["device"] = device
        logger.info(f"Device: {device}")

        STATE["transform"] = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        # Load backbone
        logger.info(f"Loading EfficientNet-B0 from {BACKBONE_PATH}...")
        if not BACKBONE_PATH.exists():
            raise FileNotFoundError(f"Backbone not found: {BACKBONE_PATH}")
        STATE["extractor"] = FeatureExtractor(BACKBONE_PATH, device)
        logger.info("Backbone loaded ✓")

        # Auto-discover pkl files — no hardcoding needed
        if not MODELS_DIR.exists():
            logger.warning(f"Models dir not found: {MODELS_DIR}")
        else:
            for pkl_path in sorted(MODELS_DIR.glob("patchcore_*.pkl")):
                category = pkl_path.stem.replace("patchcore_", "")
                model = _load_pkl(pkl_path)
                if not hasattr(model, "threshold"):
                    logger.warning(f"{category}: no threshold in pkl, using 0.35")
                    model.threshold = 0.35
                STATE["models"][category] = model
                logger.info(
                    f"Loaded {category}: {len(model.memory_bank):,} vectors, "
                    f"threshold={model.threshold:.4f} ✓"
                )

        # Free any temporary allocation
        gc.collect()
        logger.info(f"Startup complete — {len(STATE['models'])} categories ready")

    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise

    yield

    STATE["models"].clear()
    logger.info("Shutdown complete")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Industrial Anomaly Detection API",
    description="PatchCore + EfficientNet-B0 (memory-optimised for Render free tier)",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {
        "status": "online",
        "backbone": "EfficientNet-B0",
        "categories_loaded": list(STATE["models"].keys()),
        "device": str(STATE["device"]),
    }


@app.get("/categories")
def get_categories():
    return {"categories": list(STATE["models"].keys())}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    category: str = Form(...),
):
    # Validate
    if category not in STATE["models"]:
        raise HTTPException(
            status_code=400,
            detail=f"Category '{category}' not available. "
                   f"Available: {list(STATE['models'].keys())}"
        )
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    # Read + preprocess
    raw       = await file.read()
    pil_img   = Image.open(io.BytesIO(raw)).convert("RGB")
    img_tensor = STATE["transform"](pil_img).unsqueeze(0)

    # Extract features
    patches, hw = STATE["extractor"].extract(img_tensor)

    # Score
    model     = STATE["models"][category]
    score     = model.score_image(patches)
    score_map = model.score_map(patches)

    # Heatmap overlay
    img_resized = pil_img.resize((224, 224))
    img_np      = np.array(img_resized, dtype=np.float32) / 255.0
    norm_map    = (score_map - score_map.min()) / (score_map.max() - score_map.min() + 1e-8)
    hmap_color  = cv2.applyColorMap((norm_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    hmap_rgb    = cv2.cvtColor(hmap_color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay     = np.clip(0.55 * img_np + 0.45 * hmap_rgb, 0, 1)

    def encode(arr: np.ndarray) -> str:
        _, buf = cv2.imencode(".jpg",
                              cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR),
                              [cv2.IMWRITE_JPEG_QUALITY, 88])
        return base64.b64encode(buf).decode()

    original_b64 = encode(np.array(img_resized))
    heatmap_b64  = encode(
        (norm_map * 255).astype(np.uint8)[:, :, None].repeat(3, axis=2)
    )
    overlay_b64  = encode((overlay * 255).astype(np.uint8))

    # Verdict — threshold comes from pkl, no hardcoding
    threshold  = model.threshold
    is_anomaly = score > threshold

    return JSONResponse({
        "score":      round(float(score), 6),
        "threshold":  round(float(threshold), 6),
        "is_anomaly": is_anomaly,
        "verdict":    "Defect detected" if is_anomaly else "Normal — no anomaly",
        "category":   category,
        "images": {
            "original": original_b64,
            "heatmap":  heatmap_b64,
            "overlay":  overlay_b64,
        }
    })
