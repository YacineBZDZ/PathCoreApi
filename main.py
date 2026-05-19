import io
import os
import json
import pickle
import base64
import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torchvision.models.feature_extraction import create_feature_extractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomUnpickler(pickle.Unpickler):
    """Custom unpickler that can resolve PatchCore from current module."""
    def find_class(self, module, name):
        if name == "PatchCore":
            return PatchCore
        return super().find_class(module, name)


def custom_unpickler(file_obj):
    """Unpickle with custom class resolution."""
    return CustomUnpickler(file_obj).load()

MODELS_DIR = Path(__file__).parent / "models"
METADATA_PATH = MODELS_DIR / "models_metadata.json"
BACKBONE_PATH = MODELS_DIR / "efficientnet_b4.pth"

STATE = {
    "extractor": None,
    "models": {},       
    "metadata": {},
    "device": None,
    "transform": None,
}


class PatchCore:
    def __init__(self):
        self.memory_bank = None
        self.knn = None
        self.hw_shape = None

    def score_image(self, patch_features: np.ndarray) -> float:
        dists, _ = self.knn.kneighbors(patch_features)
        return float(dists[:, 0].max())

    def score_map(self, patch_features: np.ndarray) -> np.ndarray:
        H, W = self.hw_shape
        dists, _ = self.knn.kneighbors(patch_features)
        scores = dists[:, 0].reshape(H, W)
        scores_up = cv2.resize(scores, (224, 224), interpolation=cv2.INTER_LINEAR)
        return gaussian_filter(scores_up, sigma=4)


class FeatureExtractor:
    def __init__(self, backbone_path: Path, device: torch.device):
        backbone = efficientnet_b4(weights=None)
        self.extractor = create_feature_extractor(
            backbone,
            return_nodes={"features.4": "layer2", "features.6": "layer3"}
        )
        self.extractor.load_state_dict(
            torch.load(backbone_path, map_location=device)
        )
        self.extractor = self.extractor.to(device).eval()
        for p in self.extractor.parameters():
            p.requires_grad = False
        self.device = device

    @torch.no_grad()
    def extract(self, img_tensor: torch.Tensor):
        """img_tensor: (1, 3, 224, 224)"""
        img_tensor = img_tensor.to(self.device)
        feats = self.extractor(img_tensor)
        f2 = feats["layer2"]
        f3 = feats["layer3"]
        f3_up = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        f2 = F.avg_pool2d(f2, kernel_size=3, stride=1, padding=1)
        f3_up = F.avg_pool2d(f3_up, kernel_size=3, stride=1, padding=1)
        combined = torch.cat([f2, f3_up], dim=1)
        B, C, H, W = combined.shape
        patches = combined.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return patches[0].cpu().numpy(), (H, W)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        STATE["device"] = device
        logger.info(f"Device: {device}")

        # Build transform (standard ImageNet preprocessing — same for all categories)
        STATE["transform"] = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        logger.info("Loading EfficientNet-B4 backbone...")
        if not BACKBONE_PATH.exists():
            logger.error(f"Backbone file not found: {BACKBONE_PATH}")
            raise FileNotFoundError(f"Backbone file not found: {BACKBONE_PATH}")
        STATE["extractor"] = FeatureExtractor(BACKBONE_PATH, device)
        logger.info("Backbone loaded ✓")

        if not MODELS_DIR.exists():
            logger.warning(f"Models directory not found: {MODELS_DIR}")
        else:
            for pkl_path in sorted(MODELS_DIR.glob("patchcore_*.pkl")):
                category = pkl_path.stem.replace("patchcore_", "")
                with open(pkl_path, "rb") as f:
                    model = custom_unpickler(f)
                if not hasattr(model, "threshold"):
                    logger.warning(f"{category}: no threshold found in pkl — using fallback 0.35")
                    model.threshold = 0.35
                STATE["models"][category] = model
                nb = len(model.memory_bank)
                logger.info(f"Loaded {category}: {nb:,} vectors, threshold={model.threshold:.4f} ✓")

        logger.info(f"Ready — {len(STATE['models'])} categories loaded")
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise
    
    yield

    STATE["models"].clear()
    logger.info("Shutdown complete")


app = FastAPI(
    title="Industrial Anomaly Detection API",
    description="PatchCore + EfficientNet-B4 - MVTec AD",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


#  Routes 
@app.get("/")
def health():
    return {
        "status": "online",
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
    #  Validate
    if category not in STATE["models"]:
        raise HTTPException(
            status_code=400,
            detail=f"Category '{category}' not available. "
                   f"Available: {list(STATE['models'].keys())}"
        )
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    #  Read & preprocess image
    raw = await file.read()
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    img_tensor = STATE["transform"](pil_img).unsqueeze(0)  # (1, 3, 224, 224)

    # Extract features 
    extractor = STATE["extractor"]
    patches, hw = extractor.extract(img_tensor)            # (H*W, C), (H, W)

    #  Score
    model = STATE["models"][category]
    score = model.score_image(patches)
    score_map = model.score_map(patches)

    #  heatmap overlay 
    img_resized = pil_img.resize((224, 224))
    img_np = np.array(img_resized, dtype=np.float32) / 255.0

    norm_map = (score_map - score_map.min()) / (score_map.max() - score_map.min() + 1e-8)
    hmap_color = cv2.applyColorMap((norm_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    hmap_rgb = cv2.cvtColor(hmap_color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = np.clip(0.55 * img_np + 0.45 * hmap_rgb, 0, 1)
    overlay_uint8 = (overlay * 255).astype(np.uint8)

    #  Encode images 
    def encode(arr: np.ndarray) -> str:
        _, buf = cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR),
                              [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.b64encode(buf).decode()

    original_b64 = encode(np.array(img_resized))
    heatmap_b64  = encode((norm_map * 255).astype(np.uint8)[:, :, None]
                          .repeat(3, axis=2))
    overlay_b64  = encode(overlay_uint8)

    # Threshold from pkl 
   
    threshold = model.threshold
    is_anomaly = score > threshold

    return JSONResponse({
        "score":       round(float(score), 6),
        "threshold":   round(threshold, 6),
        "is_anomaly":  is_anomaly,
        "verdict":     "Defect detected" if is_anomaly else "Normal — no anomaly",
        "category":    category,
        "images": {
            "original": original_b64,
            "heatmap":  heatmap_b64,
            "overlay":  overlay_b64,
        }
    })
