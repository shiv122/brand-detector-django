"""
Classification model service for YOLO classification models
"""
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from config.app_config import AppConfig
from apps.core.enums import DeviceType


def _is_rq_worker() -> bool:
    """Skip preload when running as an rqworker — see model_service.py."""
    if os.getenv("SKIP_MODEL_PRELOAD", "").lower() in ("1", "true", "yes"):
        return True
    return any("rqworker" in arg for arg in sys.argv)


class ClassificationModelService:
    """Service for managing YOLO classification models"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.models: Dict[str, YOLO] = {}
        self.current_model: Optional[YOLO] = None
        self.classification_weights_dir = Path(self.config.weights_dir) / "classification_weights"
        self.classification_weights: List[Dict] = []
        if _is_rq_worker():
            print("⏭️  Skipping ClassificationModelService init (rqworker context)")
            self.device = DeviceType.CPU.value
            return
        self.device = self._get_optimal_device()
        self._load_available_weights()
        self._preload_all_models()
    
    def _get_optimal_device(self) -> str:
        """Get the optimal device for inference"""
        if torch.backends.mps.is_available():
            print("🚀 Using MPS (Metal Performance Shaders) for Apple Silicon")
            return DeviceType.MPS.value
        elif torch.cuda.is_available():
            print("🚀 Using CUDA GPU")
            return DeviceType.CUDA.value
        else:
            print("⚠️ Using CPU (no GPU acceleration)")
            return DeviceType.CPU.value
    
    def _load_available_weights(self):
        """Load all available weights from the classification_weights directory"""
        if not self.classification_weights_dir.exists():
            print(f"❌ Classification weights directory not found: {self.classification_weights_dir}")
            return
        
        for weight_file in self.classification_weights_dir.glob("*.pt"):
            try:
                size = weight_file.stat().st_size
                self.classification_weights.append({
                    "name": weight_file.name,
                    "path": str(weight_file),
                    "size": size,
                    "description": f"YOLO Classification model ({self._format_size(size)})"
                })
                print(f"✅ Found classification weight: {weight_file.name} ({self._format_size(size)})")
            except Exception as e:
                print(f"❌ Error loading weight {weight_file.name}: {str(e)}")
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format"""
        if size_bytes == 0:
            return "0B"
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        return f"{size_bytes:.1f}{size_names[i]}"

    def _preload_all_models(self):
        """Preload all available classification models into memory at startup"""
        if _is_rq_worker():
            print("⏭️  Skipping classification-model preload (rqworker context)")
            return
        print(f"🔄 Preloading all {len(self.classification_weights)} classification models...")
        for weight_info in self.classification_weights:
            weight_name = weight_info["name"]
            if weight_name not in self.models:
                try:
                    weight_path = self.classification_weights_dir / weight_name
                    print(f"🔄 Loading classification model: {weight_name} on {self.device}")
                    model = YOLO(str(weight_path))
                    model.to(self.device)

                    if self.device == DeviceType.MPS.value:
                        model.model.eval()

                    self.models[weight_name] = model
                    print(f"✅ Classification model loaded: {weight_name} on {self.device}")
                except Exception as e:
                    print(f"❌ Error loading classification model {weight_name}: {str(e)}")

        # Set default current_model for backward compatibility
        default = self.config.selected_classification_weight
        if default in self.models:
            self.current_model = self.models[default]
        elif self.models:
            first_name = next(iter(self.models))
            self.current_model = self.models[first_name]

        print(f"✅ All classification models preloaded: {list(self.models.keys())}")

    def get_model(self, weight_name: str) -> YOLO:
        """Get a specific model by weight name"""
        if weight_name in self.models:
            return self.models[weight_name]
        # Attempt lazy load as fallback
        success = self.switch_model(weight_name)
        if success and weight_name in self.models:
            return self.models[weight_name]
        raise ValueError(f"Classification model not found: {weight_name}")

    def switch_model(self, weight_name: str) -> bool:
        """Switch to a different weight file"""
        try:
            print(f"🔄 Attempting to switch to classification model: {weight_name}")
            
            weight_path = self.classification_weights_dir / weight_name
            if not weight_path.exists():
                print(f"❌ Classification weight file not found: {weight_path}")
                return False
            
            if weight_name not in self.models:
                print(f"🔄 Loading classification model: {weight_name} on {self.device}")
                model = YOLO(str(weight_path))
                model.to(self.device)
                
                if self.device == DeviceType.MPS.value:
                    model.model.eval()
                
                self.models[weight_name] = model
                print(f"✅ Classification model loaded successfully: {weight_name} on {self.device}")
            else:
                print(f"📦 Using cached classification model: {weight_name}")
            
            self.current_model = self.models[weight_name]
            print(f"✅ Switched to classification model: {weight_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error switching to classification model {weight_name}: {str(e)}")
            return False
    
    def is_loaded(self) -> bool:
        """Check if any model is loaded"""
        return self.current_model is not None
    
    def get_current_model_name(self) -> str:
        """Get the name of the currently loaded model"""
        if self.current_model is None:
            return ""
        for name, model in self.models.items():
            if model == self.current_model:
                return name
        return ""
    
    def get_available_weights(self) -> List[Dict]:
        """Get list of available classification weights"""
        return self.classification_weights
    
    def classify_image(self, image_data: bytes, top_k: int = 5,
                       weight_name: Optional[str] = None) -> List[Dict]:
        """Classify an image and return top-k predictions"""
        model = self.get_model(weight_name) if weight_name else self.current_model
        if model is None:
            raise RuntimeError("Classification model not loaded")

        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Could not decode image")

        results = model(
            img,
            save=False,
            device=self.device,
            verbose=False
        )

        classifications = []

        for result in results:
            if hasattr(result, 'probs'):
                probs = result.probs

                if hasattr(probs, 'data'):
                    probs_tensor = probs.data
                    if self.device in [DeviceType.MPS.value, DeviceType.CUDA.value]:
                        all_probs = probs_tensor.cpu().numpy()
                    else:
                        all_probs = probs_tensor.numpy()

                    top_k_indices = np.argsort(all_probs)[-top_k:][::-1]
                    top_k_confidences = all_probs[top_k_indices]
                elif hasattr(probs, 'top5') and hasattr(probs, 'top5conf'):
                    top5_indices = probs.top5[:top_k]
                    top5_confidences = probs.top5conf[:top_k]
                    top_k_indices = top5_indices
                    top_k_confidences = top5_confidences
                else:
                    try:
                        all_probs = np.array(probs)
                        top_k_indices = np.argsort(all_probs)[-top_k:][::-1]
                        top_k_confidences = all_probs[top_k_indices]
                    except:
                        print("Warning: Could not extract probabilities from classification result")
                        continue

                if hasattr(result, 'names') and result.names:
                    class_names = result.names
                elif hasattr(model, 'names') and model.names:
                    class_names = model.names
                else:
                    class_names = {}

                for idx, conf in zip(top_k_indices, top_k_confidences):
                    class_id = int(idx)
                    class_name = class_names.get(class_id, f"Class_{class_id}")
                    confidence = float(conf)

                    classifications.append({
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence
                    })
            else:
                print("Warning: Classification result does not have probs attribute")

        classifications.sort(key=lambda x: x["confidence"], reverse=True)
        return classifications[:top_k]

