import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from config.app_config import AppConfig
from apps.core.enums import DeviceType


class DetectionResult:
    """Simple data class for detection results"""
    
    def __init__(self, bbox: List[float], confidence: float, class_id: int, class_name: str):
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
    
    def to_dict(self):
        return {
            "bbox": self.bbox,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


class ModelService:
    """Service for managing YOLO detection models"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.models: Dict[str, YOLO] = {}  # Cache for loaded models
        self.current_model: Optional[YOLO] = None
        self.device = self._get_optimal_device()
        self.available_weights: List[Dict] = []
        self._load_available_weights()
        self._load_default_model()
    
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
        """Load all available weights from the weights directory"""
        weights_dir = Path(self.config.weights_dir)
        if not weights_dir.exists():
            print(f"❌ Weights directory not found: {weights_dir}")
            return
        
        for weight_file in weights_dir.glob("*.pt"):
            try:
                size = weight_file.stat().st_size
                self.available_weights.append({
                    "name": weight_file.name,
                    "path": str(weight_file),
                    "size": size,
                    "description": f"YOLO model ({self._format_size(size)})",
                })
                print(f"✅ Found weight: {weight_file.name} ({self._format_size(size)})")
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
    
    def _load_default_model(self):
        """Load the default model"""
        self.switch_model(self.config.selected_weight)
    
    def switch_model(self, weight_name: str) -> bool:
        """Switch to a different weight file"""
        try:
            print(f"🔄 Attempting to switch to model: {weight_name}")
            
            # Check if weight exists
            weight_path = Path(self.config.weights_dir) / weight_name
            if not weight_path.exists():
                print(f"❌ Weight file not found: {weight_path}")
                return False
            
            # Update config
            self.config.selected_weight = weight_name
            print(f"📝 Updated config selected_weight to: {weight_name}")
            
            # Load model if not already cached
            if weight_name not in self.models:
                print(f"🔄 Loading model: {weight_name} on {self.device}")
                model = YOLO(str(weight_path))
                model.to(self.device)
                
                # GPU optimizations
                if self.device == DeviceType.CUDA.value:
                    model.model.eval()
                    try:
                        if hasattr(torch.backends.cudnn, "enabled"):
                            torch.backends.cudnn.benchmark = True
                        if torch.cuda.is_available():
                            props = torch.cuda.get_device_properties(0)
                            total_memory_gb = props.total_memory / 1024**3
                            device_name = props.name
                            print(f"🚀 GPU: {device_name} | Total Memory: {total_memory_gb:.1f}GB")
                            torch.backends.cudnn.deterministic = False
                            torch.backends.cudnn.benchmark = True
                    except Exception as e:
                        print(f"⚠️ Could not enable CUDA optimizations: {e}")
                
                elif self.device == DeviceType.MPS.value:
                    model.model.eval()
                
                self.models[weight_name] = model
                print(f"✅ Model loaded successfully: {weight_name} on {self.device}")
            else:
                print(f"📦 Using cached model: {weight_name}")
            
            # Set as current model
            self.current_model = self.models[weight_name]
            print(f"✅ Switched to model: {weight_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error switching to model {weight_name}: {str(e)}")
            return False
    
    def is_loaded(self) -> bool:
        """Check if any model is loaded"""
        return self.current_model is not None
    
    def get_current_model_name(self) -> str:
        """Get the name of the currently loaded model"""
        return self.config.selected_weight
    
    def get_available_weights(self) -> List[Dict]:
        """Get list of available weights"""
        return self.available_weights
    
    def get_device_info(self) -> Dict:
        """Get information about the current device"""
        info = {"device": self.device, "device_name": "Unknown"}
        
        if self.device == DeviceType.MPS.value:
            info["device_name"] = "Apple Silicon MPS"
        elif self.device == DeviceType.CUDA.value:
            info["device_name"] = torch.cuda.get_device_name(0)
            info["memory_total"] = torch.cuda.get_device_properties(0).total_memory
            info["memory_allocated"] = torch.cuda.memory_allocated(0)
            info["memory_cached"] = torch.cuda.memory_reserved(0)
        else:
            info["device_name"] = "CPU"
        
        return info
    
    def detect_in_image(
        self, image_data: bytes, confidence_threshold: float = 0.5
    ) -> Tuple[List[DetectionResult], Optional[np.ndarray]]:
        """Detect logos in a single image"""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode image")
        
        # Run detection
        results = self.current_model(
            img,
            save=False,
            conf=confidence_threshold,
            device=self.device,
            verbose=False,
        )
        
        detections = []
        annotated_img = None
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.current_model.names[class_id]
                    
                    detections.append(
                        DetectionResult(
                            bbox=[float(x1), float(y1), float(x2), float(y2)],
                            confidence=confidence,
                            class_id=class_id,
                            class_name=class_name,
                        )
                    )
            
            # Get annotated image
            annotated_img = result.plot()
        
        return detections, annotated_img
    
    def detect_in_frame(
        self, frame: np.ndarray, confidence_threshold: float = 0.5
    ) -> Tuple[List[DetectionResult], Optional[np.ndarray]]:
        """Detect logos in a video frame"""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        # Run detection
        results = self.current_model(
            frame,
            save=False,
            conf=confidence_threshold,
            device=self.device,
            verbose=False,
            half=(self.device == DeviceType.CUDA.value),
            imgsz=640,
        )
        
        detections = []
        annotated_frame = None
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.current_model.names[class_id]
                    
                    detections.append(
                        DetectionResult(
                            bbox=[float(x1), float(y1), float(x2), float(y2)],
                            confidence=confidence,
                            class_id=class_id,
                            class_name=class_name,
                        )
                    )
            
            # Get annotated frame
            annotated_frame = result.plot()
        
        return detections, annotated_frame

