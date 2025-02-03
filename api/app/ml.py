import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
from pathlib import Path
import io

class ImageClassifier:
    def __init__(self, model_path: str, device: str = None):
        """Initialize the classifier with model and preprocessing"""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # initialize model architecture (same as in training)
        self.model = models.resnet18(weights=None)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)
        
        # load trained weights
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()
        
        # use same transforms as in training
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # class mapping (adjust based on your classes)
        self.classes = ["negative", "positive"]
    
    def predict(self, image_bytes: bytes) -> dict:
        """Predict class for a given image"""
        # convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # preprocess
        image_tensor = self.transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        # inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # get predictions
        predicted_class = int(torch.argmax(probabilities))
        
        # prepare response
        all_probs = {
            self.classes[i]: float(prob)
            for i, prob in enumerate(probabilities)
        }
        
        return {
            "class_name": self.classes[predicted_class],
            "probability": float(probabilities[predicted_class]),
            "all_probabilities": all_probs
        }
