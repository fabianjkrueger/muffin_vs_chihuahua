import requests
from pathlib import Path
from PIL import Image
from typing import Dict, Union
from IPython.display import Image as IPythonImage, display

class ImageClassifierClient:
    """Client for the image classification API"""
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.predict_url = f"{base_url}/predict"
    
    def predict(self, image_path: Union[str, Path]) -> Dict:
        """Query the model with an image and display it.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Model response containing prediction and probability.
        """
        # display the image
        display(IPythonImage(image_path))
        
        # prepare the image file and make prediction
        with open(image_path, "rb") as f:
            files = {"file": (str(image_path).split("/")[-1], f, "image/jpeg")}
            response = requests.post(self.predict_url, files=files)
        
        # interpret response
        result = response.json()
        if result["class_name"] == "positive":
            print("This is a muffin!")
        elif result["class_name"] == "negative":
            print("This is a chihuahua!")
        print(f"Probability: {result['probability']*100:.4f}%")
        
        return result
    
    def predict_user_image(self, image_path: Union[str, Path]) -> Dict:
        """Query the model with a user image, reformatting if necessary.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Model response containing prediction and probability.
        """
        # directory for reformatted images
        user_images_dir = Path("data/user_images")
        
        # check image format
        image = Image.open(image_path)
        correct_format = image.size == (224, 224) and image.mode == "RGB"
        
        if not correct_format:
            # reformat image
            reformatted_image = image.resize((224, 224))
            if reformatted_image.mode != "RGB":
                reformatted_image = reformatted_image.convert("RGB")
            
            # save reformatted image
            user_images_dir.mkdir(parents=True, exist_ok=True)
            filename = Path(image_path).name
            save_path = user_images_dir / filename
            reformatted_image.save(save_path)
            
            return self.predict(save_path)
        
        return self.predict(image_path)
