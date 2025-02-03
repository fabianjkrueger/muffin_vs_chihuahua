from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from pathlib import Path
from .ml import ImageClassifier
from .models import PredictionResponse

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# initialize FastAPI app
app = FastAPI(
    title="Image Classification API",
    description="API for classifying images using PyTorch model",
    version="1.0.0"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# model path
MODEL_PATH = Path(__file__).parent.parent / "models" / "final_model.pt"

# initialize model
@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    try:
        app.state.model = ImageClassifier(str(MODEL_PATH))
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise RuntimeError("Failed to load model")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict class for uploaded image
    """
    if file is None:
        raise HTTPException(status_code=400, detail="No file provided")
        
    # validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        # read image
        contents = await file.read()
        
        # get prediction
        prediction = app.state.model.predict(contents)
        
        return PredictionResponse(**prediction)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error processing image"
        )
