"""
Cloud API - Improved Version with Metrics Tracking
FastAPI server that tracks inference time, CPU, memory on server side
"""

from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import time
import psutil
import io
import hashlib

# ============================================================================
# CONFIGURATION
# ============================================================================
app = FastAPI(title="Edge vs Cloud AI - Cloud Inference API")

# Load model globally (loaded once when server starts)
print("Loading MobileNetV2 model...")
model = tf.keras.applications.MobileNetV2(weights="imagenet")
print("✓ Model loaded successfully")

# Keep track of total requests
request_count = 0
total_data_received = 0


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_image_hash(image_bytes):
    """Calculate MD5 hash for verification"""
    return hashlib.md5(image_bytes).hexdigest()

def preprocess_image_data(image_bytes):
    """Preprocess image from bytes"""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


# ============================================================================
# MAIN PREDICTION ENDPOINT
# ============================================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Cloud inference endpoint with comprehensive metrics
    
    Returns:
        - predictions: Top 3 predictions with confidence
        - metrics: Server-side performance metrics
        - verification: Image hash for testing verification
    """
    global request_count, total_data_received
    request_count += 1
    
    # Read image bytes
    image_bytes = await file.read()
    image_size_bytes = len(image_bytes)
    total_data_received += image_size_bytes
    
    # Calculate hash for verification
    img_hash = get_image_hash(image_bytes)
    
    # Measure resources BEFORE inference
    cpu_before = psutil.cpu_percent(interval=0.1)
    mem_before = psutil.virtual_memory()
    
    # Measure inference time
    start_time = time.perf_counter()
    
    # Preprocess and predict
    img_array = preprocess_image_data(image_bytes)
    predictions = model.predict(img_array)
    
    end_time = time.perf_counter()
    
    # Measure resources AFTER inference
    cpu_after = psutil.cpu_percent(interval=0.1)
    mem_after = psutil.virtual_memory()
    
    # Calculate metrics
    inference_time_ms = (end_time - start_time) * 1000
    cpu_usage = max(cpu_after, cpu_before)
    memory_used_mb = (mem_after.used - mem_before.used) / (1024**2)
    memory_percent = mem_after.percent
    
    # Decode predictions
    decoded = decode_predictions(predictions, top=3)[0]
    
    results = []
    for i in decoded:
        results.append({
            "class_id": i[0],
            "label": i[1],
            "confidence": float(i[2])
        })
    
    # Return comprehensive response
    return {
        "predictions": results,
        "metrics": {
            "server_inference_time_ms": round(inference_time_ms, 2),
            "server_cpu_percent": round(cpu_usage, 2),
            "server_memory_used_mb": round(memory_used_mb, 2),
            "server_memory_percent": round(memory_percent, 2),
            "image_size_bytes": image_size_bytes,
            "image_size_kb": round(image_size_bytes / 1024, 2)
        },
        "verification": {
            "image_hash": img_hash,
            "request_number": request_count
        }
    }


# ============================================================================
# HEALTH CHECK ENDPOINT
# ============================================================================
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "MobileNetV2",
        "total_requests": request_count,
        "total_data_received_mb": round(total_data_received / (1024**2), 2)
    }


# ============================================================================
# STATISTICS ENDPOINT
# ============================================================================
@app.get("/stats")
async def get_stats():
    """Get server statistics"""
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    
    return {
        "server_stats": {
            "total_requests_served": request_count,
            "total_data_received_mb": round(total_data_received / (1024**2), 2),
            "current_cpu_percent": cpu,
            "current_memory_percent": mem.percent,
            "available_memory_mb": round(mem.available / (1024**2), 2)
        }
    }


# ============================================================================
# ROOT ENDPOINT
# ============================================================================
@app.get("/")
async def root():
    """API information"""
    return {
        "message": "Edge vs Cloud AI - Cloud Inference API",
        "model": "MobileNetV2",
        "endpoints": {
            "/predict": "POST - Upload image for classification",
            "/health": "GET - Health check",
            "/stats": "GET - Server statistics"
        },
        "status": "running"
    }


# ============================================================================
# RUN SERVER
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("Starting Cloud Inference API Server")
    print("="*60)
    print("API will be available at: http://127.0.0.1:8000")
    print("API docs at: http://127.0.0.1:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
