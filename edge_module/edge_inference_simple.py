"""
Edge Inference - Using Full TensorFlow Model (No TFLite conversion needed)
"""

import time
import numpy as np
import psutil
import csv
import os
import hashlib
from pathlib import Path
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as keras_image

# Configuration
IMAGE_DIR = "dataset"
RESULTS_FILE = "results/edge_results.csv"

# Load model
print("Loading MobileNetV2 model...")
model = MobileNetV2(weights='imagenet')
print("✓ Model loaded successfully")

def get_image_hash(image_path):
    """Calculate MD5 hash of image"""
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def preprocess_image(image_path):
    """Preprocess image for MobileNetV2"""
    img = keras_image.load_img(image_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def measure_inference(image_path):
    """Measure inference with comprehensive metrics"""
    
    img_hash = get_image_hash(image_path)
    
    # Measure resources BEFORE
    cpu_before = psutil.cpu_percent(interval=0.1)
    mem_before = psutil.virtual_memory()
    
    # Preprocess
    img_array = preprocess_image(image_path)
    
    # Warm-up (first prediction is slower)
    _ = model.predict(img_array, verbose=0)
    
    # Actual measurement
    start_time = time.perf_counter()
    predictions = model.predict(img_array, verbose=0)
    end_time = time.perf_counter()
    
    # Measure resources AFTER
    cpu_after = psutil.cpu_percent(interval=0.1)
    mem_after = psutil.virtual_memory()
    
    # Calculate metrics
    inference_time_ms = (end_time - start_time) * 1000
    cpu_usage = max(cpu_after, cpu_before)
    memory_used_mb = (mem_after.used - mem_before.used) / (1024**2)
    memory_percent = mem_after.percent
    
    # Decode predictions
    decoded = decode_predictions(predictions, top=3)[0]
    
    return {
        'image_path': image_path,
        'image_hash': img_hash,
        'inference_time_ms': inference_time_ms,
        'cpu_percent': cpu_usage,
        'memory_used_mb': memory_used_mb,
        'memory_percent': memory_percent,
        'top_class': decoded[0][1],
        'top_confidence': float(decoded[0][2]),
        'network_usage_bytes': 0
    }

def test_multiple_images(image_dir, num_images=100):
    """Test model on multiple images"""
    
    image_files = list(Path(image_dir).glob("*.JPEG"))[:num_images]
    
    if len(image_files) == 0:
        print(f"❌ No images found in {image_dir}")
        return []
    
    print(f"\n{'='*60}")
    print(f"Testing {len(image_files)} images on EDGE")
    print(f"{'='*60}\n")
    
    results = []
    os.makedirs("results", exist_ok=True)
    
    # Write CSV header
    with open(RESULTS_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'image_path', 'image_hash', 'inference_time_ms', 
            'cpu_percent', 'memory_used_mb', 'memory_percent',
            'top_class', 'top_confidence', 'network_usage_bytes'
        ])
    
    # Test each image
    for i, img_path in enumerate(image_files, 1):
        try:
            print(f"[{i}/{len(image_files)}] Testing {img_path.name}...", end=' ')
            
            result = measure_inference(str(img_path))
            results.append(result)
            
            # Save to CSV
            with open(RESULTS_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    result['image_path'],
                    result['image_hash'],
                    f"{result['inference_time_ms']:.2f}",
                    f"{result['cpu_percent']:.2f}",
                    f"{result['memory_used_mb']:.2f}",
                    f"{result['memory_percent']:.2f}",
                    result['top_class'],
                    f"{result['top_confidence']:.4f}",
                    result['network_usage_bytes']
                ])
            
            print(f"✓ {result['inference_time_ms']:.2f}ms | {result['top_class']} ({result['top_confidence']:.2%})")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return results

def print_statistics(results):
    """Print statistical summary"""
    if not results:
        return
    
    inference_times = [r['inference_time_ms'] for r in results]
    cpu_usage = [r['cpu_percent'] for r in results]
    
    print(f"\n{'='*60}")
    print("EDGE INFERENCE STATISTICS")
    print(f"{'='*60}")
    print(f"Total images tested: {len(results)}")
    print(f"\nInference Time (ms):")
    print(f"  Mean:      {np.mean(inference_times):.2f}")
    print(f"  Median:    {np.median(inference_times):.2f}")
    print(f"  Std Dev:   {np.std(inference_times):.2f}")
    print(f"  Min:       {np.min(inference_times):.2f}")
    print(f"  Max:       {np.max(inference_times):.2f}")
    print(f"  95th %%ile: {np.percentile(inference_times, 95):.2f}")
    print(f"\nCPU Usage (%%): {np.mean(cpu_usage):.2f}")
    print(f"\nResults saved to: {RESULTS_FILE}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    results = test_multiple_images(IMAGE_DIR, num_images=100)
    print_statistics(results)
    print("\n✅ Edge testing complete!")
