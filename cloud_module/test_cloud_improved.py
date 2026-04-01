"""
Cloud Testing Client - Improved Version
Tests cloud API with 100 images and comprehensive metrics tracking
"""

import requests
import time
import csv
import os
import hashlib
from pathlib import Path
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================
# API_URL = "http://127.0.0.1:8000/predict"  # Local testing
API_URL = "http://99.79.127.147:8000/predict"  # AWS deployment

IMAGE_DIR = "dataset"
# RESULTS_FILE = "results/cloud_results.csv" #local
RESULTS_FILE = "results/cloud_results_aws.csv" #aws

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_image_hash(image_path):
    """Calculate MD5 hash of image"""
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def get_image_size(image_path):
    """Get image file size in bytes"""
    return os.path.getsize(image_path)


# ============================================================================
# CLOUD INFERENCE FUNCTION
# ============================================================================
def test_cloud_inference(image_path):
    """
    Test cloud inference with comprehensive metrics
    
    Measures:
    - Total latency (including network round-trip)
    - Server inference time (from API response)
    - Network latency (total - server inference)
    - Data transferred (request + response size)
    """
    
    # Get image info
    img_hash = get_image_hash(image_path)
    img_size = get_image_size(image_path)
    
    # Prepare file for upload
    with open(image_path, 'rb') as f:
        files = {"file": (Path(image_path).name, f, "image/jpeg")}
        
        # Measure TOTAL latency (client perspective)
        start_time = time.perf_counter()
        
        try:
            response = requests.post(API_URL, files=files, timeout=30)
            response.raise_for_status()  # Raise error for bad status codes
            
            end_time = time.perf_counter()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            return None
    
    # Calculate total latency
    total_latency_ms = (end_time - start_time) * 1000
    
    # Parse response
    data = response.json()
    
    # Extract metrics from server
    server_inference_ms = data['metrics']['server_inference_time_ms']
    server_cpu = data['metrics']['server_cpu_percent']
    server_memory_mb = data['metrics']['server_memory_used_mb']
    server_memory_percent = data['metrics']['server_memory_percent']
    
    # Calculate network latency
    network_latency_ms = total_latency_ms - server_inference_ms
    
    # Calculate data transferred
    request_size_bytes = img_size
    response_size_bytes = len(response.content)
    total_data_bytes = request_size_bytes + response_size_bytes
    
    # Extract predictions
    predictions = data['predictions']
    top_prediction = predictions[0]
    
    # Verify hash matches
    server_hash = data['verification']['image_hash']
    hash_match = (img_hash == server_hash)
    
    # Return all metrics
    return {
        'image_path': image_path,
        'image_hash': img_hash,
        'hash_verified': hash_match,
        'total_latency_ms': total_latency_ms,
        'server_inference_ms': server_inference_ms,
        'network_latency_ms': network_latency_ms,
        'server_cpu_percent': server_cpu,
        'server_memory_mb': server_memory_mb,
        'server_memory_percent': server_memory_percent,
        'request_size_bytes': request_size_bytes,
        'response_size_bytes': response_size_bytes,
        'total_data_bytes': total_data_bytes,
        'top_class': top_prediction['label'],
        'top_confidence': top_prediction['confidence'],
        'predictions': predictions
    }


# ============================================================================
# BATCH TESTING WITH 100 IMAGES
# ============================================================================
def test_multiple_images(image_dir, num_images=100):
    """Test cloud API with multiple images"""
    
    # Get list of images
    image_files = list(Path(image_dir).glob("*.JPEG"))[:num_images]
    
    if len(image_files) == 0:
        print(f"❌ No images found in {image_dir}")
        print("   Please add test images to the dataset/test_images folder")
        return []
    
    print(f"\n{'='*60}")
    print(f"Testing {len(image_files)} images on CLOUD (FastAPI)")
    print(f"API URL: {API_URL}")
    print(f"{'='*60}\n")
    
    # Check if server is running
    try:
        health = requests.get(API_URL.replace('/predict', '/health'), timeout=5)
        print(f"✓ Cloud server is running")
        print(f"  {health.json()}\n")
    except requests.exceptions.RequestException:
        print(f"❌ Cannot connect to cloud server at {API_URL}")
        print(f"   Make sure the API server is running:")
        print(f"   python api_improved.py")
        return []
    
    results = []
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Write CSV header
    with open(RESULTS_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'image_path', 'image_hash', 'hash_verified',
            'total_latency_ms', 'server_inference_ms', 'network_latency_ms',
            'server_cpu_percent', 'server_memory_mb', 'server_memory_percent',
            'request_size_bytes', 'response_size_bytes', 'total_data_bytes',
            'top_class', 'top_confidence'
        ])
    
    # Test each image
    for i, img_path in enumerate(image_files, 1):
        try:
            print(f"[{i}/{len(image_files)}] Testing {img_path.name}...", end=' ')
            
            result = test_cloud_inference(str(img_path))
            
            if result is None:
                print("❌ Failed")
                continue
            
            results.append(result)
            
            # Save to CSV
            with open(RESULTS_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    result['image_path'],
                    result['image_hash'],
                    result['hash_verified'],
                    f"{result['total_latency_ms']:.2f}",
                    f"{result['server_inference_ms']:.2f}",
                    f"{result['network_latency_ms']:.2f}",
                    f"{result['server_cpu_percent']:.2f}",
                    f"{result['server_memory_mb']:.2f}",
                    f"{result['server_memory_percent']:.2f}",
                    result['request_size_bytes'],
                    result['response_size_bytes'],
                    result['total_data_bytes'],
                    result['top_class'],
                    f"{result['top_confidence']:.4f}"
                ])
            
            print(f"✓ {result['total_latency_ms']:.2f}ms | {result['top_class']} ({result['top_confidence']:.2%})")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return results


# ============================================================================
# STATISTICAL SUMMARY
# ============================================================================
def print_statistics(results):
    """Print statistical summary of cloud results"""
    if not results:
        return
    
    total_latency = [r['total_latency_ms'] for r in results]
    server_inference = [r['server_inference_ms'] for r in results]
    network_latency = [r['network_latency_ms'] for r in results]
    data_transferred = [r['total_data_bytes'] for r in results]
    
    print(f"\n{'='*60}")
    print("CLOUD INFERENCE STATISTICS")
    print(f"{'='*60}")
    print(f"Total images tested: {len(results)}")
    
    print(f"\nTotal Latency (ms) - includes network:")
    print(f"  Mean:      {np.mean(total_latency):.2f}")
    print(f"  Median:    {np.median(total_latency):.2f}")
    print(f"  Std Dev:   {np.std(total_latency):.2f}")
    print(f"  Min:       {np.min(total_latency):.2f}")
    print(f"  Max:       {np.max(total_latency):.2f}")
    print(f"  95th %ile: {np.percentile(total_latency, 95):.2f}")
    
    print(f"\nServer Inference Time (ms) - server only:")
    print(f"  Mean:      {np.mean(server_inference):.2f}")
    print(f"  Median:    {np.median(server_inference):.2f}")
    
    print(f"\nNetwork Latency (ms) - round-trip:")
    print(f"  Mean:      {np.mean(network_latency):.2f}")
    print(f"  Median:    {np.median(network_latency):.2f}")
    print(f"  % of Total: {(np.mean(network_latency) / np.mean(total_latency) * 100):.1f}%")
    
    print(f"\nData Transferred:")
    print(f"  Per image (avg): {np.mean(data_transferred) / 1024:.2f} KB")
    print(f"  Total: {np.sum(data_transferred) / (1024**2):.2f} MB")
    
    print(f"\nHash Verification:")
    verified = sum(1 for r in results if r['hash_verified'])
    print(f"  Verified: {verified}/{len(results)} ({verified/len(results)*100:.1f}%)")
    
    print(f"\nResults saved to: {RESULTS_FILE}")
    print(f"{'='*60}\n")


# ============================================================================
# PRIVACY ASSESSMENT
# ============================================================================
def assess_privacy(results):
    """Assess privacy implications of cloud deployment"""
    if not results:
        return
    
    print(f"\n{'='*60}")
    print("PRIVACY ASSESSMENT")
    print(f"{'='*60}")
    
    total_data_mb = sum(r['total_data_bytes'] for r in results) / (1024**2)
    num_images = len(results)
    
    print(f"Images sent to cloud: {num_images}")
    print(f"Total data exposed: {total_data_mb:.2f} MB")
    print(f"Average per image: {total_data_mb / num_images:.2f} MB")
    
    print(f"\n⚠️  PRIVACY RISK:")
    print(f"  - All {num_images} images were transmitted over the network")
    print(f"  - Images are exposed to cloud provider")
    print(f"  - Requires HTTPS/encryption for secure transmission")
    print(f"  - Data subject to cloud provider's data retention policies")
    
    print(f"\n✅ EDGE COMPARISON:")
    print(f"  - 0 images transmitted")
    print(f"  - 0 MB data exposed")
    print(f"  - Complete local privacy")
    
    print(f"{'='*60}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    
    # Test with 100 images
    results = test_multiple_images(IMAGE_DIR, num_images=100)
    
    # Print statistics
    print_statistics(results)
    
    # Assess privacy
    assess_privacy(results)
    
    print("\n✅ Cloud testing complete!")
    print(f"   Results saved to: {RESULTS_FILE}")
