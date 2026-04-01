"""
Compare Edge vs Cloud (Local) vs Cloud (AWS) Results
"""

import pandas as pd
import numpy as np
from scipy import stats

# Load all results
print("\n" + "="*70)
print("EDGE vs CLOUD (Local) vs CLOUD (AWS) - COMPREHENSIVE COMPARISON")
print("="*70 + "\n")

try:
    edge_df = pd.read_csv("results/edge_results.csv")
    print(f"✓ Loaded edge results: {len(edge_df)} images")
except:
    print("❌ Edge results not found")
    edge_df = None

try:
    local_df = pd.read_csv("results/cloud_results_local.csv")
    print(f"✓ Loaded cloud (local) results: {len(local_df)} images")
except:
    print("⚠️  Cloud local results not found (optional)")
    local_df = None

try:
    aws_df = pd.read_csv("results/cloud_results.csv")
    print(f"✓ Loaded cloud (AWS) results: {len(aws_df)} images")
except:
    print("❌ Cloud AWS results not found")
    aws_df = None

print("\n" + "="*70)
print("LATENCY COMPARISON")
print("="*70)

# Create comparison table
data = []

if edge_df is not None:
    edge_latency = edge_df['inference_time_ms']
    data.append({
        'Deployment': 'Edge (Local)',
        'Mean (ms)': edge_latency.mean(),
        'Median (ms)': edge_latency.median(),
        'Std Dev (ms)': edge_latency.std(),
        'Min (ms)': edge_latency.min(),
        'Max (ms)': edge_latency.max(),
        '95th %ile (ms)': edge_latency.quantile(0.95),
        'Network (ms)': 0
    })

if local_df is not None:
    local_latency = local_df['total_latency_ms']
    local_network = local_df['network_latency_ms'] if 'network_latency_ms' in local_df.columns else 0
    data.append({
        'Deployment': 'Cloud (Local)',
        'Mean (ms)': local_latency.mean(),
        'Median (ms)': local_latency.median(),
        'Std Dev (ms)': local_latency.std(),
        'Min (ms)': local_latency.min(),
        'Max (ms)': local_latency.max(),
        '95th %ile (ms)': local_latency.quantile(0.95),
        'Network (ms)': local_network.mean() if hasattr(local_network, 'mean') else 0
    })

if aws_df is not None:
    aws_latency = aws_df['total_latency_ms']
    aws_network = aws_df['network_latency_ms'] if 'network_latency_ms' in aws_df.columns else 0
    data.append({
        'Deployment': 'Cloud (AWS)',
        'Mean (ms)': aws_latency.mean(),
        'Median (ms)': aws_latency.median(),
        'Std Dev (ms)': aws_latency.std(),
        'Min (ms)': aws_latency.min(),
        'Max (ms)': aws_latency.max(),
        '95th %ile (ms)': aws_latency.quantile(0.95),
        'Network (ms)': aws_network.mean() if hasattr(aws_network, 'mean') else 0
    })

comparison_df = pd.DataFrame(data)

# Format numbers
for col in comparison_df.columns:
    if col != 'Deployment':
        comparison_df[col] = comparison_df[col].round(2)

print("\n" + comparison_df.to_string(index=False))

# Statistical significance testing
print("\n" + "="*70)
print("STATISTICAL SIGNIFICANCE TESTS")
print("="*70 + "\n")

if edge_df is not None and aws_df is not None:
    # T-test: Edge vs AWS
    t_stat, p_value = stats.ttest_ind(edge_df['inference_time_ms'], aws_df['total_latency_ms'])
    print(f"Edge vs Cloud (AWS):")
    print(f"  T-statistic: {t_stat:.4f}")
    print(f"  P-value: {p_value:.6f}")
    if p_value < 0.05:
        print(f"  ✅ STATISTICALLY SIGNIFICANT (p < 0.05)")
        print(f"  Conclusion: Difference is real, not due to chance")
    else:
        print(f"  ⚠️  Not significant (p ≥ 0.05)")
    
    # Effect size (Cohen's d)
    mean_diff = edge_df['inference_time_ms'].mean() - aws_df['total_latency_ms'].mean()
    pooled_std = np.sqrt((edge_df['inference_time_ms'].std()**2 + aws_df['total_latency_ms'].std()**2) / 2)
    cohens_d = mean_diff / pooled_std
    print(f"  Effect size (Cohen's d): {cohens_d:.4f}")
    if abs(cohens_d) > 0.8:
        print(f"  Effect: LARGE (practical significance)")
    elif abs(cohens_d) > 0.5:
        print(f"  Effect: MEDIUM")
    else:
        print(f"  Effect: SMALL")

if local_df is not None and aws_df is not None:
    print(f"\nCloud Local vs Cloud AWS:")
    t_stat, p_value = stats.ttest_ind(local_df['total_latency_ms'], aws_df['total_latency_ms'])
    print(f"  T-statistic: {t_stat:.4f}")
    print(f"  P-value: {p_value:.6f}")
    if p_value < 0.05:
        print(f"  ✅ STATISTICALLY SIGNIFICANT (p < 0.05)")
    else:
        print(f"  ⚠️  Not significant (p ≥ 0.05)")

# Privacy & Network Analysis
print("\n" + "="*70)
print("PRIVACY & NETWORK ANALYSIS")
print("="*70 + "\n")

privacy_data = []

if edge_df is not None:
    privacy_data.append({
        'Deployment': 'Edge (Local)',
        'Data Transferred (MB)': 0,
        'Images Exposed': 0,
        'Privacy Score': '100% Private'
    })

if local_df is not None and 'total_data_bytes' in local_df.columns:
    total_mb = local_df['total_data_bytes'].sum() / (1024**2)
    privacy_data.append({
        'Deployment': 'Cloud (Local)',
        'Data Transferred (MB)': round(total_mb, 2),
        'Images Exposed': len(local_df),
        'Privacy Score': 'Local Network Only'
    })

if aws_df is not None and 'total_data_bytes' in aws_df.columns:
    total_mb = aws_df['total_data_bytes'].sum() / (1024**2)
    privacy_data.append({
        'Deployment': 'Cloud (AWS)',
        'Data Transferred (MB)': round(total_mb, 2),
        'Images Exposed': len(aws_df),
        'Privacy Score': '0% Private (Internet)'
    })

privacy_df = pd.DataFrame(privacy_data)
print(privacy_df.to_string(index=False))

# Key Findings
print("\n" + "="*70)
print("KEY FINDINGS & RECOMMENDATIONS")
print("="*70 + "\n")

if edge_df is not None and aws_df is not None:
    speedup = aws_df['total_latency_ms'].mean() / edge_df['inference_time_ms'].mean()
    print(f"📊 PERFORMANCE:")
    print(f"  • Edge is {speedup:.1f}x FASTER than Cloud (AWS)")
    print(f"  • Edge: {edge_df['inference_time_ms'].mean():.2f}ms avg")
    print(f"  • AWS:  {aws_df['total_latency_ms'].mean():.2f}ms avg")
    
    if 'network_latency_ms' in aws_df.columns:
        network_pct = (aws_df['network_latency_ms'].mean() / aws_df['total_latency_ms'].mean()) * 100
        print(f"  • Network accounts for {network_pct:.1f}% of cloud latency")

print(f"\n🔒 PRIVACY:")
print(f"  • Edge: Complete privacy (0 MB transferred)")
if aws_df is not None and 'total_data_bytes' in aws_df.columns:
    print(f"  • AWS:  {aws_df['total_data_bytes'].sum() / (1024**2):.2f} MB exposed to internet")

print(f"\n✅ RECOMMENDATIONS:")
print(f"  • Use EDGE for: Real-time apps, privacy-critical, offline scenarios")
print(f"  • Use CLOUD for: Batch processing, high scalability, powerful GPUs")

print("\n" + "="*70)
print("Analysis complete! Use these findings in your report.")
print("="*70 + "\n")
