"""
Streamlit Dashboard - Edge vs Cloud AI Comparison
Interactive visualization of performance metrics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import os

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Edge vs Cloud AI Comparison",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .winner {
        color: #2ecc71;
        font-weight: bold;
    }
    .loser {
        color: #e74c3c;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# TITLE
# ============================================================================
st.markdown('<div class="main-header">🤖 Edge vs Cloud AI: Performance Comparison</div>', unsafe_allow_html=True)
st.markdown("---")

# ============================================================================
# LOAD DATA
# ============================================================================
@st.cache_data
def load_results():
    """Load edge and cloud results from CSV files"""
    edge_file = "results/edge_results.csv"
    cloud_file = "results/cloud_results.csv"
    
    edge_exists = os.path.exists(edge_file)
    cloud_exists = os.path.exists(cloud_file)
    
    edge_df = pd.read_csv(edge_file) if edge_exists else None
    cloud_df = pd.read_csv(cloud_file) if cloud_exists else None
    
    return edge_df, cloud_df, edge_exists, cloud_exists

edge_df, cloud_df, edge_exists, cloud_exists = load_results()

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.header("📊 Dashboard Controls")
    
    # Data status
    st.subheader("Data Status")
    if edge_exists:
        st.success(f"✓ Edge data loaded ({len(edge_df)} images)")
    else:
        st.error("✗ No edge data found")
        st.info("Run: `python edge_inference_improved.py`")
    
    if cloud_exists:
        st.success(f"✓ Cloud data loaded ({len(cloud_df)} images)")
    else:
        st.error("✗ No cloud data found")
        st.info("Run: `python test_cloud_improved.py`")
    
    st.markdown("---")
    
    # Settings
    st.subheader("⚙️ Settings")
    show_raw_data = st.checkbox("Show raw data tables", value=False)
    show_statistical_tests = st.checkbox("Show statistical tests", value=True)
    
    st.markdown("---")
    st.info("**Project:** Edge vs Cloud AI\n\n**Course:** CS-6905\n\n**Team:** Group 01")

# ============================================================================
# CHECK IF DATA EXISTS
# ============================================================================
if not edge_exists and not cloud_exists:
    st.error("### ❌ No data available")
    st.info("""
    **To generate data:**
    1. Run edge inference: `python edge_inference_improved.py`
    2. Start cloud API: `python api_improved.py`
    3. Run cloud tests: `python test_cloud_improved.py`
    4. Refresh this dashboard
    """)
    st.stop()

# ============================================================================
# EXECUTIVE SUMMARY
# ============================================================================
st.header("📈 Executive Summary")

if edge_exists and cloud_exists:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        edge_mean = edge_df['inference_time_ms'].mean()
        st.metric(
            "Edge Latency (avg)",
            f"{edge_mean:.2f} ms",
            delta=None
        )
    
    with col2:
        cloud_mean = cloud_df['total_latency_ms'].mean()
        st.metric(
            "Cloud Latency (avg)",
            f"{cloud_mean:.2f} ms",
            delta=f"{((cloud_mean - edge_mean) / edge_mean * 100):.1f}% slower" if cloud_mean > edge_mean else f"{((edge_mean - cloud_mean) / cloud_mean * 100):.1f}% faster"
        )
    
    with col3:
        edge_privacy = 0
        cloud_privacy = cloud_df['total_data_bytes'].sum() / (1024**2)
        st.metric(
            "Edge Data Exposed",
            "0 MB",
            delta="100% private",
            delta_color="normal"
        )
    
    with col4:
        st.metric(
            "Cloud Data Exposed",
            f"{cloud_privacy:.2f} MB",
            delta=f"{len(cloud_df)} images sent",
            delta_color="inverse"
        )

st.markdown("---")

# ============================================================================
# TAB NAVIGATION
# ============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Performance Comparison",
    "⏱️ Latency Analysis",
    "💻 Resource Usage",
    "🔒 Privacy & Network",
    "📈 Statistical Analysis"
])

# ============================================================================
# TAB 1: PERFORMANCE COMPARISON
# ============================================================================
with tab1:
    st.header("Performance Comparison Overview")
    
    if edge_exists and cloud_exists:
        # Create comparison DataFrame
        comparison_data = {
            'Metric': [
                'Avg Latency (ms)',
                'Median Latency (ms)',
                'Min Latency (ms)',
                'Max Latency (ms)',
                '95th Percentile (ms)',
                'Std Deviation (ms)'
            ],
            'Edge': [
                edge_df['inference_time_ms'].mean(),
                edge_df['inference_time_ms'].median(),
                edge_df['inference_time_ms'].min(),
                edge_df['inference_time_ms'].max(),
                edge_df['inference_time_ms'].quantile(0.95),
                edge_df['inference_time_ms'].std()
            ],
            'Cloud (Total)': [
                cloud_df['total_latency_ms'].mean(),
                cloud_df['total_latency_ms'].median(),
                cloud_df['total_latency_ms'].min(),
                cloud_df['total_latency_ms'].max(),
                cloud_df['total_latency_ms'].quantile(0.95),
                cloud_df['total_latency_ms'].std()
            ]
        }
        
        comp_df = pd.DataFrame(comparison_data)
        comp_df['Edge'] = comp_df['Edge'].round(2)
        comp_df['Cloud (Total)'] = comp_df['Cloud (Total)'].round(2)
        comp_df['Difference (%)'] = (((comp_df['Cloud (Total)'] - comp_df['Edge']) / comp_df['Edge']) * 100).round(1)
        
        # Add winner column
        comp_df['Winner'] = comp_df.apply(
            lambda row: '🏆 Edge' if row['Edge'] < row['Cloud (Total)'] else '🏆 Cloud',
            axis=1
        )
        
        st.dataframe(comp_df, use_container_width=True, height=250)
        
        # Bar chart comparison
        st.subheader("📊 Average Latency Comparison")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Edge',
            x=['Average Latency'],
            y=[edge_df['inference_time_ms'].mean()],
            marker_color='#2ecc71',
            text=[f"{edge_df['inference_time_ms'].mean():.2f} ms"],
            textposition='auto'
        ))
        fig.add_trace(go.Bar(
            name='Cloud (Total)',
            x=['Average Latency'],
            y=[cloud_df['total_latency_ms'].mean()],
            marker_color='#3498db',
            text=[f"{cloud_df['total_latency_ms'].mean():.2f} ms"],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Edge vs Cloud: Average Latency",
            yaxis_title="Latency (ms)",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 2: LATENCY ANALYSIS
# ============================================================================
with tab2:
    st.header("Detailed Latency Analysis")
    
    if edge_exists and cloud_exists:
        col1, col2 = st.columns(2)
        
        with col1:
            # Edge latency distribution
            st.subheader("Edge Latency Distribution")
            fig = px.histogram(
                edge_df,
                x='inference_time_ms',
                nbins=30,
                title="Edge Inference Time Distribution",
                labels={'inference_time_ms': 'Latency (ms)'},
                color_discrete_sequence=['#2ecc71']
            )
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Cloud latency distribution
            st.subheader("Cloud Latency Distribution")
            fig = px.histogram(
                cloud_df,
                x='total_latency_ms',
                nbins=30,
                title="Cloud Total Latency Distribution",
                labels={'total_latency_ms': 'Latency (ms)'},
                color_discrete_sequence=['#3498db']
            )
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        # Box plot comparison
        st.subheader("📦 Latency Box Plot Comparison")
        
        # Combine data for box plot
        combined = pd.DataFrame({
            'Latency (ms)': list(edge_df['inference_time_ms']) + list(cloud_df['total_latency_ms']),
            'Type': ['Edge'] * len(edge_df) + ['Cloud'] * len(cloud_df)
        })
        
        fig = px.box(
            combined,
            x='Type',
            y='Latency (ms)',
            color='Type',
            title="Latency Distribution Comparison",
            color_discrete_map={'Edge': '#2ecc71', 'Cloud': '#3498db'}
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
        
        # Cloud latency breakdown
        if 'network_latency_ms' in cloud_df.columns:
            st.subheader("☁️ Cloud Latency Breakdown")
            
            avg_server = cloud_df['server_inference_ms'].mean()
            avg_network = cloud_df['network_latency_ms'].mean()
            
            fig = go.Figure(data=[go.Pie(
                labels=['Server Inference', 'Network Round-Trip'],
                values=[avg_server, avg_network],
                hole=0.4,
                marker_colors=['#e74c3c', '#f39c12']
            )])
            fig.update_layout(
                title=f"Average Cloud Latency Breakdown ({cloud_df['total_latency_ms'].mean():.2f} ms total)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Avg Server Inference", f"{avg_server:.2f} ms", f"{avg_server/cloud_df['total_latency_ms'].mean()*100:.1f}% of total")
            with col2:
                st.metric("Avg Network Latency", f"{avg_network:.2f} ms", f"{avg_network/cloud_df['total_latency_ms'].mean()*100:.1f}% of total")

# ============================================================================
# TAB 3: RESOURCE USAGE
# ============================================================================
with tab3:
    st.header("Resource Usage Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if edge_exists:
            st.subheader("💻 Edge CPU Usage")
            fig = px.line(
                edge_df.reset_index(),
                x='index',
                y='cpu_percent',
                title="Edge CPU Usage Over Time",
                labels={'index': 'Test Number', 'cpu_percent': 'CPU Usage (%)'}
            )
            fig.add_hline(y=edge_df['cpu_percent'].mean(), line_dash="dash", 
                         annotation_text=f"Mean: {edge_df['cpu_percent'].mean():.2f}%")
            st.plotly_chart(fig, use_container_width=True)
            
            st.metric("Average CPU Usage", f"{edge_df['cpu_percent'].mean():.2f}%")
    
    with col2:
        if cloud_exists and 'server_cpu_percent' in cloud_df.columns:
            st.subheader("☁️ Cloud Server CPU Usage")
            fig = px.line(
                cloud_df.reset_index(),
                x='index',
                y='server_cpu_percent',
                title="Cloud Server CPU Usage Over Time",
                labels={'index': 'Test Number', 'server_cpu_percent': 'CPU Usage (%)'}
            )
            fig.add_hline(y=cloud_df['server_cpu_percent'].mean(), line_dash="dash",
                         annotation_text=f"Mean: {cloud_df['server_cpu_percent'].mean():.2f}%")
            st.plotly_chart(fig, use_container_width=True)
            
            st.metric("Average Server CPU Usage", f"{cloud_df['server_cpu_percent'].mean():.2f}%")
    
    # Memory usage
    st.subheader("🧠 Memory Usage Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if edge_exists:
            st.metric(
                "Edge Memory (avg)",
                f"{edge_df['memory_percent'].mean():.2f}%",
                f"{edge_df['memory_used_mb'].mean():.2f} MB used"
            )
    
    with col2:
        if cloud_exists and 'server_memory_percent' in cloud_df.columns:
            st.metric(
                "Cloud Server Memory (avg)",
                f"{cloud_df['server_memory_percent'].mean():.2f}%",
                f"{cloud_df['server_memory_mb'].mean():.2f} MB used"
            )

# ============================================================================
# TAB 4: PRIVACY & NETWORK
# ============================================================================
with tab4:
    st.header("Privacy & Network Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔒 Edge Privacy")
        st.success("✅ **COMPLETE PRIVACY**")
        st.write("- All processing happens locally")
        st.write("- Zero data transmitted over network")
        st.write("- No exposure to third parties")
        st.write("- Works fully offline")
        
        if edge_exists:
            st.metric("Total Data Exposed", "0 MB", "100% Private")
            st.metric("Images Sent to Cloud", "0", "Complete Local Processing")
    
    with col2:
        st.subheader("☁️ Cloud Privacy")
        st.warning("⚠️ **PRIVACY CONSIDERATIONS**")
        st.write("- Images transmitted over network")
        st.write("- Requires HTTPS for security")
        st.write("- Subject to cloud provider policies")
        st.write("- Requires internet connection")
        
        if cloud_exists:
            total_data_mb = cloud_df['total_data_bytes'].sum() / (1024**2)
            avg_data_kb = cloud_df['total_data_bytes'].mean() / 1024
            
            st.metric("Total Data Exposed", f"{total_data_mb:.2f} MB", f"{len(cloud_df)} images")
            st.metric("Avg Data per Image", f"{avg_data_kb:.2f} KB", "Request + Response")
    
    # Network bandwidth visualization
    if cloud_exists and 'total_data_bytes' in cloud_df.columns:
        st.subheader("📡 Network Bandwidth Usage")
        
        fig = px.line(
            cloud_df.reset_index(),
            x='index',
            y=cloud_df['total_data_bytes'] / 1024,  # Convert to KB
            title="Data Transferred per Request",
            labels={'index': 'Request Number', 'y': 'Data (KB)'}
        )
        fig.add_hline(y=cloud_df['total_data_bytes'].mean() / 1024, line_dash="dash",
                     annotation_text=f"Mean: {cloud_df['total_data_bytes'].mean() / 1024:.2f} KB")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 5: STATISTICAL ANALYSIS
# ============================================================================
with tab5:
    st.header("Statistical Analysis")
    
    if edge_exists and cloud_exists and show_statistical_tests:
        st.subheader("📊 Statistical Significance Testing")
        
        # T-test
        edge_latency = edge_df['inference_time_ms']
        cloud_latency = cloud_df['total_latency_ms']
        
        t_stat, p_value = stats.ttest_ind(edge_latency, cloud_latency)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("T-Statistic", f"{t_stat:.4f}")
        
        with col2:
            st.metric("P-Value", f"{p_value:.6f}")
        
        with col3:
            if p_value < 0.05:
                st.success("✅ Statistically Significant")
                st.write("(p < 0.05)")
            else:
                st.info("ℹ️ Not Significant")
                st.write("(p ≥ 0.05)")
        
        st.info(f"""
        **Interpretation:**
        - The difference between edge ({edge_latency.mean():.2f}ms) and cloud ({cloud_latency.mean():.2f}ms) latency is {'**statistically significant**' if p_value < 0.05 else '**not statistically significant**'}.
        - This means the observed difference is {'**NOT due to random chance**' if p_value < 0.05 else '**could be due to random variation**'}.
        - Confidence level: {(1 - p_value) * 100:.2f}%
        """)
        
        # Summary statistics
        st.subheader("📈 Detailed Statistics")
        
        stats_df = pd.DataFrame({
            'Statistic': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', '25th %ile', '75th %ile', '95th %ile'],
            'Edge (ms)': [
                len(edge_latency),
                edge_latency.mean(),
                edge_latency.median(),
                edge_latency.std(),
                edge_latency.min(),
                edge_latency.max(),
                edge_latency.quantile(0.25),
                edge_latency.quantile(0.75),
                edge_latency.quantile(0.95)
            ],
            'Cloud (ms)': [
                len(cloud_latency),
                cloud_latency.mean(),
                cloud_latency.median(),
                cloud_latency.std(),
                cloud_latency.min(),
                cloud_latency.max(),
                cloud_latency.quantile(0.25),
                cloud_latency.quantile(0.75),
                cloud_latency.quantile(0.95)
            ]
        })
        
        stats_df['Edge (ms)'] = stats_df['Edge (ms)'].round(2)
        stats_df['Cloud (ms)'] = stats_df['Cloud (ms)'].round(2)
        
        st.dataframe(stats_df, use_container_width=True)

# ============================================================================
# RAW DATA SECTION
# ============================================================================
if show_raw_data:
    st.markdown("---")
    st.header("📋 Raw Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if edge_exists:
            st.subheader("Edge Results")
            st.dataframe(edge_df, use_container_width=True, height=400)
            st.download_button(
                "📥 Download Edge CSV",
                edge_df.to_csv(index=False),
                "edge_results.csv",
                "text/csv"
            )
    
    with col2:
        if cloud_exists:
            st.subheader("Cloud Results")
            st.dataframe(cloud_df, use_container_width=True, height=400)
            st.download_button(
                "📥 Download Cloud CSV",
                cloud_df.to_csv(index=False),
                "cloud_results.csv",
                "text/csv"
            )

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p><strong>Edge vs Cloud AI: Comparative Performance Analysis</strong></p>
    <p>CS-6905 Emerging Trends in AI | University of New Brunswick | March 2025</p>
    <p>Team: Muhammad Wasiq Malik & Ashhad Ahmed Memon</p>
</div>
""", unsafe_allow_html=True)
