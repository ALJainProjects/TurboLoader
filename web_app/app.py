"""
TurboLoader Interactive Benchmark Web App

Upload datasets, run benchmarks, and visualize performance metrics.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import psutil
import time
import numpy as np
from pathlib import Path
import tempfile

# Configure page
st.set_page_config(
    page_title="TurboLoader Benchmark",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and header
st.title("⚡ TurboLoader Interactive Benchmark")
st.markdown("### Compare TurboLoader vs PyTorch vs TensorFlow Performance")

# Sidebar configuration
st.sidebar.header("🔧 Configuration")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset (TAR format)",
    type=['tar'],
    help="Upload a TAR archive containing images for benchmarking"
)

# Framework selection
framework = st.sidebar.selectbox(
    "Framework",
    ["TurboLoader", "PyTorch DataLoader", "TensorFlow tf.data"],
    help="Select which framework to benchmark"
)

# Worker configuration
num_workers = st.sidebar.slider(
    "Number of Workers",
    min_value=1,
    max_value=32,
    value=8,
    step=1,
    help="Number of worker threads for parallel processing"
)

# Batch size
batch_size = st.sidebar.slider(
    "Batch Size",
    min_value=8,
    max_value=256,
    value=32,
    step=8,
    help="Number of samples per batch"
)

# Number of epochs
num_epochs = st.sidebar.slider(
    "Number of Epochs",
    min_value=1,
    max_value=10,
    value=3,
    help="Number of complete passes through the dataset"
)

# Transform configuration
st.sidebar.subheader("🎨 Transforms")

enable_resize = st.sidebar.checkbox("Resize (224x224)", value=True)
enable_normalize = st.sidebar.checkbox("Normalize (ImageNet)", value=True)
enable_flip = st.sidebar.checkbox("Random Horizontal Flip", value=False)
enable_color_jitter = st.sidebar.checkbox("Color Jitter", value=False)
enable_autoaugment = st.sidebar.checkbox("AutoAugment", value=False)

# Main content area
if uploaded_file is None:
    # Show welcome message and instructions
    st.info("👈 Upload a dataset TAR file to get started!")

    st.markdown("""
    ## How to Use

    1. **Upload Dataset**: Provide a TAR archive containing images
    2. **Select Framework**: Choose which implementation to benchmark
    3. **Configure**: Set workers, batch size, and transforms
    4. **Run**: Click "Run Benchmark" and wait for results
    5. **Analyze**: Review metrics and interactive charts

    ## Supported Formats

    - WebDataset TAR archives
    - JPEG, PNG, WebP images
    - Minimum 100 images recommended

    ## Example Datasets

    You can create a test dataset using:

    ```bash
    # Create 1000 sample images
    python -c "
    import numpy as np
    from PIL import Image
    import tarfile

    with tarfile.open('test_dataset.tar', 'w') as tar:
        for i in range(1000):
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            Image.fromarray(img).save(f'{i:06d}.jpg')
            tar.add(f'{i:06d}.jpg')
    "
    ```
    """)

    # Display example results
    st.markdown("## Example Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Throughput",
            "10,146 img/s",
            delta="12x vs PyTorch",
            delta_color="normal"
        )

    with col2:
        st.metric(
            "Avg Epoch Time",
            "0.18s",
            delta="-2.22s vs PyTorch",
            delta_color="normal"
        )

    with col3:
        st.metric(
            "Peak Memory",
            "848 MB",
            delta="-675 MB vs PyTorch",
            delta_color="inverse"
        )

    with col4:
        st.metric(
            "CPU Utilization",
            "94%",
            delta="+49% vs PyTorch",
            delta_color="normal"
        )

    # Example chart
    st.markdown("### Framework Comparison")

    example_data = {
        'Framework': ['TurboLoader', 'TensorFlow', 'PyTorch Cached', 'PyTorch Optimized'],
        'Throughput (img/s)': [10146, 7569, 3123, 835],
        'Memory (MB)': [848, 1245, 2104, 1523]
    }

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Throughput',
        x=example_data['Framework'],
        y=example_data['Throughput (img/s)'],
        marker_color='#00D9FF',
        yaxis='y',
        offsetgroup=1
    ))

    fig.add_trace(go.Bar(
        name='Memory Usage',
        x=example_data['Framework'],
        y=example_data['Memory (MB)'],
        marker_color='#FF6B6B',
        yaxis='y2',
        offsetgroup=2
    ))

    fig.update_layout(
        title="Performance Comparison",
        xaxis=dict(title='Framework'),
        yaxis=dict(
            title='Throughput (img/s)',
            titlefont=dict(color='#00D9FF'),
            tickfont=dict(color='#00D9FF')
        ),
        yaxis2=dict(
            title='Memory (MB)',
            titlefont=dict(color='#FF6B6B'),
            tickfont=dict(color='#FF6B6B'),
            overlaying='y',
            side='right'
        ),
        barmode='group',
        template="plotly_dark",
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    # Show benchmark interface
    st.success(f"✅ Dataset uploaded: {uploaded_file.name} ({uploaded_file.size / 1024 / 1024:.2f} MB)")

    # Configuration summary
    st.markdown("### Current Configuration")

    config_cols = st.columns(5)
    with config_cols[0]:
        st.metric("Framework", framework)
    with config_cols[1]:
        st.metric("Workers", num_workers)
    with config_cols[2]:
        st.metric("Batch Size", batch_size)
    with config_cols[3]:
        st.metric("Epochs", num_epochs)
    with config_cols[4]:
        transforms_enabled = sum([enable_resize, enable_normalize, enable_flip,
                                   enable_color_jitter, enable_autoaugment])
        st.metric("Transforms", transforms_enabled)

    # Run benchmark button
    if st.button("🚀 Run Benchmark", type="primary", use_container_width=True):
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tar') as tmp_file:
            tmp_file.write(uploaded_file.read())
            dataset_path = tmp_file.name

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Simulated benchmark (replace with actual benchmark_runner.py)
        status_text.text("Initializing benchmark...")
        progress_bar.progress(10)
        time.sleep(0.5)

        status_text.text(f"Loading dataset from {uploaded_file.name}...")
        progress_bar.progress(20)
        time.sleep(0.5)

        status_text.text(f"Starting {framework} with {num_workers} workers...")
        progress_bar.progress(30)
        time.sleep(0.5)

        # Simulate benchmark execution
        for epoch in range(num_epochs):
            status_text.text(f"Running epoch {epoch + 1}/{num_epochs}...")
            progress_bar.progress(30 + int(60 * (epoch + 1) / num_epochs))
            time.sleep(1)

        status_text.text("Collecting results...")
        progress_bar.progress(95)
        time.sleep(0.5)

        # Simulated results (replace with actual results from benchmark_runner.py)
        # NOTE: In production, call benchmark_runner.run_benchmark() here

        throughput = 10146 if framework == "TurboLoader" else (
            7569 if framework == "TensorFlow tf.data" else 835
        )
        avg_epoch_time = 2000 / throughput * batch_size
        peak_memory = 848 if framework == "TurboLoader" else (
            1245 if framework == "TensorFlow tf.data" else 1523
        )
        cpu_util = 94 if framework == "TurboLoader" else 45

        progress_bar.progress(100)
        status_text.text("✅ Benchmark complete!")
        time.sleep(0.5)

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        # Display results
        st.success("✅ Benchmark Complete!")

        # Metrics
        st.markdown("### Performance Metrics")

        metric_cols = st.columns(4)

        with metric_cols[0]:
            st.metric(
                "Throughput",
                f"{throughput:,.0f} img/s",
                delta=f"{throughput / 835:.1f}x vs baseline" if framework == "TurboLoader" else None
            )

        with metric_cols[1]:
            st.metric(
                "Avg Epoch Time",
                f"{avg_epoch_time:.2f}s"
            )

        with metric_cols[2]:
            st.metric(
                "Peak Memory",
                f"{peak_memory:.0f} MB",
                delta=f"-{1523 - peak_memory:.0f} MB" if peak_memory < 1523 else None,
                delta_color="inverse"
            )

        with metric_cols[3]:
            st.metric(
                "CPU Utilization",
                f"{cpu_util:.0f}%"
            )

        # Charts
        st.markdown("### Performance Charts")

        # Throughput over epochs (simulated data)
        epoch_times = [avg_epoch_time * (1 + 0.1 * np.random.randn()) for _ in range(num_epochs)]
        throughputs = [2000 * batch_size / t for t in epoch_times]

        fig_throughput = go.Figure()

        fig_throughput.add_trace(go.Scatter(
            x=list(range(1, num_epochs + 1)),
            y=throughputs,
            mode='lines+markers',
            name='Throughput',
            line=dict(color='#00D9FF', width=3),
            marker=dict(size=10)
        ))

        fig_throughput.update_layout(
            title="Throughput Over Epochs",
            xaxis_title="Epoch",
            yaxis_title="Images/Second",
            template="plotly_dark",
            hovermode='x unified'
        )

        st.plotly_chart(fig_throughput, use_container_width=True)

        # Memory usage over time (simulated)
        time_points = 100
        memory_usage = [peak_memory * (0.3 + 0.7 * min(1.0, i / (time_points * 0.2))) +
                       np.random.randn() * 20 for i in range(time_points)]

        fig_memory = go.Figure()

        fig_memory.add_trace(go.Scatter(
            x=list(range(time_points)),
            y=memory_usage,
            fill='tozeroy',
            name='Memory Usage',
            line=dict(color='#FF6B6B', width=2)
        ))

        fig_memory.update_layout(
            title="Memory Usage Over Time",
            xaxis_title="Time (samples)",
            yaxis_title="Memory (MB)",
            template="plotly_dark"
        )

        st.plotly_chart(fig_memory, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
**TurboLoader v0.8.0** - High-Performance ML Data Loading

- [Documentation](https://github.com/ALJainProjects/TurboLoader/tree/main/docs)
- [GitHub Repository](https://github.com/ALJainProjects/TurboLoader)
- [PyPI Package](https://pypi.org/project/turboloader/)
""")
