# TurboLoader Interactive Benchmark Web App

Interactive Streamlit application for benchmarking TurboLoader against other frameworks.

## Features

- Upload custom datasets (TAR format)
- Compare TurboLoader vs PyTorch vs TensorFlow
- Configure workers, batch size, and transforms
- Real-time performance metrics
- Interactive Plotly charts
- Memory profiling visualization

## Installation

```bash
cd web_app
pip install -r requirements.txt
```

## Usage

### Run the App

```bash
streamlit run app.py
```

The app will open in your browser at http://localhost:8501

### Prepare a Dataset

Create a test TAR archive:

```bash
# Create 1000 synthetic images
python -c "
import numpy as np
from PIL import Image
import tarfile
import os

# Create temp directory
os.makedirs('temp_images', exist_ok=True)

with tarfile.open('test_dataset.tar', 'w') as tar:
    for i in range(1000):
        filename = f'temp_images/{i:06d}.jpg'
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        Image.fromarray(img).save(filename)
        tar.add(filename, arcname=f'{i:06d}.jpg')
        os.remove(filename)

os.rmdir('temp_images')
print('Created test_dataset.tar')
"
```

### Using the App

1. **Upload Dataset**: Click "Upload Dataset (TAR format)" and select your TAR file
2. **Select Framework**: Choose TurboLoader, PyTorch, or TensorFlow
3. **Configure Parameters**:
   - Number of workers (1-32)
   - Batch size (8-256)
   - Number of epochs (1-10)
   - Enable/disable transforms
4. **Run Benchmark**: Click "Run Benchmark" button
5. **View Results**: See metrics and interactive charts

## Extending the App

### Add Real Benchmark Runner

Replace the simulated benchmark in `app.py` with actual benchmark execution:

```python
# In app.py, replace the simulated section with:
from benchmark_runner import run_benchmark

results = run_benchmark(
    framework=framework,
    dataset_path=dataset_path,
    num_workers=num_workers,
    batch_size=batch_size,
    num_epochs=num_epochs,
    transforms={
        'resize': enable_resize,
        'normalize': enable_normalize,
        'flip': enable_flip,
        'color_jitter': enable_color_jitter,
        'autoaugment': enable_autoaugment
    }
)
```

### Add Custom Charts

Add new visualizations in `app.py`:

```python
# CPU utilization chart
fig_cpu = go.Figure()
fig_cpu.add_trace(go.Scatter(
    x=results['time_points'],
    y=results['cpu_timeline'],
    mode='lines',
    name='CPU %'
))
st.plotly_chart(fig_cpu, use_container_width=True)
```

## Architecture

```
web_app/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── benchmark_runner.py       # Benchmark execution (to be implemented)
├── memory_profiler.py        # Memory tracking (to be implemented)
├── chart_generator.py        # Chart utilities (to be implemented)
└── README.md                 # This file
```

## Dependencies

- **streamlit** - Web app framework
- **plotly** - Interactive charts
- **pandas** - Data manipulation
- **numpy** - Numerical operations
- **psutil** - System monitoring
- **turboloader** - Benchmarking target
- **torch** - PyTorch comparison
- **pillow** - Image processing

## Deployment

### Local Deployment

```bash
streamlit run app.py --server.port 8501
```

### Cloud Deployment (Streamlit Cloud)

1. Push code to GitHub repository
2. Visit https://share.streamlit.io
3. Connect repository and select `web_app/app.py`
4. Deploy

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'turboloader'"

**Solution:** Install turboloader:
```bash
pip install turboloader
```

### Issue: Port 8501 already in use

**Solution:** Use a different port:
```bash
streamlit run app.py --server.port 8502
```

### Issue: Upload size too large

**Solution:** Increase upload limit in `~/.streamlit/config.toml`:
```toml
[server]
maxUploadSize = 1000  # MB
```

## Future Enhancements

- [ ] Implement actual benchmark_runner.py
- [ ] Add memory profiler integration
- [ ] Support for multiple datasets comparison
- [ ] Export results to CSV/JSON
- [ ] Historical results tracking
- [ ] GPU benchmarking support
- [ ] Custom transform configuration UI
- [ ] Real-time progress streaming

## License

MIT License - see main repository LICENSE file.

## Support

- [TurboLoader Documentation](../docs/)
- [GitHub Issues](https://github.com/ALJainProjects/TurboLoader/issues)
- [Main Repository](https://github.com/ALJainProjects/TurboLoader)
