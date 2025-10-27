# DeepSeek OCR Batch Processing with Dagster

This guide explains how to use the batch processing system for DeepSeek OCR with Dagster orchestration.

## Overview

The batch processing system provides:
- **Directory-based input**: Process multiple PDFs organized in `<input_dir>/<W*>/W*.pdf` structure
- **Warmup + batching**: Runs 1 warmup PDF, then processes `n` PDFs at a time
- **Progress tracking**: Reports every `m` PDFs with average and recent speed metrics
- **Structured output**: Creates `<output_dir>/<W*>/` directories with markdown, images, and annotated PDFs
- **Dagster integration**: Full configuration and monitoring through Dagster UI

## Files

- `run_dpsk_ocr_pdf_batch.py` - Core batch processing logic
- `dagster_deepseek_ocr.py` - Dagster asset definitions
- `config.py` - Model and processing configuration

## Quick Start

### 1. Standalone Usage (Without Dagster)

```python
from run_dpsk_ocr_pdf_batch import PDFBatchProcessor

processor = PDFBatchProcessor(
    input_dir="/path/to/input",      # Directory with W* subdirs
    output_dir="/path/to/output",    # Output base directory
    batch_size=10,                   # Process 10 PDFs at a time
    progress_interval=10,            # Report every 10 PDFs
    warmup=True,                     # Run warmup PDF first
    cuda_visible_devices="0",        # Use GPU 0
)

metrics = processor.run()
print(f"Processed {metrics['pdfs_processed']} PDFs")
print(f"Average speed: {metrics['avg_speed']:.2f} PDFs/sec")
```

### 2. Dagster Usage (Recommended)

#### Install Dagster

```bash
pip install dagster dagster-webserver
```

#### Launch Dagster UI

```bash
cd DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm
dagster dev -f dagster_deepseek_ocr.py
```

This will start the Dagster UI at `http://localhost:3000`

#### Configure and Run

1. Open the Dagster UI in your browser
2. Navigate to "Assets" → "batch_processed_pdfs"
3. Click "Materialize" button
4. In the configuration panel, set your parameters:

```yaml
config:
  input_dir: "/path/to/input"
  output_dir: "/path/to/output"
  batch_size: 10
  progress_interval: 10
  warmup: true
  max_concurrency: 100
  num_workers: 64
  crop_mode: true
  skip_repeat: true
  cuda_visible_devices: "0"  # Use GPU 0
```

5. Click "Launch Run"

## Input Directory Structure

Your input directory must follow this structure:

```
input_dir/
├── W1/
│   └── W1.pdf
├── W2/
│   └── W2.pdf
├── W123/
│   └── W123.pdf
└── ...
```

The system will:
- Discover all directories matching `W*`
- Find PDFs matching `<W*>/<W*>.pdf` pattern
- Process them in batches

## Output Directory Structure

For each input PDF, the system creates:

```
output_dir/
├── W1/
│   ├── W1.mmd              # Clean markdown output
│   ├── W1_det.mmd          # Markdown with detection metadata
│   ├── W1_layouts.pdf      # Annotated PDF with bounding boxes
│   └── images/             # Extracted images from the PDF
│       ├── 0_0.jpg
│       ├── 0_1.jpg
│       └── ...
├── W2/
│   └── ...
└── ...
```

## Configuration Parameters

### Core Parameters

- **input_dir** (str): Base directory containing W* subdirectories with PDFs
- **output_dir** (str): Output base directory where results will be saved
- **batch_size** (int, default=10): Number of PDFs to process in each batch
  - Higher values = better GPU utilization but more memory usage
  - Recommended: 10-20 for most systems
- **progress_interval** (int, default=10): Report progress every N PDFs
- **warmup** (bool, default=True): Run warmup PDF before batch processing
  - Recommended: True for accurate speed measurements

### Advanced Parameters

- **max_concurrency** (int, default=100): Maximum concurrent sequences for vLLM
  - Adjust based on GPU memory
- **num_workers** (int, default=64): Worker threads for image preprocessing
  - Higher = faster preprocessing, but more CPU usage
- **crop_mode** (bool, default=True): Enable crop mode for better quality
- **skip_repeat** (bool, default=True): Skip pages without proper EOS token
- **cuda_visible_devices** (str, default="0"): GPU selection for CUDA
  - Single GPU: "0" (use GPU 0)
  - Multiple GPUs: "0,1" (use GPUs 0 and 1)
  - Specific GPU: "2" (use only GPU 2)

## Performance Optimization

### Speed Tips

1. **Batch Size**: Start with 10, increase if you have GPU memory
2. **Num Workers**: Set to number of CPU cores for optimal preprocessing
3. **Max Concurrency**: Increase if GPU memory allows (monitor with `nvidia-smi`)
4. **Warmup**: Always enable for accurate benchmarking

### Expected Performance

- **Warmup**: First PDF is slower (model loading, GPU warmup)
- **Batch Processing**: Speed increases after warmup
- **Progress Reports**: Monitor "Recent speed (last 10)" for current performance

Example output:
```
============================================================
Progress Report - 20 PDFs processed
Average speed: 2.35 PDFs/sec
Recent speed (last 10): 2.58 PDFs/sec
Elapsed time: 8.5s
============================================================
```

## Monitoring with Dagster

Dagster provides rich monitoring capabilities:

### Metrics Tracked

- **pdfs_processed**: Total PDFs successfully processed
- **failed_pdfs**: Number of failed PDFs
- **total_time_seconds**: Total processing time
- **average_speed_pdfs_per_sec**: Overall processing speed
- **success_rate_percent**: Percentage of successful PDFs
- **failed_pdf_list**: List of failed PDF paths (if any)

### Viewing Results

1. In Dagster UI, go to "Runs"
2. Click on your run
3. View "Metadata" tab for detailed metrics
4. Check logs for progress reports

## Error Handling

The system continues processing even if individual PDFs fail:

- Failed PDFs are logged and tracked
- Processing continues with remaining PDFs
- Final report includes list of failed PDFs
- Check Dagster logs for error details

## Scheduling (Optional)

To run processing on a schedule:

```python
from dagster import ScheduleDefinition

daily_schedule = ScheduleDefinition(
    name="daily_pdf_processing",
    target=batch_processed_pdfs,
    cron_schedule="0 2 * * *",  # 2 AM daily
)

defs = Definitions(
    assets=[batch_processed_pdfs, processing_summary],
    schedules=[daily_schedule],
)
```

## Troubleshooting

### No PDFs Found

- Check input directory structure matches `<input_dir>/<W*>/W*.pdf`
- Verify directory names start with 'W'
- Ensure PDF names match directory names

### Out of Memory

- Reduce `batch_size`
- Reduce `max_concurrency`
- Monitor GPU memory with `nvidia-smi`

### Slow Processing

- Increase `batch_size` if GPU memory allows
- Increase `num_workers` for faster preprocessing
- Check GPU utilization with `nvidia-smi`

### Import Errors

- Ensure you're in the correct directory
- Check all dependencies are installed
- Verify `config.py` exists and is properly configured

## Example: Processing 100 PDFs

```python
from run_dpsk_ocr_pdf_batch import PDFBatchProcessor

# Configure for high-speed processing
processor = PDFBatchProcessor(
    input_dir="/data/pdfs",
    output_dir="/data/output",
    batch_size=20,           # Process 20 at a time
    progress_interval=10,    # Report every 10 PDFs
    warmup=True,
    max_concurrency=100,
    num_workers=64,
    cuda_visible_devices="0",  # Use GPU 0
)

# Run processing
metrics = processor.run()

# Results
print(f"""
Processing Complete!
- PDFs processed: {metrics['pdfs_processed']}
- Failed: {metrics['failed']}
- Total time: {metrics['total_time']:.1f}s
- Average speed: {metrics['avg_speed']:.2f} PDFs/sec
""")
```

Expected output:
```
Discovered 100 PDFs across 100 W* directories
Running warmup with first PDF...
Warmup complete!
Processing 99 PDFs in batches of 20...

Processing batch 1/5
============================================================
Progress Report - 10 PDFs processed
Average speed: 2.45 PDFs/sec
Recent speed (last 10): 2.45 PDFs/sec
Elapsed time: 4.1s
============================================================

...

============================================================
Processing Complete!
Total PDFs processed: 99
Total time: 42.3s
Average speed: 2.34 PDFs/sec
============================================================
```

## Advanced: Custom Configuration

You can override config.py settings:

```python
processor = PDFBatchProcessor(
    input_dir="/data/pdfs",
    output_dir="/data/output",
    batch_size=15,
    progress_interval=5,
    warmup=True,
    max_concurrency=150,     # Override config.py
    num_workers=128,         # Override config.py
    crop_mode=False,         # Disable cropping
    skip_repeat=False,       # Process all pages
    cuda_visible_devices="1",  # Use GPU 1 instead of GPU 0
)
```

## Multi-GPU Usage

To use multiple GPUs, set `cuda_visible_devices` to a comma-separated list:

```python
processor = PDFBatchProcessor(
    input_dir="/data/pdfs",
    output_dir="/data/output",
    batch_size=20,
    cuda_visible_devices="0,1",  # Use both GPU 0 and GPU 1
)
```

**Note**: The current implementation uses `tensor_parallel_size=1`, so it will only use the first GPU in the list. For true multi-GPU processing, you would need to adjust the vLLM configuration.

## Support

For issues or questions:
1. Check this README
2. Review Dagster logs for detailed error messages
3. Monitor GPU memory and utilization
4. Verify input directory structure

## Performance Benchmarks

Typical performance on a single GPU:
- **Small PDFs** (5-10 pages): 3-4 PDFs/sec
- **Medium PDFs** (20-50 pages): 1-2 PDFs/sec
- **Large PDFs** (100+ pages): 0.5-1 PDFs/sec

Actual performance depends on:
- GPU model and memory
- PDF complexity (images, tables, etc.)
- Batch size and concurrency settings
- CPU cores for preprocessing
