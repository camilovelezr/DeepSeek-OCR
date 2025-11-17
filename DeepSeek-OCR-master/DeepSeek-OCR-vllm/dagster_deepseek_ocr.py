"""
Dagster definitions for DeepSeek OCR batch processing.

This module provides Dagster assets for processing PDFs with DeepSeek OCR,
including configuration, monitoring, and metrics tracking.
"""

from dagster import (
    asset,
    AssetExecutionContext,
    Config,
    MaterializeResult,
    MetadataValue,
    Definitions,
)
from pydantic import Field
from typing import Optional
import sys
from pathlib import Path

# Add the current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from run_dpsk_ocr_pdf_batch import PDFBatchProcessor


class DeepSeekOCRConfig(Config):
    """Configuration for DeepSeek OCR batch processing."""
    
    input_dir: str = Field(
        description="Base directory containing W* subdirectories with PDFs (e.g., /path/to/input)",
        default="/polus2/velezramirezc2/pdf-extraction/downloaded_pdfs"
    )
    
    output_dir: str = Field(
        description="Output base directory where W* subdirectories will be created",
        default="/polus2/velezramirezc2/pdf-extraction/deepseek_ocr_output_v1"
    )
    
    batch_size: int = Field(
        description="Number of PDFs to process in each batch (n)",
        default=10,
        ge=1,
        le=100
    )
    
    progress_interval: int = Field(
        description="Report progress every m PDFs",
        default=10,
        ge=1
    )
    
    warmup: bool = Field(
        description="Whether to run a warmup PDF before batch processing",
        default=True
    )
    
    max_concurrency: int = Field(
        description="Maximum concurrent sequences for vLLM",
        default=100,
        ge=1
    )
    
    num_workers: int = Field(
        description="Number of worker threads for image preprocessing",
        default=64,
        ge=1
    )
    
    crop_mode: bool = Field(
        description="Enable crop mode for image processing",
        default=True
    )
    
    skip_repeat: bool = Field(
        description="Skip pages with repeated content (no EOS token)",
        default=True
    )
    
    cuda_visible_devices: str = Field(
        description="CUDA_VISIBLE_DEVICES setting (e.g., '0' for GPU 0, '0,1' for GPUs 0 and 1)",
        default="0"
    )
    
    use_w_pattern: bool = Field(
        description="If True, use W* directory pattern (<input_dir>/<W*>/W*.pdf). If False, find all PDFs recursively.",
        default=True
    )

    # Performance Tuning
    quantization: Optional[str] = Field(
        description="Quantization method (e.g., 'awq'). If None, uses float16.",
        default=None
    )
    
    tensor_parallel_size: int = Field(
        description="Number of GPUs to use for tensor parallelism.",
        default=1,
        ge=1
    )

    gpu_memory_utilization: float = Field(
        description="GPU memory utilization ratio for vLLM.",
        default=0.90,
        ge=0.1,
        le=1.0
    )

    dpi: int = Field(
        description="DPI for rendering PDF pages to images.",
        default=144,
        ge=72,
        le=300
    )

    enable_prefix_caching: bool = Field(
        description="Enable prefix caching in vLLM. Recommended to be False for OCR.",
        default=False
    )


@asset(
    name="batch_processed_pdfs",
    description="Process PDFs from input directory using DeepSeek OCR in batches",
    compute_kind="deepseek-ocr",
    group_name="pdf_processing",
)
def batch_processed_pdfs(
    context: AssetExecutionContext,
    config: DeepSeekOCRConfig,
) -> MaterializeResult:
    """
    Process all PDFs from input_dir in batches with DeepSeek OCR.
    
    This asset:
    1. Discovers PDFs based on use_w_pattern setting:
       - If True: Finds PDFs matching pattern <input_dir>/<W*>/W*.pdf
       - If False: Finds all .pdf files recursively in input_dir
    2. Runs a warmup PDF (optional)
    3. Processes remaining PDFs in batches of size n
    4. Reports progress every m PDFs
    5. Outputs to appropriate subdirectories with markdown, images, tables, and annotated PDFs
    
    Args:
        context: Dagster execution context
        config: Configuration for batch processing
        
    Returns:
        MaterializeResult with processing metrics
    """
    context.log.info("Starting DeepSeek OCR batch processing")
    context.log.info(f"Input directory: {config.input_dir}")
    context.log.info(f"Output directory: {config.output_dir}")
    context.log.info(f"Batch size: {config.batch_size}")
    context.log.info(f"Progress interval: {config.progress_interval}")
    
    # Initialize processor with Dagster logger
    processor = PDFBatchProcessor(
        input_dir=config.input_dir,
        output_dir=config.output_dir,
        batch_size=config.batch_size,
        progress_interval=config.progress_interval,
        warmup=config.warmup,
        max_concurrency=config.max_concurrency,
        num_workers=config.num_workers,
        crop_mode=config.crop_mode,
        skip_repeat=config.skip_repeat,
        cuda_visible_devices=config.cuda_visible_devices,
        logger=context.log,  # Pass Dagster logger for real-time log streaming
        use_w_pattern=config.use_w_pattern,
        
        # Pass performance tuning configs
        quantization=config.quantization,
        tensor_parallel_size=config.tensor_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        dpi=config.dpi,
        enable_prefix_caching=config.enable_prefix_caching,
    )
    
    # Run processing
    context.log.info("Running batch processor...")
    metrics = processor.run()
    
    # Log results
    context.log.info(f"Processing complete!")
    context.log.info(f"PDFs processed: {metrics['pdfs_processed']}")
    context.log.info(f"Failed PDFs: {metrics['failed']}")
    context.log.info(f"Total time: {metrics['total_time']:.1f}s")
    context.log.info(f"Average speed: {metrics['avg_speed']:.2f} PDFs/sec")
    
    # Prepare metadata
    metadata = {
        "pdfs_processed": MetadataValue.int(metrics["pdfs_processed"]),
        "failed_pdfs": MetadataValue.int(metrics["failed"]),
        "total_time_seconds": MetadataValue.float(metrics["total_time"]),
        "average_speed_pdfs_per_sec": MetadataValue.float(metrics["avg_speed"]),
        "input_directory": MetadataValue.text(config.input_dir),
        "output_directory": MetadataValue.text(config.output_dir),
        "batch_size": MetadataValue.int(config.batch_size),
        "progress_interval": MetadataValue.int(config.progress_interval),
        "cuda_visible_devices": MetadataValue.text(config.cuda_visible_devices),
        "use_w_pattern": MetadataValue.bool(config.use_w_pattern),
        "quantization": MetadataValue.text(str(config.quantization)),
        "tensor_parallel_size": MetadataValue.int(config.tensor_parallel_size),
        "gpu_memory_utilization": MetadataValue.float(config.gpu_memory_utilization),
        "dpi": MetadataValue.int(config.dpi),
        "enable_prefix_caching": MetadataValue.bool(config.enable_prefix_caching),
    }
    
    # Add failed PDFs list if any
    if metrics["failed_pdfs"]:
        metadata["failed_pdf_list"] = MetadataValue.md(
            "\n".join([f"- {pdf}" for pdf in metrics["failed_pdfs"]])
        )
    
    # Calculate success rate
    total_attempted = metrics["pdfs_processed"] + metrics["failed"]
    if total_attempted > 0:
        success_rate = (metrics["pdfs_processed"] / total_attempted) * 100
        metadata["success_rate_percent"] = MetadataValue.float(success_rate)
    
    return MaterializeResult(
        metadata=metadata,
    )




# Define the Dagster definitions
defs = Definitions(
    assets=[batch_processed_pdfs],
)


if __name__ == "__main__":
    # For testing purposes
    print("Dagster definitions loaded successfully!")
    print(f"Assets: {[asset.key for asset in defs.get_asset_graph().all_asset_keys]}")
