#!/usr/bin/env python3
"""
Example script demonstrating batch PDF processing with DeepSeek OCR.

This script shows how to use the PDFBatchProcessor for processing
multiple PDFs organized in W* directories.
"""

from run_dpsk_ocr_pdf_batch import PDFBatchProcessor
from pathlib import Path


def example_basic_usage():
    """Basic usage example."""
    print("=" * 60)
    print("Example 1: Basic Batch Processing")
    print("=" * 60)
    
    processor = PDFBatchProcessor(
        input_dir="/polus2/velezramirezc2/pdf-extraction-code/aithena-pdf/data_deepseek",
        output_dir="/polus2/velezramirezc2/pdf-extraction-code/aithena-pdf/data_deepseek_output",
        batch_size=10,
        progress_interval=10,
        warmup=True,
        cuda_visible_devices="0",
    )
    
    # Run processing
    metrics = processor.run()
    
    # Print results
    print("\n" + "=" * 60)
    print("Processing Results:")
    print("=" * 60)
    print(f"PDFs processed: {metrics['pdfs_processed']}")
    print(f"Failed PDFs: {metrics['failed']}")
    print(f"Total time: {metrics['total_time']:.1f}s")
    print(f"Average speed: {metrics['avg_speed']:.2f} PDFs/sec")
    
    if metrics['failed_pdfs']:
        print(f"\nFailed PDFs:")
        for pdf in metrics['failed_pdfs']:
            print(f"  - {pdf}")


def example_custom_config():
    """Example with custom configuration."""
    print("\n" + "=" * 60)
    print("Example 2: Custom Configuration")
    print("=" * 60)
    
    processor = PDFBatchProcessor(
        input_dir="/path/to/input",
        output_dir="/path/to/output",
        batch_size=20,              # Larger batches
        progress_interval=5,        # More frequent reports
        warmup=True,
        max_concurrency=150,        # Higher concurrency
        num_workers=128,            # More preprocessing workers
        crop_mode=True,
        skip_repeat=True,
        cuda_visible_devices="0",   # Use GPU 0
    )
    
    print("Configuration:")
    print(f"  Batch size: {processor.batch_size}")
    print(f"  Progress interval: {processor.progress_interval}")
    print(f"  Max concurrency: {processor.max_concurrency}")
    print(f"  Num workers: {processor.num_workers}")
    print(f"  Crop mode: {processor.crop_mode}")
    print(f"  Skip repeat: {processor.skip_repeat}")
    print(f"  CUDA visible devices: {processor.cuda_visible_devices}")
    
    # Note: Don't actually run this example unless paths exist
    print("\n(This is a configuration example - not running actual processing)")


def example_check_input_structure():
    """Example showing how to verify input directory structure."""
    print("\n" + "=" * 60)
    print("Example 3: Verify Input Directory Structure")
    print("=" * 60)
    
    input_dir = Path("/polus2/velezramirezc2/pdf-extraction-code/aithena-pdf/data_deepseek")
    
    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}")
        return
    
    # Find W* directories
    w_dirs = sorted(input_dir.glob("W*"))
    
    print(f"Found {len(w_dirs)} W* directories:")
    
    total_pdfs = 0
    for w_dir in w_dirs[:10]:  # Show first 10
        if w_dir.is_dir():
            w_name = w_dir.name
            pdf_path = w_dir / f"{w_name}.pdf"
            
            if pdf_path.exists():
                print(f"  ✓ {w_name}/ → {w_name}.pdf (exists)")
                total_pdfs += 1
            else:
                print(f"  ✗ {w_name}/ → {w_name}.pdf (missing)")
    
    if len(w_dirs) > 10:
        print(f"  ... and {len(w_dirs) - 10} more directories")
    
    print(f"\nTotal PDFs found: {total_pdfs}")


def example_dagster_config():
    """Example showing Dagster configuration format."""
    print("\n" + "=" * 60)
    print("Example 4: Dagster Configuration")
    print("=" * 60)
    
    config_yaml = """
# Copy this configuration to Dagster UI when materializing the asset

config:
  input_dir: "/polus2/velezramirezc2/pdf-extraction-code/aithena-pdf/data_deepseek"
  output_dir: "/polus2/velezramirezc2/pdf-extraction-code/aithena-pdf/data_deepseek_output"
  batch_size: 10
  progress_interval: 10
  warmup: true
  max_concurrency: 100
  num_workers: 64
  crop_mode: true
  skip_repeat: true
  cuda_visible_devices: "0"  # Use GPU 0
"""
    
    print("Dagster Configuration (YAML):")
    print(config_yaml)
    
    print("\nTo use with Dagster:")
    print("1. Run: dagster dev -f dagster_deepseek_ocr.py")
    print("2. Open: http://localhost:3000")
    print("3. Navigate to Assets → batch_processed_pdfs")
    print("4. Click 'Materialize' and paste the config above")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DeepSeek OCR Batch Processing Examples")
    print("=" * 60)
    
    # Run examples
    try:
        # Example 3: Check input structure (safe to run)
        example_check_input_structure()
        
        # Example 2: Show custom configuration
        example_custom_config()
        
        # Example 4: Show Dagster config
        example_dagster_config()
        
        # Example 1: Actual processing (uncomment to run)
        # WARNING: This will actually process PDFs!
        # example_basic_usage()
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: Some examples require valid input directories.")
        print("Modify the paths in this script to match your setup.")
    
    print("\n" + "=" * 60)
    print("Examples Complete!")
    print("=" * 60)
    print("\nFor more information, see BATCH_PROCESSING_README.md")
