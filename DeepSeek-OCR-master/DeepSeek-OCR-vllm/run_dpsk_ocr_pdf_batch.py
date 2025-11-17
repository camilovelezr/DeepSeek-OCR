import os
import fitz
import img2pdf
import io
import re
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path
from typing import List, Dict, Tuple
import glob

if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ['VLLM_USE_V1'] = '0'
# CUDA_VISIBLE_DEVICES will be set in __init__ based on config

from config import MODEL_PATH, PROMPT, SKIP_REPEAT, MAX_CONCURRENCY, NUM_WORKERS, CROP_MODE

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from deepseek_ocr import DeepseekOCRForCausalLM

from vllm.model_executor.models.registry import ModelRegistry
from vllm import LLM, SamplingParams
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor

ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)


class Colors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    RESET = '\033[0m'


class PDFBatchProcessor:
    """Batch processor for DeepSeek OCR with warmup and progress tracking."""
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        batch_size: int = 10,
        progress_interval: int = 10,
        warmup: bool = True,
        max_concurrency: int = MAX_CONCURRENCY,
        num_workers: int = NUM_WORKERS,
        crop_mode: bool = CROP_MODE,
        skip_repeat: bool = SKIP_REPEAT,
        cuda_visible_devices: str = "0",
        logger=None,
        use_w_pattern: bool = True,
        # Performance Tuning
        quantization: str = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        dpi: int = 144,
        enable_prefix_caching: bool = False,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.progress_interval = progress_interval
        self.warmup = warmup
        self.max_concurrency = max_concurrency
        self.num_workers = num_workers
        self.crop_mode = crop_mode
        self.skip_repeat = skip_repeat
        self.cuda_visible_devices = cuda_visible_devices
        self.logger = logger
        self.use_w_pattern = use_w_pattern
        
        # Performance attributes
        self.quantization = quantization
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dpi = dpi
        self.enable_prefix_caching = enable_prefix_caching

        # Set CUDA_VISIBLE_DEVICES environment variable
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        self.log(f"CUDA_VISIBLE_DEVICES set to: {self.cuda_visible_devices}")
        
        # Initialize vLLM model with performance settings
        self.log("Initializing vLLM engine with the following settings:")
        self.log(f"  Model: deepseek-ai/DeepSeek-OCR")
        self.log(f"  Quantization: {self.quantization}")
        self.log(f"  Tensor Parallel Size: {self.tensor_parallel_size}")
        self.log(f"  GPU Memory Utilization: {self.gpu_memory_utilization}")
        self.log(f"  Enable Prefix Caching: {self.enable_prefix_caching}")

        self.llm = LLM(
            model="deepseek-ai/DeepSeek-OCR",
            quantization=self.quantization,
            dtype="float16" if self.quantization is None else "auto",
            trust_remote_code=True,
            enable_prefix_caching=self.enable_prefix_caching,
            max_model_len=8192,
            swap_space=0,
            max_num_seqs=self.max_concurrency,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            disable_mm_preprocessor_cache=True,
            max_num_batched_tokens=8192,
        )
        
        # Sampling parameters
        logits_processors = [
            NoRepeatNGramLogitsProcessor(
                ngram_size=20,
                window_size=50,
                whitelist_token_ids={128821, 128822}
            )
        ]
        
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=8192,
            logits_processors=logits_processors,
            skip_special_tokens=False,
        )
        
        # Progress tracking
        self.total_pdfs_processed = 0
        self.start_time = None
        self.batch_times = []
        self.failed_pdfs = []
    
    def log(self, message: str, level: str = "info"):
        """Log message to Dagster logger if available, otherwise print."""
        # Strip color codes for Dagster logs
        clean_message = message
        for color in [Colors.RED, Colors.GREEN, Colors.YELLOW, Colors.BLUE, Colors.RESET]:
            clean_message = clean_message.replace(color, '')
        
        if self.logger:
            if level == "error":
                self.logger.error(clean_message)
            elif level == "warning":
                self.logger.warning(clean_message)
            else:
                self.logger.info(clean_message)
        else:
            # Use colored output for console
            print(message, flush=True)
        
    def discover_pdfs(self) -> List[Tuple[str, str]]:
        """
        Discover PDFs based on use_w_pattern setting.
        
        If use_w_pattern=True: Find PDFs matching pattern <input_dir>/<W*>/W*.pdf
        If use_w_pattern=False: Find all .pdf files recursively in input_dir
        
        Returns list of tuples: (pdf_path, output_subdir_name)
        Uses os.scandir() and os.walk() for maximum speed.
        """
        pdf_files = []
        input_dir_str = str(self.input_dir)
        
        if self.use_w_pattern:
            # Original W* pattern matching
            with os.scandir(input_dir_str) as entries:
                for entry in entries:
                    # Check if it's a directory starting with 'W'
                    if entry.is_dir() and entry.name.startswith('W'):
                        w_name = entry.name
                        # Check if W*.pdf exists in this directory
                        pdf_path = os.path.join(entry.path, f"{w_name}.pdf")
                        if os.path.isfile(pdf_path):
                            pdf_files.append((pdf_path, w_name))
        else:
            # Find all PDFs recursively
            for root, dirs, files in os.walk(input_dir_str):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        pdf_path = os.path.join(root, file)
                        # Create output subdirectory name from relative path
                        rel_path = os.path.relpath(pdf_path, input_dir_str)
                        # Use the parent directory structure as the output subdir
                        # e.g., "subdir1/subdir2/file.pdf" -> "subdir1/subdir2"
                        output_subdir = os.path.dirname(rel_path)
                        if not output_subdir:
                            # If PDF is directly in input_dir, use filename without extension
                            output_subdir = Path(file).stem
                        pdf_files.append((pdf_path, output_subdir))
        
        self.log(f"Discovered {len(pdf_files)} PDFs")
        return pdf_files
    
    def pdf_to_images_high_quality(self, pdf_path: str, dpi: int) -> List[Image.Image]:
        """
        Convert PDF to high-quality images efficiently.
        This optimized version creates PIL Images directly from raw pixel data,
        avoiding the slow intermediate step of encoding to PNG.
        """
        images = []
        try:
            pdf_document = fitz.open(pdf_path)
            zoom = dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            
            for page in pdf_document:
                pixmap = page.get_pixmap(matrix=matrix, alpha=False, colorspace=fitz.csRGB)
                # Create a PIL Image directly from the raw pixel data
                img = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
                images.append(img)
            
            pdf_document.close()
        except Exception as e:
            self.log(f"{Colors.RED}Error converting PDF {pdf_path}: {e}{Colors.RESET}", level="error")
            raise
        
        return images
    
    def process_single_image(self, image: Image.Image) -> dict:
        """Process single image for vLLM input."""
        cache_item = {
            "prompt": PROMPT,
            "multi_modal_data": {
                "image": DeepseekOCRProcessor().tokenize_with_images(
                    images=[image],
                    bos=True,
                    eos=True,
                    cropping=self.crop_mode
                )
            },
        }
        return cache_item
    
    def pil_to_pdf_img2pdf(self, pil_images: List[Image.Image], output_path: str):
        """Convert PIL images to PDF."""
        if not pil_images:
            return
        
        image_bytes_list = []
        
        for img in pil_images:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='JPEG', quality=95)
            img_bytes = img_buffer.getvalue()
            image_bytes_list.append(img_bytes)
        
        try:
            pdf_bytes = img2pdf.convert(image_bytes_list)
            with open(output_path, "wb") as f:
                f.write(pdf_bytes)
        except Exception as e:
            self.log(f"{Colors.RED}Error creating PDF: {e}{Colors.RESET}", level="error")
    
    def re_match(self, text: str):
        """Extract reference patterns from text."""
        pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
        matches = re.findall(pattern, text, re.DOTALL)
        
        mathes_image = []
        mathes_table = []
        mathes_other = []
        for a_match in matches:
            if '<|ref|>image<|/ref|>' in a_match[0]:
                mathes_image.append(a_match[0])
            elif '<|ref|>table<|/ref|>' in a_match[0]:
                mathes_table.append(a_match[0])
            else:
                mathes_other.append(a_match[0])
        return matches, mathes_image, mathes_table, mathes_other
    
    def extract_coordinates_and_label(self, ref_text, image_width: int, image_height: int):
        """Extract coordinates and labels from reference text."""
        try:
            label_type = ref_text[1]
            cor_list = eval(ref_text[2])
        except Exception as e:
            print(e)
            return None
        return (label_type, cor_list)
    
    def draw_bounding_boxes(self, image: Image.Image, refs, jdx: int, images_dir: str, tables_dir: str):
        """Draw bounding boxes on image."""
        image_width, image_height = image.size
        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)
        
        overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
        draw2 = ImageDraw.Draw(overlay)
        
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 30)
        except IOError:
            font = ImageFont.load_default()
        
        img_idx = 0
        table_idx = 0
        
        for i, ref in enumerate(refs):
            try:
                result = self.extract_coordinates_and_label(ref, image_width, image_height)
                if result:
                    label_type, points_list = result
                    
                    color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))
                    color_a = color + (20,)
                    
                    for points in points_list:
                        x1, y1, x2, y2 = points
                        
                        x1 = int(x1 / 999 * image_width)
                        y1 = int(y1 / 999 * image_height)
                        x2 = int(x2 / 999 * image_width)
                        y2 = int(y2 / 999 * image_height)
                        
                        if label_type == 'image':
                            try:
                                cropped = image.crop((x1, y1, x2, y2))
                                cropped.save(f"{images_dir}/{jdx}_{img_idx}.jpg")
                            except Exception as e:
                                print(e)
                                pass
                            img_idx += 1
                        elif label_type == 'table':
                            try:
                                cropped = image.crop((x1, y1, x2, y2))
                                cropped.save(f"{tables_dir}/{jdx}_{table_idx}.jpg")
                            except Exception as e:
                                print(e)
                                pass
                            table_idx += 1
                        
                        try:
                            if label_type == 'title':
                                draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                                draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                            else:
                                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                                draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                            
                            id_text = f"{i+1}"
                            text_bbox = draw.textbbox((0, 0), id_text, font=font)
                            text_width = text_bbox[2] - text_bbox[0]
                            text_height = text_bbox[3] - text_bbox[1]
                            
                            text_x = x2 + 5
                            text_y = y1
                            
                            draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height],
                                         fill=(255, 255, 255, 128))
                            draw.text((text_x, text_y), id_text, font=font, fill=color)
                        except:
                            pass
            except:
                continue
        
        img_draw.paste(overlay, (0, 0), overlay)
        return img_draw, img_idx, table_idx
    
    def process_pdf_outputs(self, outputs_list, images_list, pdf_name: str, w_dir: str):
        """Process outputs for a single PDF and save results."""
        output_w_dir = self.output_dir / w_dir
        output_w_dir.mkdir(parents=True, exist_ok=True)
        
        images_dir = output_w_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        tables_dir = output_w_dir / "tables"
        tables_dir.mkdir(exist_ok=True)
        
        mmd_det_path = output_w_dir / f"{pdf_name}_det.mmd"
        mmd_path = output_w_dir / f"{pdf_name}.mmd"
        pdf_out_path = output_w_dir / f"{pdf_name}_layouts.pdf"
        
        contents_det = ''
        contents = ''
        draw_images = []
        
        for jdx, (output, img) in enumerate(zip(outputs_list, images_list)):
            # Extract text from vLLM output
            try:
                content = output.outputs[0].text
            except (AttributeError, IndexError) as e:
                self.log(f"Error extracting text from output {jdx}: {e}", level="error")
                continue
            
            # Handle end-of-sentence token - check for various possible formats
            eos_variants = [
                '<｜end▁of▁sentence｜>',  # Original format
                '<|end_of_sentence|>',     # Alternative format
                '<|endoftext|>',           # Common LLM EOS
                '</s>',                     # Another common EOS
            ]
            
            for eos_token in eos_variants:
                if eos_token in content:
                    content = content.replace(eos_token, '')
                    break
            
            # Skip only if content is actually empty or very short (likely an error)
            if self.skip_repeat and len(content.strip()) < 10:
                self.log(f"Skipping page {jdx} - content too short", level="warning")
                continue
            
            page_num_text = f'\n<--- Page Split --->'
            
            matches_ref, matches_images, matches_tables, mathes_other = self.re_match(content)
            
            modified_content = content
            for i, match in enumerate(matches_ref):
                replacement = f'<|page|>{jdx+1}<|/page|><|id|>{i+1}<|/id|><|det|>'
                modified_match = match[0].replace('<|det|>', replacement)
                modified_content = modified_content.replace(match[0], modified_match)
            
            contents_det += modified_content + f'\n{page_num_text}\n'
            
            image_draw = img.copy()
            result_image, img_count, table_count = self.draw_bounding_boxes(image_draw, matches_ref, jdx, str(images_dir), str(tables_dir))
            draw_images.append(result_image)
            
            for idx, a_match_image in enumerate(matches_images):
                content = content.replace(a_match_image, f'![](images/{jdx}_{idx}.jpg)\n')
            
            for idx, a_match_table in enumerate(matches_tables):
                content = content.replace(a_match_table, f'![](tables/{jdx}_{idx}.jpg)\n')
            
            for idx, a_match_other in enumerate(mathes_other):
                content = content.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:').replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n')
            
            contents += content + f'\n{page_num_text}\n'
        
        # Write outputs
        with open(mmd_det_path, 'w', encoding='utf-8') as f:
            f.write(contents_det)
        
        with open(mmd_path, 'w', encoding='utf-8') as f:
            f.write(contents)
        
        self.pil_to_pdf_img2pdf(draw_images, str(pdf_out_path))
        
        # Validation: Only log warnings for empty outputs
        if len(contents_det) == 0:
            self.log(f"WARNING: {pdf_name}_det.mmd is EMPTY!", level="warning")
        
        if len(contents) == 0:
            self.log(f"WARNING: {pdf_name}.mmd is EMPTY!", level="warning")
        
        if len(draw_images) == 0:
            self.log(f"WARNING: No layout images generated for {pdf_name}_layouts.pdf!", level="warning")
    
    def process_single_pdf(self, pdf_path: str, w_dir: str) -> bool:
        """Process a single PDF and save outputs."""
        try:
            pdf_name = Path(pdf_path).stem
            
            # Convert PDF to images
            images = self.pdf_to_images_high_quality(pdf_path, dpi=self.dpi)
            
            # Preprocess images in parallel
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                batch_inputs = list(executor.map(self.process_single_image, images))
            
            # Run inference
            outputs_list = self.llm.generate(batch_inputs, sampling_params=self.sampling_params)
            
            # Process and save outputs
            self.process_pdf_outputs(outputs_list, images, pdf_name, w_dir)
            
            return True
        except Exception as e:
            self.log(f"{Colors.RED}Error processing {pdf_path}: {e}{Colors.RESET}", level="error")
            self.failed_pdfs.append(pdf_path)
            return False
    
    def process_batch(self, pdf_batch: List[Tuple[str, str]]) -> int:
        """Process a batch of PDFs CONCURRENTLY with optimized batching."""
        batch_start = time.time()
        
        # Phase 1: Convert ALL PDFs in batch to images
        phase1_start = time.time()
        all_images = []
        all_w_dirs = []
        all_pdf_names = []
        
        for pdf_path, w_dir in tqdm(pdf_batch, desc="Phase 1: PDF → Images", leave=False):
            try:
                images = self.pdf_to_images_high_quality(pdf_path, dpi=self.dpi)
                all_images.append(images)
                all_w_dirs.append(w_dir)
                all_pdf_names.append(Path(pdf_path).stem)
            except Exception as e:
                self.log(f"Error loading {pdf_path}: {e}", level="error")
                self.failed_pdfs.append(pdf_path)
        
        phase1_time = time.time() - phase1_start
        num_pdfs = len(all_images)
        
        # Phase 2: Preprocess ALL images from ALL PDFs in parallel
        phase2_start = time.time()
        all_batch_inputs = []
        total_pages = 0
        for images in tqdm(all_images, desc="Phase 2: Preprocess Images", leave=False):
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                batch_inputs = list(executor.map(self.process_single_image, images))
            all_batch_inputs.extend(batch_inputs)
            total_pages += len(images)
        
        phase2_time = time.time() - phase2_start
        
        # Phase 3: Run inference on ALL pages from ALL PDFs in the batch at once
        phase3_start = time.time()
        all_outputs = self.llm.generate(all_batch_inputs, sampling_params=self.sampling_params)
        phase3_time = time.time() - phase3_start
        
        # Phase 4: Split outputs back to individual PDFs and save
        phase4_start = time.time()
        output_idx = 0
        successful = 0
        for images, w_dir, pdf_name in tqdm(zip(all_images, all_w_dirs, all_pdf_names), 
                                            total=len(all_images), 
                                            desc="Phase 4: Save Outputs", 
                                            leave=False):
            num_pages = len(images)
            pdf_outputs = all_outputs[output_idx:output_idx + num_pages]
            output_idx += num_pages
            
            try:
                self.process_pdf_outputs(pdf_outputs, images, pdf_name, w_dir)
                successful += 1
                self.total_pdfs_processed += 1
            except Exception as e:
                self.log(f"Error saving outputs for {pdf_name}: {e}", level="error")
                self.failed_pdfs.append(f"{w_dir}/{pdf_name}.pdf")
        
        phase4_time = time.time() - phase4_start
        batch_time = time.time() - batch_start
        self.batch_times.append(batch_time)
        
        # Report detailed phase metrics
        self._report_batch_metrics(num_pdfs, total_pages, phase1_time, phase2_time, phase3_time, phase4_time, batch_time)
        
        # Report overall progress at intervals
        if self.total_pdfs_processed % self.progress_interval == 0:
            self.report_progress()
        
        return successful
    
    def _report_batch_metrics(self, num_pdfs: int, total_pages: int, 
                              phase1_time: float, phase2_time: float, 
                              phase3_time: float, phase4_time: float, 
                              batch_time: float):
        """Report detailed metrics for each phase of batch processing."""
        self.log(f"\n{Colors.BLUE}{'━' * 80}{Colors.RESET}")
        
        # Phase 1: PDF → Images
        if num_pdfs > 0:
            p1_per_pdf = phase1_time / num_pdfs
            p1_per_min = (60 / p1_per_pdf) if p1_per_pdf > 0 else 0
            p1_per_hour = p1_per_min * 60
            p1_per_day = p1_per_hour * 24
            self.log(f"{Colors.YELLOW}Phase 1: PDF → Images ({num_pdfs} PDFs){Colors.RESET}")
            self.log(f"  Time: {phase1_time:.1f}s | {p1_per_pdf:.2f}s/PDF | {p1_per_min:.0f} PDFs/min | {p1_per_hour:.0f} PDFs/hour | {p1_per_day:,.0f} PDFs/day")
        
        # Phase 2: Preprocess Images
        if num_pdfs > 0:
            p2_per_pdf = phase2_time / num_pdfs
            p2_per_min = (60 / p2_per_pdf) if p2_per_pdf > 0 else 0
            p2_per_hour = p2_per_min * 60
            p2_per_day = p2_per_hour * 24
            self.log(f"{Colors.YELLOW}Phase 2: Preprocess Images ({total_pages} pages){Colors.RESET}")
            self.log(f"  Time: {phase2_time:.1f}s | {p2_per_pdf:.2f}s/PDF | {p2_per_min:.0f} PDFs/min | {p2_per_hour:.0f} PDFs/hour | {p2_per_day:,.0f} PDFs/day")
        
        # Phase 3: Inference
        if num_pdfs > 0:
            p3_per_pdf = phase3_time / num_pdfs
            p3_per_min = (60 / p3_per_pdf) if p3_per_pdf > 0 else 0
            p3_per_hour = p3_per_min * 60
            p3_per_day = p3_per_hour * 24
            self.log(f"{Colors.YELLOW}Phase 3: Inference ({total_pages} pages){Colors.RESET}")
            self.log(f"  Time: {phase3_time:.1f}s | {p3_per_pdf:.2f}s/PDF | {p3_per_min:.0f} PDFs/min | {p3_per_hour:.0f} PDFs/hour | {p3_per_day:,.0f} PDFs/day")
        
        # Phase 4: Save Outputs
        if num_pdfs > 0:
            p4_per_pdf = phase4_time / num_pdfs
            p4_per_min = (60 / p4_per_pdf) if p4_per_pdf > 0 else 0
            p4_per_hour = p4_per_min * 60
            p4_per_day = p4_per_hour * 24
            self.log(f"{Colors.YELLOW}Phase 4: Save Outputs ({num_pdfs} PDFs){Colors.RESET}")
            self.log(f"  Time: {phase4_time:.1f}s | {p4_per_pdf:.2f}s/PDF | {p4_per_min:.0f} PDFs/min | {p4_per_hour:.0f} PDFs/hour | {p4_per_day:,.0f} PDFs/day")
        
        # Batch Total
        if num_pdfs > 0:
            batch_per_pdf = batch_time / num_pdfs
            batch_per_min = (60 / batch_per_pdf) if batch_per_pdf > 0 else 0
            batch_per_hour = batch_per_min * 60
            batch_per_day = batch_per_hour * 24
            self.log(f"\n{Colors.GREEN}Batch Total: {batch_time:.1f}s | {batch_per_pdf:.1f}s/PDF | {batch_per_min:.0f} PDFs/min | {batch_per_hour:.0f} PDFs/hour | {batch_per_day:,.0f} PDFs/day{Colors.RESET}")
        
        self.log(f"{Colors.BLUE}{'━' * 80}{Colors.RESET}\n")
    
    def report_progress(self, force=False):
        """Report processing progress and speed metrics."""
        elapsed = time.time() - self.start_time
        avg_speed = self.total_pdfs_processed / elapsed if elapsed > 0 else 0
        
        # Calculate speed for last 10 PDFs
        recent_batches = self.batch_times[-10:]
        recent_time = sum(recent_batches)
        recent_pdfs = min(10, len(recent_batches) * self.batch_size)
        recent_speed = recent_pdfs / recent_time if recent_time > 0 else 0
        
        # Estimate remaining time
        remaining_pdfs = getattr(self, 'total_pdfs_to_process', 0) - self.total_pdfs_processed
        eta_seconds = remaining_pdfs / avg_speed if avg_speed > 0 else 0
        eta_hours = eta_seconds / 3600
        
        self.log(f"\n{Colors.BLUE}{'='*80}{Colors.RESET}")
        self.log(f"{Colors.GREEN}Progress: {self.total_pdfs_processed:,} / {getattr(self, 'total_pdfs_to_process', '?'):,} PDFs processed{Colors.RESET}")
        self.log(f"{Colors.YELLOW}Average speed: {avg_speed:.3f} PDFs/sec ({avg_speed * 3600:.1f} PDFs/hour){Colors.RESET}")
        self.log(f"{Colors.YELLOW}Recent speed (last 10): {recent_speed:.3f} PDFs/sec ({recent_speed * 3600:.1f} PDFs/hour){Colors.RESET}")
        self.log(f"{Colors.YELLOW}Elapsed: {elapsed/3600:.2f} hours | ETA: {eta_hours:.2f} hours{Colors.RESET}")
        if self.failed_pdfs:
            self.log(f"{Colors.RED}Failed: {len(self.failed_pdfs)} PDFs{Colors.RESET}", level="warning")
        
        # Progress bar for large datasets
        if hasattr(self, 'total_pdfs_to_process') and self.total_pdfs_to_process > 0:
            progress_pct = (self.total_pdfs_processed / self.total_pdfs_to_process) * 100
            bar_length = 50
            filled = int(bar_length * progress_pct / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            self.log(f"{Colors.BLUE}[{bar}] {progress_pct:.2f}%{Colors.RESET}")
        
        self.log(f"{Colors.BLUE}{'='*80}{Colors.RESET}\n")
    
    def run(self) -> Dict:
        """
        Main processing loop with warmup and batching.
        Returns metrics dictionary.
        """
        self.log(f"{Colors.GREEN}Starting PDF batch processing...{Colors.RESET}")
        self.log(f"Input directory: {self.input_dir}")
        self.log(f"Looking for pattern: {self.input_dir}/W*/W*.pdf")
        
        # Discover PDFs
        self.log("Calling discover_pdfs()...")
        try:
            pdf_files = self.discover_pdfs()
            self.log(f"discover_pdfs() returned {len(pdf_files)} files")
        except Exception as e:
            self.log(f"ERROR in discover_pdfs(): {e}", level="error")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}", level="error")
            raise
        
        if not pdf_files:
            self.log(f"{Colors.RED}No PDFs found matching pattern!{Colors.RESET}", level="error")
            return {"pdfs_processed": 0, "failed": 0}
        
        self.start_time = time.time()
        
        # Warmup (process and save, but don't count in timing metrics)
        if self.warmup and len(pdf_files) > 0:
            self.log(f"Running warmup with first PDF...")
            warmup_pdf, warmup_w = pdf_files[0]
            self.process_single_pdf(warmup_pdf, warmup_w)
            self.log(f"Warmup complete! Output saved.")
            
            # Reset timing counters after warmup (output is already saved)
            self.total_pdfs_processed = 0
            self.batch_times = []
            self.start_time = time.time()
            
            # Remove warmup PDF from list to avoid double-processing
            pdf_files = pdf_files[1:]
        
        # Process in batches
        total_pdfs = len(pdf_files)
        self.total_pdfs_to_process = total_pdfs  # Store for progress tracking
        self.log(f"{Colors.GREEN}Processing {total_pdfs:,} PDFs in batches of {self.batch_size}...{Colors.RESET}")
        
        for i in range(0, total_pdfs, self.batch_size):
            batch = pdf_files[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (total_pdfs + self.batch_size - 1) // self.batch_size
            
            self.log(f"\n{Colors.BLUE}Batch {batch_num}/{total_batches} ({i+1}-{min(i+self.batch_size, total_pdfs)} of {total_pdfs:,}){Colors.RESET}")
            self.process_batch(batch)
        
        # Final report
        total_time = time.time() - self.start_time
        avg_speed = self.total_pdfs_processed / total_time if total_time > 0 else 0
        
        self.log(f"\n{Colors.GREEN}{'='*80}{Colors.RESET}")
        self.log(f"{Colors.GREEN}🎉 PROCESSING COMPLETE! 🎉{Colors.RESET}")
        self.log(f"{Colors.GREEN}Total PDFs processed: {self.total_pdfs_processed:,}{Colors.RESET}")
        self.log(f"{Colors.GREEN}Total time: {total_time/3600:.2f} hours ({total_time:.1f}s){Colors.RESET}")
        self.log(f"{Colors.GREEN}Average speed: {avg_speed:.3f} PDFs/sec ({avg_speed * 3600:.1f} PDFs/hour){Colors.RESET}")
        self.log(f"{Colors.GREEN}Throughput: {avg_speed * 86400:,.0f} PDFs/day{Colors.RESET}")
        if self.failed_pdfs:
            self.log(f"{Colors.RED}Failed PDFs ({len(self.failed_pdfs)}): {self.failed_pdfs[:10]}{'...' if len(self.failed_pdfs) > 10 else ''}{Colors.RESET}", level="warning")
        self.log(f"{Colors.GREEN}{'='*80}{Colors.RESET}\n")
        
        return {
            "pdfs_processed": self.total_pdfs_processed,
            "failed": len(self.failed_pdfs),
            "failed_pdfs": self.failed_pdfs,
            "total_time": total_time,
            "avg_speed": avg_speed,
        }


if __name__ == "__main__":
    # Example usage
    processor = PDFBatchProcessor(
        input_dir="/polus2/velezramirezc2/pdf-extraction-code/data_deepseek_large",
        output_dir="/polus2/velezramirezc2/pdf-extraction-code/data_deepseek_output",
        batch_size=50, # best so far: 50
        progress_interval=50,
        warmup=False,
        max_concurrency=1024,
        use_w_pattern=False,
        gpu_memory_utilization=0.9,
        cuda_visible_devices="0",
    )
    
    metrics = processor.run()
    print(f"Final metrics: {metrics}")
