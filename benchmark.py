# /kaggle/working/LMU-Net/benchmark.py
import torch
import argparse
import time
import os
import sys
import logging
from thop import profile, clever_format

# --- Setup Project Path ---
project_path = '/kaggle/working/LMU-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# --- Custom Module Imports ---
import config
from light_lasa_unet import Light_LASA_Unet
from misc import check_mkdir

def setup_logging(log_dir, filename='benchmark.log'):
    """Sets up a dedicated logger for the benchmark results."""
    log_file = os.path.join(log_dir, filename)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', 
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)])

def benchmark_fps(model, device, input_h, input_w, num_warmup=20, num_inference=100):
    """Measures the frames per second of a model."""
    model.to(device).eval()
    dummy_input = torch.randn(1, 3, input_h, input_w, dtype=torch.float32).to(device)

    logging.info(f"Performing GPU warm-up with {num_warmup} inferences...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
    
    logging.info(f"Starting benchmark with {num_inference} inferences...")
    torch.cuda.synchronize(device)
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_inference):
            _ = model(dummy_input)
            
    torch.cuda.synchronize(device)
    end_time = time.time()

    total_time = end_time - start_time
    fps = num_inference / total_time
    return fps

class ModelWrapperForThop(torch.nn.Module):
    """Wrapper to handle the tuple output for thop compatibility."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # --- MODIFICATION START ---
        # Unpack the tuple and return only the segmentation part for benchmarking
        seg_outputs, _ = self.model(x)
        return seg_outputs[-1] # Return the final segmentation output
        # --- MODIFICATION END ---

def main():
    parser = argparse.ArgumentParser(description='Benchmark Model FPS, Parameters, and FLOPs')
    parser.add_argument('--backbone', type=str, default='mobilenet_v2', 
                        choices=list(config.BACKBONE_CHANNELS.keys()) + ['mobilenet_v2'], 
                        help="Backbone architecture for the model.")
    parser.add_argument('--input-h', type=int, default=256, help="Input image height for benchmarking.")
    parser.add_argument('--input-w', type=int, default=256, help="Input image width for benchmarking.")
    parser.add_argument('--num-warmup', type=int, default=50, help="Number of warm-up iterations before timing.")
    parser.add_argument('--num-inference', type=int, default=200, help="Number of inference iterations to time.")

    # --- MODIFICATION START ---
    # Add this argument to benchmark the correct model architecture
    parser.add_argument('--multi-task-enabled', action='store_true',
                        help='Set this flag if benchmarking a model with the classification head.')
    # --- MODIFICATION END ---
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("ERROR: A GPU is required for meaningful FPS benchmarking.")
        return

    benchmark_log_dir = os.path.join(config.CKPT_ROOT, 'benchmark_logs')
    check_mkdir(benchmark_log_dir)
    log_filename = f"benchmark_{args.backbone}_{'multitask' if args.multi_task_enabled else 'seg_only'}_{args.input_h}x{args.input_w}.log"
    setup_logging(benchmark_log_dir, filename=log_filename)

    logging.info("=" * 50)
    logging.info(f"Benchmarking Light_LASA_Unet with '{args.backbone}' backbone")
    logging.info(f"Multi-task head enabled: {args.multi_task_enabled}")
    logging.info(f"Input Resolution: {args.input_h}x{args.input_w}")
    logging.info("=" * 50)
    
    # --- MODIFICATION START ---
    # Instantiate the correct version of the model for benchmarking
    num_image_classes = 2 if args.multi_task_enabled else None
    model = Light_LASA_Unet(
        num_classes=2,
        num_image_classes=num_image_classes,
        backbone_name=args.backbone,
        lasa_kernels=config.DEFAULT_ARGS.get('lasa_kernels', [1, 3, 5, 7])
    )
    # --- MODIFICATION END ---
    
    model.to(device).eval()
    
    dummy_input = torch.randn(1, 3, args.input_h, args.input_w).to(device)
    
    model_for_thop = ModelWrapperForThop(model)
    
    logging.info("Calculating model parameters and FLOPs...")
    macs, params = profile(model_for_thop, inputs=(dummy_input,), verbose=False)
    macs_formatted, params_formatted = clever_format([macs, params], "%.3f")

    logging.info("\n--- Model Complexity ---")
    logging.info(f"Total Parameters: {params_formatted}")
    logging.info(f"FLOPs (GFLOPs): {macs_formatted}")
    logging.info("------------------------\n")

    fps = benchmark_fps(model, device, args.input_h, args.input_w, args.num_warmup, args.num_inference)
    
    logging.info(f"--- Performance on {torch.cuda.get_device_name(0)} ---")
    logging.info(f"Frames Per Second (FPS): {fps:.2f}")
    logging.info(f"Avg. Inference Time:     {(1000/fps):.2f} ms")
    logging.info("----------------------------------------")
    logging.info("✅ Benchmark completed successfully.")

if __name__ == '__main__':
    main()