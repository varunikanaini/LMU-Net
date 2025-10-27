import torch
import argparse
import time
import os
import sys
import logging
from thop import profile, clever_format

# --- Setup Project Path ---
project_path = '/kaggle/working/ARAA-Net'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# --- Custom Module Imports ---
import config
from light_lasa_unet import Light_LASA_Unet
from misc import check_mkdir

# --- Logging Setup ---
def setup_logging(log_dir, filename='benchmark.log'):
    """Sets up a dedicated logger for the benchmark results."""
    log_file = os.path.join(log_dir, filename)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', 
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)])

# --- FPS Benchmarking Function ---
def benchmark_fps(model, device, input_h, input_w, num_warmup=20, num_inference=100):
    """Measures the frames per second of a model."""
    model.to(device).eval()
    dummy_input = torch.randn(1, 3, input_h, input_w, dtype=torch.float32).to(device)

    # --- GPU Warm-up ---
    logging.info(f"Performing GPU warm-up with {num_warmup} inferences...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
    
    # --- Timed Inference ---
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

# --- Model Wrapper for FLOPs Calculation ---
class ModelWrapperForThop(torch.nn.Module):
    """
    Wrapper for models with deep supervision (tuple output) to make them
    compatible with the thop library for FLOPs calculation.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        # Return only the final output for FLOPs calculation
        return outputs[-1]

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description='Benchmark Model FPS, Parameters, and FLOPs')
    # Restrict choice to the model you have
    parser.add_argument('--backbone', type=str, default='mobilenet_v2', choices=['mobilenet_v2'], help="The only supported backbone is 'mobilenet_v2'.")
    parser.add_argument('--input-h', type=int, default=256, help="Input image height for benchmarking.")
    parser.add_argument('--input-w', type=int, default=256, help="Input image width for benchmarking.")
    parser.add_argument('--num-warmup', type=int, default=50, help="Number of warm-up iterations before timing.")
    parser.add_argument('--num-inference', type=int, default=200, help="Number of inference iterations to time.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("ERROR: A GPU is required for meaningful FPS benchmarking.")
        return

    # --- Setup Logging ---
    benchmark_log_dir = os.path.join(config.CKPT_ROOT, 'benchmark_logs')
    check_mkdir(benchmark_log_dir)
    log_filename = f'benchmark_{args.backbone}_{args.input_h}x{args.input_w}.log'
    setup_logging(benchmark_log_dir, filename=log_filename)

    logging.info("=" * 50)
    logging.info(f"Benchmarking Light_LASA_Unet with '{args.backbone}' backbone")
    logging.info(f"Input Resolution: {args.input_h}x{args.input_w}")
    logging.info("=" * 50)

    # --- Instantiate Model ---
    # Assuming num_classes=2 and default kernels for benchmarking purposes
    model = Light_LASA_Unet(num_classes=2, lasa_kernels=config.DEFAULT_ARGS.get('lasa_kernels', [1, 3, 5, 7]))
    model.to(device).eval()
    
    # --- Calculate Parameters and FLOPs ---
    dummy_input = torch.randn(1, 3, args.input_h, args.input_w).to(device)
    
    # Use the wrapper for thop profiling
    model_for_thop = ModelWrapperForThop(model)
    
    logging.info("Calculating model parameters and FLOPs...")
    macs, params = profile(model_for_thop, inputs=(dummy_input,), verbose=False)
    macs_formatted, params_formatted = clever_format([macs, params], "%.3f")

    logging.info("\n--- Model Complexity ---")
    logging.info(f"Total Parameters: {params_formatted}")
    logging.info(f"FLOPs (GFLOPs): {macs_formatted}") # thop calculates MACs, often reported as GFLOPs
    logging.info("------------------------\n")

    # --- Benchmark FPS ---
    fps = benchmark_fps(model, device, args.input_h, args.input_w, args.num_warmup, args.num_inference)
    
    logging.info(f"--- Performance on {torch.cuda.get_device_name(0)} ---")
    logging.info(f"Frames Per Second (FPS): {fps:.2f}")
    logging.info(f"Avg. Inference Time:     {(1000/fps):.2f} ms")
    logging.info("----------------------------------------")
    logging.info("✅ Benchmark completed successfully.")

if __name__ == '__main__':
    main()