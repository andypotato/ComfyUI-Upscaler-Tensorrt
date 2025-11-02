import os
import folder_paths
import numpy as np
import torch
from comfy.utils import ProgressBar
from .trt_utilities import Engine
from .utilities import download_file, ColoredLogger, get_final_resolutions
import comfy.model_management as mm
import time
import tensorrt
import json
import subprocess
import re
import atexit

logger = ColoredLogger("ComfyUI-Upscaler-Tensorrt")

# --- Hardware/Driver Info Caching ---
# We cache these values so nvidia-smi isn't called 1000 times
global_hw_info = {
    "driver": None,
    "gpu": None
}

def get_driver_version():
    if global_hw_info["driver"]:
        return global_hw_info["driver"]
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], 
                                capture_output=True, text=True, check=True)
        driver_version = result.stdout.strip()
        global_hw_info["driver"] = re.sub(r'[\s.]', '_', driver_version)
        return global_hw_info["driver"]
    except Exception as e:
        logger.error(f"Failed to get NVIDIA driver version: {e}. Using 'unknown_driver'.")
        return "unknown_driver"

def get_gpu_name():
    if global_hw_info["gpu"]:
        return global_hw_info["gpu"]
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=gpu_name', '--format=csv,noheader'], 
                                capture_output=True, text=True, check=True)
        gpu_name = result.stdout.strip()
        global_hw_info["gpu"] = re.sub(r'[^A-Za-z0-9]+', '_', gpu_name)
        return global_hw_info["gpu"]
    except Exception as e:
        logger.error(f"Failed to get NVIDIA GPU name: {e}. Using 'unknown_gpu'.")
        return "unknown_gpu"

# --- Config Loading ---
def load_node_config(config_filename="load_upscaler_config.json"):
    current_dir = os.path.dirname(__file__)
    config_path = os.path.join(current_dir, config_filename)
    default_config = {
        "model": { 
            "options": ["4x-UltraSharp", "RealESRGAN_x4"], 
            "default": "4x-UltraSharp" 
        },
        "precision": { "options": ["fp16", "fp32"], "default": "fp16" }
    }
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Successfully loaded configuration from {config_filename}")
        return config
    except Exception as e:
        logger.warning(f"Failed to load '{config_path}': {e}. Using default fallback.")
        return default_config

LOAD_UPSCALER_NODE_CONFIG = load_node_config()

# --- Global Lock Management ---
# A set of all lock files created by this process
# This ensures that if the process crashes, it cleans up its own locks
active_locks = set()

def cleanup_locks():
    for lock_path in active_locks:
        if os.path.exists(lock_path):
            logger.warning(f"Process exiting, cleaning up lock file: {lock_path}")
            os.remove(lock_path)
atexit.register(cleanup_locks)


# --- The New Merged Node ---

class UpscaleWithTensorRT:
    @classmethod
    def INPUT_TYPES(cls):
        model_config = LOAD_UPSCALER_NODE_CONFIG.get("model", {})
        precision_config = LOAD_UPSCALER_NODE_CONFIG.get("precision", {})
        
        return {
            "required": {
                "images": ("IMAGE",),
                "model": (model_config.get("options", ["4x-UltraSharp"]), 
                          {"default": model_config.get("default", "4x-UltraSharp")}),
                "precision": (precision_config.get("options", ["fp16", "fp32"]), 
                              {"default": precision_config.get("default", "fp16")}),
                "resize_to": (["none", "HD", "FHD", "2k", "4K", "2x", "3x"], 
                              {"default": "none"}),
            }
        }

    RETURN_NAMES = ("IMAGE",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "tensorrt"
    DESCRIPTION = "Upscale images with a self-caching TensorRT engine."

    def get_engine_path(self, model_name, precision, batch, channel, height, width):
        """
        Generates a unique, persistent path for the engine file
        based on hardware and image dimensions.
        """

        static_height = height
        static_width = width

        driver_version_str = get_driver_version()
        gpu_name_str = get_gpu_name()
        
        # Use Comfy's default models directory
        tensorrt_models_dir = os.path.join(folder_paths.models_dir, "tensorrt", "upscaler")
        os.makedirs(tensorrt_models_dir, exist_ok=True)

        # Build the final, unique filename
        filename = (
            f"{model_name}_{precision}_STATIC_{batch}x{channel}x{static_height}x{static_width}"
            f"_{tensorrt.__version__}_GPU_{gpu_name_str}_DRIVER_{driver_version_str}.trt"
        )
        
        return os.path.join(tensorrt_models_dir, filename)

    def build_engine_with_lock(self, onnx_model_path, tensorrt_model_path, precision, batch, height, width):
        lock_path = tensorrt_model_path + ".lock"

        # --- Build Race Lock ---
        if not os.path.exists(lock_path):
            # I am the builder. Create the lock.
            logger.info(f"Creating lock file: {lock_path}")
            try:
                open(lock_path, 'w').close()
                active_locks.add(lock_path)
                
                logger.info(f"Building STATIC TensorRT engine for {tensorrt_model_path}")
                mm.soft_empty_cache()
                s = time.time()
                
                engine = Engine(tensorrt_model_path)
                static_profile = [{"input": [batch, 3, height, width]}]
                
                build_result = engine.build(
                    onnx_path=onnx_model_path,
                    fp16=(precision == "fp16"),
                    input_profile=static_profile,
                )
                
                e = time.time()
                logger.info(f"Time taken to build: {(e-s):.2f} seconds")

                if build_result != 0:
                    raise Exception("Build failed. Check console for errors.")
                
            except Exception as e:
                logger.error(f"Failed to build engine: {e}")
                raise e # Propagate the error
            finally:
                # Build succeeded or failed, release the lock
                if os.path.exists(lock_path):
                    logger.info(f"Releasing lock file: {lock_path}")
                    os.remove(lock_path)
                    active_locks.remove(lock_path)
        else:
            # I am a waiter. Someone else is building.
            logger.warning(f"Engine is being built by another process. Waiting...")
            wait_time = 0
            max_wait = 300 # 5 minutes
            while os.path.exists(lock_path):
                if wait_time > max_wait:
                    raise Exception(f"Waited too long for lock {lock_path}, aborting.")
                time.sleep(5)
                wait_time += 5
                logger.info(f"Still waiting for lock... ({wait_time}s)")
            
            if not os.path.exists(tensorrt_model_path):
                raise Exception(f"Lock was released but engine file {tensorrt_model_path} was not created.")
            
            logger.info("Builder finished. Proceeding.")

    def upscale(self, images, model, precision, resize_to):
        images_bchw = images.permute(0, 3, 1, 2)
        B, C, H, W = images_bchw.shape
        
        # --- 1. Get Engine Path ---
        tensorrt_model_path = self.get_engine_path(model, precision, 1, C, H, W) # Batch size 1
        
        # --- 2. Check if Engine Exists ---
        if not os.path.exists(tensorrt_model_path):
            # Download ONNX model if it's also missing
            onnx_models_dir = os.path.join(folder_paths.models_dir, "onnx")
            os.makedirs(onnx_models_dir, exist_ok=True)
            onnx_model_path = os.path.join(onnx_models_dir, f"{model}.onnx")

            if not os.path.exists(onnx_model_path):
                onnx_model_download_url = f"httpsfs://huggingface.co/yuvraj108c/ComfyUI-Upscaler-Onnx/resolve/main/{model}.onnx"
                logger.info(f"Downloading {onnx_model_download_url}")
                download_file(url=onnx_model_download_url, save_path=onnx_model_path)
            
            # Build the engine using the lock-aware function
            self.build_engine_with_lock(onnx_model_path, tensorrt_model_path, precision, 1, H, W)

        # --- 3. Load Engine ---
        logger.info(f"Loading TensorRT engine: {tensorrt_model_path}")
        mm.soft_empty_cache()
        engine = Engine(tensorrt_model_path)
        engine.load()
        engine.activate()

        # --- 4. Allocate Buffers (Once) ---
        # Note: This engine was built for [1, C, H, W]
        shape_dict = {
            "input": {"shape": (1, C, H, W)},
            "output": {"shape": (1, C, H*4, W*4)},
        }
        engine.allocate_buffers(shape_dict=shape_dict)

        # --- 5. Run Inference Loop (Async) ---
        final_width, final_height = get_final_resolutions(W, H, resize_to)
        logger.info(f"Upscaling {B} images from H:{H}, W:{W} | Final resolution: H:{final_height}, W:{final_width}")

        s = torch.cuda.Stream()
        cudaStream = s.cuda_stream
        
        pbar = ProgressBar(B)
        images_list = list(torch.split(images_bchw, split_size_or_sections=1))
        upscaled_frames = torch.empty((B, C, final_height, final_width), dtype=torch.float32, device=mm.intermediate_device())
        must_resize = (W*4 != final_width) or (H*4 != final_height)

        with torch.cuda.stream(s):
            for i, img in enumerate(images_list):
                result = engine.infer({"input": img}, cudaStream)
                result = result["output"]

                if must_resize:
                    result = torch.nn.functional.interpolate(
                        result, 
                        size=(final_height, final_width),
                        mode='area' # Use 'area' for quality downscaling
                    )
                upscaled_frames[i] = result.to(mm.intermediate_device())
                pbar.update(1)
        
        s.synchronize() # Wait for all async operations to finish

        # --- 6. Cleanup ---
        output = upscaled_frames.permute(0, 2, 3, 1)
        engine.reset()
        mm.soft_empty_cache()

        logger.info(f"Output shape: {output.shape}")
        return (output,)

# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "UpscaleWithTensorRT": UpscaleWithTensorRT,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "UpscaleWithTensorRT": "Upscale (TensorRT) ⚡",
}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']