import requests
from tqdm import tqdm
import logging
import sys

class ColoredLogger:
    COLORS = {
        'RED': '\033[91m',
        'GREEN': '\033[92m',
        'YELLOW': '\033[93m',
        'BLUE': '\033[94m',
        'MAGENTA': '\033[95m',
        'RESET': '\033[0m'
    }

    LEVEL_COLORS = {
        'DEBUG': COLORS['BLUE'],
        'INFO': COLORS['GREEN'],
        'WARNING': COLORS['YELLOW'],
        'ERROR': COLORS['RED'],
        'CRITICAL': COLORS['MAGENTA']
    }

    def __init__(self, name="MY-APP"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.app_name = name
        
        # Prevent message propagation to parent loggers
        self.logger.propagate = False
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        
        # Custom formatter class to handle colored components
        class ColoredFormatter(logging.Formatter):
            def format(self, record):
                # Color the level name according to severity
                level_color = ColoredLogger.LEVEL_COLORS.get(record.levelname, '')
                colored_levelname = f"{level_color}{record.levelname}{ColoredLogger.COLORS['RESET']}"
                
                # Color the logger name in blue
                colored_name = f"{ColoredLogger.COLORS['BLUE']}{record.name}{ColoredLogger.COLORS['RESET']}"
                
                # Set the colored components
                record.levelname = colored_levelname
                record.name = colored_name
                
                return super().format(record)
        
        # Create formatter with the new format
        formatter = ColoredFormatter('[%(name)s|%(levelname)s] - %(message)s')
        handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)


    def debug(self, message):
        self.logger.debug(f"{self.COLORS['BLUE']}{message}{self.COLORS['RESET']}")

    def info(self, message):
        self.logger.info(f"{self.COLORS['GREEN']}{message}{self.COLORS['RESET']}")

    def warning(self, message):
        self.logger.warning(f"{self.COLORS['YELLOW']}{message}{self.COLORS['RESET']}")

    def error(self, message):
        self.logger.error(f"{self.COLORS['RED']}{message}{self.COLORS['RESET']}")

    def critical(self, message):
        self.logger.critical(f"{self.COLORS['MAGENTA']}{message}{self.COLORS['RESET']}")

def download_file(url, save_path):
    """
    Download a file from URL with progress bar
    
    Args:
        url (str): URL of the file to download
        save_path (str): Path to save the file as
    """
    GREEN = '\033[92m'
    RESET = '\033[0m'
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as file, tqdm(
        desc=save_path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
        colour='green',
        bar_format=f'{GREEN}{{l_bar}}{{bar}}{RESET}{GREEN}{{r_bar}}{RESET}' 
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

def get_final_resolutions(width, height, resize_to):
    """
    Calculates the final target resolution, preserving the original aspect ratio
    by fitting the 4x upscale into the target resolution "box".
    """
    # 1. Get the 4x upscaled (native) resolution
    native_width = width * 4
    native_height = height * 4
    
    if resize_to == "none":
        return (native_width, native_height)

    # 2. Get the target "box" dimensions
    target_w, target_h = None, None
    if resize_to == "HD":
        target_w, target_h = 1280, 720
    elif resize_to == "FHD":
        target_w, target_h = 1920, 1080
    elif resize_to == "2k":
        target_w, target_h = 2560, 1440
    elif resize_to == "4k":
        target_w, target_h = 3840, 2160
    elif resize_to == "2x":
        target_w, target_h = width * 2, height * 2
    elif resize_to == "3x":
        target_w, target_h = width * 3, height * 3
    else:
        # Fallback for "none" or unknown
        return (native_width, native_height)

    # 3. Calculate the downscale ratio to fit inside the box
    # We must pick the *smaller* ratio to ensure it fits
    ratio_w = target_w / native_width
    ratio_h = target_h / native_height
    ratio = min(ratio_w, ratio_h)

    # 4. Calculate final size
    final_width = round(native_width * ratio)
    final_height = round(native_height * ratio)

    return (final_width, final_height)