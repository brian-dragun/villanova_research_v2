"""
Utility functions for LLM analysis

This module contains utility functions used across the codebase.
"""

import os
import time
import torch
import logging
from datetime import datetime
from colorama import Fore, Style

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def debug_print(message):
    """Print debug messages in yellow."""
    print(Fore.YELLOW + "[DEBUG] " + message + Style.RESET_ALL)

def log_section(title):
    """Print a section title with highlighting."""
    print(Fore.YELLOW + f"\n🔍 **{title}**" + Style.RESET_ALL)

def log_skip(section):
    """Print a message that a section was skipped."""
    print(Fore.YELLOW + f"\n⏩ **{section} SKIPPED**" + Style.RESET_ALL)

def log_error(message):
    """Print an error message in red."""
    print(Fore.RED + f"❌ {message}" + Style.RESET_ALL)
    logger.error(message)

def log_success(message):
    """Print a success message in green."""
    print(Fore.GREEN + f"✅ {message}" + Style.RESET_ALL)
    logger.info(message)

def ensure_dir(directory):
    """Ensure a directory exists, creating it if necessary."""
    os.makedirs(directory, exist_ok=True)
    return directory

def get_timestamp():
    """Get a formatted timestamp for filenames."""
    return datetime.now().strftime('%Y%m%d-%H%M%S')

class Timer:
    """Simple timer class for profiling."""
    
    def __init__(self, name="Operation"):
        self.name = name
        
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        logger.info(f"{self.name} took {self.interval:.2f} seconds")
        
def get_device():
    """Get the appropriate device (CPU/GPU) for computation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device