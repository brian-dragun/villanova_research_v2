#!/bin/bash

# Define color variables
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}====== CUDA Environment Setup for bitsandbytes and PyTorch ======${NC}"

# Get CUDA version from nvcc if available
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo -e "${GREEN}Detected CUDA version: ${CUDA_VERSION}${NC}"
else
    echo -e "${YELLOW}nvcc not found. Checking for CUDA libraries directly...${NC}"
    CUDA_VERSION=""
fi

# Find CUDA libraries
echo -e "\n${YELLOW}Looking for CUDA libraries...${NC}"
CUDA_PATHS=$(find /usr -name libcuda.so 2>/dev/null)

if [ -z "$CUDA_PATHS" ]; then
    echo -e "${YELLOW}No CUDA libraries found in /usr, looking in other locations...${NC}"
    CUDA_PATHS=$(find / -path /proc -prune -o -path /sys -prune -o -path /dev -prune -o -name libcuda.so -print 2>/dev/null)
fi

if [ -n "$CUDA_PATHS" ]; then
    echo -e "${GREEN}Found CUDA libraries:${NC}"
    echo "$CUDA_PATHS"
    
    # Extract the first directory path containing libcuda.so
    CUDA_LIB_DIR=$(dirname $(echo "$CUDA_PATHS" | head -n 1))
    echo -e "${GREEN}Main CUDA library directory: ${CUDA_LIB_DIR}${NC}"
    
    # Look for specific runtime libraries
    echo -e "\n${YELLOW}Looking for CUDA runtime libraries (libcudart.so)...${NC}"
    CUDART_PATHS=$(find /usr -name libcudart.so* 2>/dev/null)
    if [ -n "$CUDART_PATHS" ]; then
        echo -e "${GREEN}Found CUDA runtime libraries:${NC}"
        echo "$CUDART_PATHS"
        CUDART_DIR=$(dirname $(echo "$CUDART_PATHS" | head -n 1))
        echo -e "${GREEN}CUDA runtime directory: ${CUDART_DIR}${NC}"
    else
        echo -e "${YELLOW}No CUDA runtime libraries found in standard locations.${NC}"
        CUDART_DIR=$CUDA_LIB_DIR
    fi
    
    # Add both directories to LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_LIB_DIR:$CUDART_DIR
    echo -e "\n${GREEN}Updated LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}${NC}"
    
    # Add to .bashrc for permanent effect if not already there
    if ! grep -q "LD_LIBRARY_PATH.*$CUDA_LIB_DIR" ~/.bashrc; then
        echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$CUDA_LIB_DIR:$CUDART_DIR" >> ~/.bashrc
        echo -e "${GREEN}Added CUDA library paths to ~/.bashrc for future sessions${NC}"
    fi
    
    # Check PyTorch CUDA setup
    echo -e "\n${YELLOW}Checking PyTorch CUDA configuration...${NC}"
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU devices: {torch.cuda.device_count()}')" || echo -e "${RED}Error checking PyTorch CUDA configuration${NC}"

    # Reinstall bitsandbytes
    echo -e "\n${YELLOW}Reinstalling bitsandbytes with CUDA support...${NC}"
    pip uninstall -y bitsandbytes
    
    if [ -n "$CUDA_VERSION" ]; then
        echo -e "${YELLOW}Installing bitsandbytes with CUDA version ${CUDA_VERSION}${NC}"
        
        # Remove dots from version string for bitsandbytes format
        BNB_CUDA_VERSION=$(echo $CUDA_VERSION | tr -d '.')
        
        # Try to build from source with the detected CUDA version
        pip install bitsandbytes --no-binary bitsandbytes --verbose
    else
        echo -e "${YELLOW}Installing bitsandbytes (auto-detect CUDA version)${NC}"
        pip install bitsandbytes --no-binary bitsandbytes
    fi
    
    # Make sure the right NVIDIA libraries are accessible
    echo -e "\n${YELLOW}Setting up NVIDIA library environment variables...${NC}"
    export CUDA_HOME=$(dirname $(dirname $(which nvcc 2>/dev/null) 2>/dev/null) 2>/dev/null)
    [ -n "$CUDA_HOME" ] && echo "export CUDA_HOME=$CUDA_HOME" >> ~/.bashrc && echo -e "${GREEN}Set CUDA_HOME to $CUDA_HOME${NC}"
    
else
    echo -e "${RED}No CUDA libraries found. Your system might not have CUDA installed.${NC}"
    echo -e "${YELLOW}If you know CUDA is installed, please provide the path manually.${NC}"
    echo -e "${YELLOW}Otherwise, you can use CPU-only mode with the --cpu-only flag.${NC}"
fi

echo -e "\n${YELLOW}Testing bitsandbytes configuration...${NC}"
python -m bitsandbytes

echo -e "\n${GREEN}To apply changes to your current terminal session, run:${NC}"
echo "source ~/.bashrc"

echo -e "\n${GREEN}To run analyses with optimized CUDA settings:${NC}"
echo "source ~/.bashrc && python run_analysis.py --model llama-2-7b --analysis sensitivity"
