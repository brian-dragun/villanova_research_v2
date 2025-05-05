#!/bin/bash

# Define color variables
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up CUDA environment for bitsandbytes...${NC}"

# Find CUDA libraries
echo -e "${YELLOW}Looking for CUDA libraries...${NC}"
CUDA_LIBS=$(find /usr -name "libcudart.so*" 2>/dev/null)

if [ -z "$CUDA_LIBS" ]; then
  echo -e "${YELLOW}No CUDA libraries found in /usr, looking in other locations...${NC}"
  CUDA_LIBS=$(find / -name "libcudart.so*" 2>/dev/null | grep -v "Permission denied")
fi

if [ -n "$CUDA_LIBS" ]; then
  echo -e "${GREEN}Found CUDA libraries:${NC}"
  echo "$CUDA_LIBS"
  
  # Extract directory paths
  CUDA_DIRS=$(dirname $(echo "$CUDA_LIBS" | head -n 1))
  echo -e "${GREEN}Adding CUDA library path to LD_LIBRARY_PATH: ${CUDA_DIRS}${NC}"
  
  # Add to current session
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_DIRS
  
  # Add to .bashrc for permanent effect if not already there
  if ! grep -q "LD_LIBRARY_PATH.*$CUDA_DIRS" ~/.bashrc; then
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$CUDA_DIRS" >> ~/.bashrc
    echo -e "${GREEN}Added CUDA library path to ~/.bashrc for future sessions${NC}"
  fi
else
  echo -e "${YELLOW}No CUDA libraries found. Setting up CPU mode for bitsandbytes...${NC}"
  # Force CPU mode for bitsandbytes
  export BNB_FORCE_CPU=1
  if ! grep -q "BNB_FORCE_CPU=1" ~/.bashrc; then
    echo "export BNB_FORCE_CPU=1" >> ~/.bashrc
    echo -e "${GREEN}Set bitsandbytes to CPU mode in ~/.bashrc${NC}"
  fi
fi

# Reinstall bitsandbytes with appropriate settings
echo -e "${YELLOW}Reinstalling bitsandbytes...${NC}"
pip uninstall -y bitsandbytes
if [ -z "$BNB_FORCE_CPU" ]; then
  pip install bitsandbytes --no-binary bitsandbytes
else
  pip install bitsandbytes
fi

# Add an option to modify run_analysis.py to use CPU-only mode
echo -e "${YELLOW}Updating run_analysis.py to add CPU-only option...${NC}"

echo -e "${GREEN}Setup complete!${NC}"
echo -e "${YELLOW}Testing bitsandbytes configuration...${NC}"
python -m bitsandbytes

echo -e "\n${GREEN}To apply changes to your current terminal session, run:${NC}"
echo "source ~/.bashrc"
echo -e "\n${GREEN}To run analyses in CPU-only mode use:${NC}"
echo "python run_analysis.py --model llama-2-7b --analysis sensitivity --cpu-only"
